import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
import chainer.links as L

import chainercv
from chainercv import transforms
from chainercv import utils

from roi_align_2d import roi_align_2d

_clip = np.log(1000 / 16)


class FasterRCNNFPNResNet101(chainer.Chain):

    _mean = np.array((122.7717, 115.9465, 102.9801))[:, None, None]
    _min_size = 800
    _max_size = 1333
    _stride = 32
    _std = (0.1, 0.2)

    def __init__(self, n_fg_class):
        super().__init__()
        with self.init_scope():
            self.extractor = FPNResNet101()
            self.rpn = RPN(self.extractor.scales)
            self.head = Head(n_fg_class + 1, self.extractor.scales[:-1])

        self.use_preset('visualize')

    def __call__(self, x):
        hs = self.extractor(x)
        rpn_locs, rpn_confs = self.rpn(hs)
        rois, roi_indices = self.rpn.decode(
            rpn_locs, rpn_confs, x.shape, [h.shape for h in hs])
        locs, confs = self.head(hs[:-1], rois, roi_indices)
        return rpn_locs, rpn_confs, rois, roi_indices, locs, confs

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.5
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.5
            self.score_thresh = 0.001
        else:
            raise ValueError('preset must be visualize or evaluate')

    def predict(self, imgs):
        sizes_orig = [img.shape[1:] for img in imgs]
        x, sizes = self._prepare(imgs)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            _, _, rois, roi_indices, locs, confs = self(x)

        bboxes = []
        labels = []
        scores = []
        for i in range(x.shape[0]):
            bbox, label, score = self._decode(
                i, rois, roi_indices, locs, confs)

            bbox = cuda.to_cpu(bbox)
            label = cuda.to_cpu(label)
            score = cuda.to_cpu(score)

            bbox = transforms.resize_bbox(bbox, sizes[i], sizes_orig[i])

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores

    def _prepare(self, imgs):
        sizes = []
        resized_imgs = []
        for img in imgs:
            _, H, W = img.shape
            scale = self._min_size / min(H, W)
            if scale * max(H, W) > self._max_size:
                scale = self._max_size / max(H, W)
            H, W = int(H * scale), int(W * scale)
            sizes.append((H, W))
            img = transforms.resize(img, (H, W))
            img -= self._mean
            resized_imgs.append(img)

        size = np.array([img.shape[1:] for img in resized_imgs]).max(axis=0)
        size = (np.ceil(size / self._stride) * self._stride).astype(int)
        x = np.zeros((len(imgs), 3, size[0], size[1]), dtype=np.float32)
        for i, img in enumerate(resized_imgs):
            _, H, W = img.shape
            x[i, :, :H, :W] = img

        x = self.xp.array(x)
        return x, sizes

    def _decode(self, i, rois, roi_indices, locs, confs):
        bbox = []
        score = []
        for l, scale in enumerate(self.extractor.scales[:-1]):
            mask = roi_indices[l] == i
            roi_l = rois[l][mask]
            loc_l = locs[l].array[mask]
            conf_l = confs[l].array[mask]

            bbox_l = self.xp.broadcast_to(
                roi_l[:, None], loc_l.shape).copy()
            bbox_l[:, :, 2:] -= bbox_l[:, :, :2] + 1
            bbox_l[:, :, :2] += bbox_l[:, :, 2:] / 2
            bbox_l[:, :, :2] += loc_l[:, :, :2] * \
                bbox_l[:, :, 2:] * self._std[0]
            bbox_l[:, :, 2:] *= self.xp.exp(
                self.xp.minimum(loc_l[:, :, 2:] * self._std[1], _clip))
            bbox_l[:, :, :2] -= bbox_l[:, :, 2:] / 2
            bbox_l[:, :, 2:] += bbox_l[:, :, :2] - 1

            conf_l = self.xp.exp(conf_l)
            score_l = conf_l / self.xp.sum(conf_l, axis=1, keepdims=True)

            bbox.append(bbox_l)
            score.append(score_l)

        bbox = self.xp.vstack(bbox)
        score = self.xp.vstack(score)
        return self._suppress(bbox, score)

    def _suppress(self, raw_bbox, raw_score):
        bbox = []
        label = []
        score = []
        for l in range(raw_score.shape[1] - 1):
            bbox_l = raw_bbox[:, l + 1]
            score_l = raw_score[:, l + 1]

            mask = score_l >= self.score_thresh
            bbox_l = bbox_l[mask]
            score_l = score_l[mask]

            indices = utils.non_maximum_suppression(
                bbox_l, self.nms_thresh, score_l)
            bbox_l = bbox_l[indices]
            score_l = score_l[indices]

            bbox.append(bbox_l)
            label.append(self.xp.array((l,) * len(bbox_l)))
            score.append(score_l)

        bbox = self.xp.vstack(bbox).astype(np.float32)
        label = self.xp.hstack(label).astype(np.int32)
        score = self.xp.hstack(score).astype(np.float32)
        return bbox, label, score


class FPNResNet101(chainer.Chain):

    scales = (1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64)

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.resnet = chainercv.links.ResNet101(n_class=1, arch='fb')
        self.resnet.pick = ('res2', 'res3', 'res4', 'res5')
        self.resnet.remove_unused()
        self.resnet.pool1 = lambda x: F.max_pooling_2d(
            x, 3, stride=2, pad=1, cover_all=False)
        for link in self.resnet.links():
            if hasattr(link, 'bn'):
                size = len(link.bn.avg_mean)
                del link.bn
                with link.init_scope():
                    link.bn = AffineChannel(size)

        for i in range(2, 5 + 1):
            self.add_link('inner{}'.format(i), L.Convolution2D(256, 1))
            self.add_link('outer{}'.format(i), L.Convolution2D(256, 3, pad=1))

    def __call__(self, x):
        h2, h3, h4, h5 = self.resnet(x)

        h5 = self.inner5(h5)
        h4 = _upsample(h5) + self.inner4(h4)
        h3 = _upsample(h4) + self.inner3(h3)
        h2 = _upsample(h3) + self.inner2(h2)

        h2 = self.outer2(h2)
        h3 = self.outer3(h3)
        h4 = self.outer4(h4)
        h5 = self.outer5(h5)
        h6 = F.max_pooling_2d(h5, 1, stride=2, cover_all=False)
        return h2, h3, h4, h5, h6


class RPN(chainer.Chain):

    _anchor_size = 32
    _anchor_ratios = (0.5, 1, 2)
    _nms_thresh = 0.7
    _nms_limit_pre = 1000
    _nms_limit_post = 1000
    _canonical_scale = 224

    def __init__(self, scales):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(256, 3, pad=1)
            self.loc = L.Convolution2D(len(self._anchor_ratios) * 4, 1)
            self.conf = L.Convolution2D(len(self._anchor_ratios), 1)

        self._scales = scales

    def __call__(self, hs):
        locs = []
        confs = []
        for h in hs:
            h = F.relu(self.conv(h))

            loc = self.loc(h)
            loc = F.transpose(loc, (0, 2, 3, 1))
            loc = F.reshape(loc, (loc.shape[0], -1, 4))
            locs.append(loc)

            conf = self.conf(h)
            conf = F.transpose(conf, (0, 2, 3, 1))
            conf = F.reshape(conf, (conf.shape[0], -1))
            confs.append(conf)

        return locs, confs

    def decode(self, locs, confs, x_shape, h_shapes):
        anchors = []
        for l, (_, _, H, W) in enumerate(h_shapes):
            v, u, ar = np.meshgrid(
                np.arange(W), np.arange(H), self._anchor_ratios)
            w = np.round(1 / np.sqrt(ar) / self._scales[l])
            h = np.round(w * ar)
            anchor = np.stack((u, v, h, w)).reshape((4, -1)).transpose()
            anchor[:, :2] = (anchor[:, :2] + 0.5) / self._scales[l]
            anchor[:, 2:] *= (self._anchor_size << l) * self._scales[l]
            anchors.append(self.xp.array(anchor))

        rois = [[] for _ in range(len(self._scales) - 1)]
        roi_indices = [[] for _ in range(len(self._scales) - 1)]
        for i in range(x_shape[0]):
            roi = []
            conf = []
            for l in range(len(self._scales)):
                loc_l = locs[l].array[i]
                conf_l = confs[l].array[i]

                roi_l = anchors[l].copy()
                roi_l[:, :2] += loc_l[:, :2] * roi_l[:, 2:]
                roi_l[:, 2:] *= self.xp.exp(
                    self.xp.minimum(loc_l[:, 2:], _clip))
                roi_l[:, :2] -= roi_l[:, 2:] / 2
                roi_l[:, 2:] += roi_l[:, :2] - 1

                order = self.xp.argsort(-conf_l)[:self._nms_limit_pre]
                roi_l = roi_l[order]
                conf_l = conf_l[order]

                roi_l[:, :2] = self.xp.maximum(roi_l[:, :2], 0)
                roi_l[:, 2:] = self.xp.minimum(
                    roi_l[:, 2:], self.xp.array(x_shape[2:]) - 1)

                mask = (roi_l[:, 2:] - roi_l[:, :2] + 1 > 0).all(axis=1)
                roi_l = roi_l[mask]
                conf_l = conf_l[mask]

                roi_l[:, 2:] += 1
                indices = utils.non_maximum_suppression(
                    roi_l, self._nms_thresh, limit=self._nms_limit_post)
                roi_l[:, 2:] -= 1
                roi_l = roi_l[indices]
                conf_l = conf_l[indices]

                roi.append(roi_l)
                conf.append(conf_l)

            roi = self.xp.vstack(roi).astype(np.float32)
            conf = self.xp.hstack(conf).astype(np.float32)

            order = self.xp.argsort(-conf)[:self._nms_limit_post]
            roi = roi[order]

            size = self.xp.sqrt(self.xp.prod(
                roi[:, 2:] - roi[:, :2] + 1, axis=1))
            level = self.xp.floor(self.xp.log2(
                size / self._canonical_scale + 1e-6)).astype(np.int32)
            level = self.xp.clip(
                level + len(self._scales) // 2, 0, len(self._scales) - 2)

            for l in range(len(self._scales) - 1):
                roi_l = roi[level == l]
                rois[l].append(roi_l)
                roi_indices[l].append(self.xp.array((i,) * len(roi_l)))

        for l in range(len(self._scales) - 1):
            rois[l] = self.xp.vstack(rois[l]).astype(np.float32)
            roi_indices[l] = self.xp.hstack(roi_indices[l]).astype(np.int32)

        return rois, roi_indices


class Head(chainer.Chain):

    _roi_size = 7
    _roi_sample_ratio = 2

    def __init__(self, n_class, scales):
        super().__init__()
        with self.init_scope():
            self.fc6 = L.Linear(1024)
            self.fc7 = L.Linear(1024)
            self.loc = L.Linear(n_class * 4)
            self.conf = L.Linear(n_class)

        self._n_class = n_class
        self._scales = scales

    def __call__(self, hs, rois, roi_indices):
        locs = []
        confs = []
        for l, h in enumerate(hs):
            if len(rois[l]) == 0:
                locs.append(chainer.Variable(
                    self.xp.empty((0, self._n_class, 4), dtype=np.float32)))
                confs.append(chainer.Variable(
                    self.xp.empty((0, self._n_class), dtype=np.float32)))
                continue

            roi_iltrb = self.xp.hstack(
                (roi_indices[l][:, None], rois[l][:, [1, 0, 3, 2]])) \
                .astype(np.float32)
            h = roi_align_2d(
                h, roi_iltrb,
                self._roi_size, self._roi_size,
                self._scales[l], self._roi_sample_ratio)

            h = F.reshape(h, (h.shape[0], -1))
            h = F.relu(self.fc6(h))
            h = F.relu(self.fc7(h))

            loc = self.loc(h)
            loc = F.reshape(loc, (loc.shape[0], -1, 4))
            locs.append(loc)

            conf = self.conf(h)
            confs.append(conf)
        return locs, confs


class AffineChannel(chainer.Link):

    def __init__(self, size):
        super().__init__()
        self.gamma = np.empty(size, dtype=np.float32)
        self.beta = np.empty(size, dtype=np.float32)
        self.register_persistent('gamma')
        self.register_persistent('beta')

    def __call__(self, x):
        return x * self.gamma[:, None, None] + self.beta[:, None, None]


def _upsample(x):
    return F.unpooling_2d(x, 2, cover_all=False)
