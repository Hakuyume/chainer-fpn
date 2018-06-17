import numpy as np

import chainer
from chainer.backends import cuda

from chainercv import transforms
from chainercv import utils

from fpn import exp_clip


class FasterRCNN(chainer.Chain):

    _min_size = 800
    _max_size = 1333
    _stride = 32
    _std = (0.1, 0.2)

    def __init__(self, extractor, rpn, head):
        super().__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

        self.use_preset('visualize')

    def __call__(self, x):
        hs = self.extractor(x)
        rpn_locs, rpn_confs = self.rpn(hs)
        rois, roi_indices = self.rpn.decode(
            rpn_locs, rpn_confs, x.shape, [h.shape for h in hs])
        locs, confs = self.head(hs, rois, roi_indices)
        return rpn_locs, rpn_confs, rois, roi_indices, locs, confs

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.5
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.5
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def predict(self, imgs):
        sizes = [img.shape[1:] for img in imgs]
        x, scales = self._prepare(imgs)

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            _, _, rois, roi_indices, locs, confs = self(x)

        bboxes = []
        labels = []
        scores = []
        for i in range(x.shape[0]):
            bbox, label, score = self._decode(
                rois, roi_indices, locs, confs, i, scales[i], sizes[i])

            bboxes.append(cuda.to_cpu(bbox))
            labels.append(cuda.to_cpu(label))
            scores.append(cuda.to_cpu(score))

        return bboxes, labels, scores

    def _prepare(self, imgs):
        scales = []
        resized_imgs = []
        for img in imgs:
            _, H, W = img.shape
            scale = self._min_size / min(H, W)
            if scale * max(H, W) > self._max_size:
                scale = self._max_size / max(H, W)
            scales.append(scale)
            H, W = int(H * scale), int(W * scale)
            img = transforms.resize(img, (H, W))
            img -= self.extractor.mean
            resized_imgs.append(img)

        size = np.array([img.shape[1:] for img in resized_imgs]).max(axis=0)
        size = (np.ceil(size / self._stride) * self._stride).astype(int)
        x = np.zeros((len(imgs), 3, size[0], size[1]), dtype=np.float32)
        for i, img in enumerate(resized_imgs):
            _, H, W = img.shape
            x[i, :, :H, :W] = img

        x = self.xp.array(x)
        return x, scales

    def _decode(self, rois, roi_indices, locs, confs, i, scale, size):
        bbox = []
        score = []
        for l in range(len(self.extractor.scales) - 1):
            mask = roi_indices[l] == i
            roi_l = rois[l][mask]
            loc_l = locs[l].array[mask]
            conf_l = confs[l].array[mask]

            bbox_l = self.xp.broadcast_to(
                roi_l[:, None], loc_l.shape) / scale
            bbox_l[:, :, 2:] -= bbox_l[:, :, :2]
            bbox_l[:, :, :2] += bbox_l[:, :, 2:] / 2
            bbox_l[:, :, :2] += loc_l[:, :, :2] * \
                bbox_l[:, :, 2:] * self._std[0]
            bbox_l[:, :, 2:] *= self.xp.exp(
                self.xp.minimum(loc_l[:, :, 2:] * self._std[1], exp_clip))
            bbox_l[:, :, :2] -= bbox_l[:, :, 2:] / 2
            bbox_l[:, :, 2:] += bbox_l[:, :, :2]

            bbox_l[:, :, :2] = self.xp.maximum(bbox_l[:, :, :2], 0)
            bbox_l[:, :, 2:] = self.xp.minimum(
                bbox_l[:, :, 2:], self.xp.array(size))

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
