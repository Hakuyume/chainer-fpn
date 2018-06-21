import numpy as np

import chainer
from chainer.backends import cuda
import chainer.functions as F
from chainer import initializers
import chainer.links as L

from chainercv import utils

from fpn import exp_clip
from fpn.roi_align_2d import roi_align_2d


class Head(chainer.Chain):

    _canonical_scale = 224
    _roi_size = 7
    _roi_sample_ratio = 2
    std = (0.1, 0.2)

    def __init__(self, n_class, scales):
        super().__init__()

        fc_init = {
            'initialW': Caffe2FCUniform(),
            'initial_bias': Caffe2FCUniform(),
        }
        with self.init_scope():
            self.fc1 = L.Linear(1024, **fc_init)
            self.fc2 = L.Linear(1024, **fc_init)
            self.loc = L.Linear(
                n_class * 4, initialW=initializers.Normal(0.001))
            self.conf = L.Linear(n_class, initialW=initializers.Normal(0.01))

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
            h = F.relu(self.fc1(h))
            h = F.relu(self.fc2(h))

            loc = self.loc(h)
            loc = F.reshape(loc, (loc.shape[0], -1, 4))
            locs.append(loc)

            conf = self.conf(h)
            confs.append(conf)

        return locs, confs

    def split(self, rois, roi_indices):
        size = self.xp.sqrt(self.xp.prod(rois[:, 2:] - rois[:, :2], axis=1))
        level = self.xp.floor(self.xp.log2(
            size / self._canonical_scale + 1e-6)).astype(np.int32)
        # skip last level
        level = self.xp.clip(
            level + len(self._scales) // 2, 0, len(self._scales) - 2)

        masks = [level == l for l in range(len(self._scales))]
        rois = [rois[mask] for mask in masks]
        roi_indices = [roi_indices[mask] for mask in masks]

        return rois, roi_indices

    def decode(self, rois, roi_indices, locs, confs,
               scales, sizes, nms_thresh, score_thresh):
        rois = self.xp.vstack(rois)
        roi_indices = self.xp.hstack(roi_indices)
        locs = self.xp.vstack([loc.array for loc in locs])
        confs = self.xp.vstack([conf.array for conf in confs])

        bboxes = []
        labels = []
        scores = []
        for i in range(len(scales)):
            mask = roi_indices == i
            roi = rois[mask]
            loc = locs[mask]
            conf = confs[mask]

            bbox = self.xp.broadcast_to(roi[:, None], loc.shape) / scales[i]
            # tlbr -> yxhw
            bbox[:, :, 2:] -= bbox[:, :, :2]
            bbox[:, :, :2] += bbox[:, :, 2:] / 2
            # offset
            bbox[:, :, :2] += loc[:, :, :2] * bbox[:, :, 2:] * self.std[0]
            bbox[:, :, 2:] *= self.xp.exp(
                self.xp.minimum(loc[:, :, 2:] * self.std[1], exp_clip))
            # yxhw -> tlbr
            bbox[:, :, :2] -= bbox[:, :, 2:] / 2
            bbox[:, :, 2:] += bbox[:, :, :2]
            # clip
            bbox[:, :, :2] = self.xp.maximum(bbox[:, :, :2], 0)
            bbox[:, :, 2:] = self.xp.minimum(
                bbox[:, :, 2:], self.xp.array(sizes[i]))

            conf = self.xp.exp(conf)
            score = conf / self.xp.sum(conf, axis=1, keepdims=True)

            bbox, label, score = _suppress(
                bbox, score, nms_thresh, score_thresh)

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)

        return bboxes, labels, scores


class Caffe2FCUniform(chainer.initializer.Initializer):

    def __call__(self, array):
        scale = 1 / np.sqrt(array.shape[-1])
        initializers.Uniform(scale)(array)


def _suppress(raw_bbox, raw_score, nms_thresh, score_thresh):
    xp = cuda.get_array_module(raw_bbox, raw_score)

    bbox = []
    label = []
    score = []
    for l in range(raw_score.shape[1] - 1):
        bbox_l = raw_bbox[:, l + 1]
        score_l = raw_score[:, l + 1]

        mask = score_l >= score_thresh
        bbox_l = bbox_l[mask]
        score_l = score_l[mask]

        indices = utils.non_maximum_suppression(bbox_l, nms_thresh, score_l)
        bbox_l = bbox_l[indices]
        score_l = score_l[indices]

        bbox.append(bbox_l)
        label.append(xp.array((l,) * len(bbox_l)))
        score.append(score_l)

    bbox = xp.vstack(bbox).astype(np.float32)
    label = xp.hstack(label).astype(np.int32)
    score = xp.hstack(score).astype(np.float32)
    return bbox, label, score


def head_loss(locs, confs, rois, roi_indices, std, bboxes, labels):
    thresh = 0.5
    batchsize_per_image = 512
    fg_ratio = 0.25

    locs = F.concat(locs, axis=0)
    confs = F.concat(confs, axis=0)

    xp = cuda.get_array_module(locs.array, confs.array)

    rois = xp.vstack(rois)
    roi_indices = xp.hstack(roi_indices)

    rois_yx = (rois[:, 2:] + rois[:, :2]) / 2
    rois_hw = rois[:, 2:] - rois[:, :2]

    loc_loss = 0
    conf_loss = 0
    for i in xp.unique(roi_indices):
        mask = roi_indices = i

        if len(bboxes[i]) > 0:
            iou = utils.bbox_iou(rois[mask], bboxes[i])
            gt_index = iou.argmax(axis=1)

            gt_loc = bboxes[i][gt_index].copy()
        else:
            gt_loc = xp.empty_like(rois[mask])
        # tlbr -> yxhw
        gt_loc[:, 2:] -= gt_loc[:, :2]
        gt_loc[:, :2] += gt_loc[:, 2:] / 2
        # offset
        gt_loc[:, :2] = (gt_loc[:, :2] - rois_yx[mask]) / \
            rois_hw[mask] / std[0]
        gt_loc[:, 2:] = xp.log(gt_loc[:, 2:] / rois_hw[mask]) / std[1]

        if len(bboxes[i]) > 0:
            gt_label = labels[i][gt_index] + 1
            gt_label[iou.max(axis=1) < thresh] = 0
        else:
            gt_label = xp.zeros(mask.sum(), dtype=np.int32)

        fg_index = xp.where(gt_label > 0)[0]
        n_fg = int(batchsize_per_image * fg_ratio)
        if len(fg_index) > n_fg:
            gt_label[xp.random.choice(
                fg_index, size=len(fg_index) - n_fg, replace=False)] = -1

        bg_index = xp.where(gt_label == 0)[0]
        n_bg = batchsize_per_image - (gt_label > 0).sum()
        if len(bg_index) > n_bg:
            gt_label[xp.random.choice(
                bg_index, size=len(bg_index) - n_bg, replace=False)] = -1

        n_sample = (gt_label >= 0).sum()
        loc_loss += F.sum(F.huber_loss(
            locs[mask][gt_label > 0], gt_loc[gt_label > 0], 1,
            reduce='no')) / n_sample
        conf_loss += F.softmax_cross_entropy(
            confs[mask][gt_label >= 0], gt_label[gt_label >= 0])

    return loc_loss, conf_loss
