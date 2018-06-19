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

    _roi_size = 7
    _roi_sample_ratio = 2
    _std = (0.1, 0.2)

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

    def decode(self, rois, roi_indices, locs, confs,
               scales, sizes, nms_thresh, score_thresh):
        bboxes = []
        labels = []
        scores = []
        for i in range(len(scales)):
            bbox = []
            score = []
            for l in range(len(self._scales) - 1):
                mask = roi_indices[l] == i
                roi_l = rois[l][mask]
                loc_l = locs[l].array[mask]
                conf_l = confs[l].array[mask]

                bbox_l = self.xp.broadcast_to(
                    roi_l[:, None], loc_l.shape) / scales[i]
                # tlbr -> yxhw
                bbox_l[:, :, 2:] -= bbox_l[:, :, :2]
                bbox_l[:, :, :2] += bbox_l[:, :, 2:] / 2
                # offset
                bbox_l[:, :, :2] += loc_l[:, :, :2] * \
                    bbox_l[:, :, 2:] * self._std[0]
                bbox_l[:, :, 2:] *= self.xp.exp(
                    self.xp.minimum(loc_l[:, :, 2:] * self._std[1], exp_clip))
                # yxhw -> tlbr
                bbox_l[:, :, :2] -= bbox_l[:, :, 2:] / 2
                bbox_l[:, :, 2:] += bbox_l[:, :, :2]
                # clip
                bbox_l[:, :, :2] = self.xp.maximum(bbox_l[:, :, :2], 0)
                bbox_l[:, :, 2:] = self.xp.minimum(
                    bbox_l[:, :, 2:], self.xp.array(sizes[i]))

                conf_l = self.xp.exp(conf_l)
                score_l = conf_l / self.xp.sum(conf_l, axis=1, keepdims=True)

                bbox.append(bbox_l)
                score.append(score_l)

            bbox = self.xp.vstack(bbox)
            score = self.xp.vstack(score)
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
