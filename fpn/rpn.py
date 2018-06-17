import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv import utils

from fpn import exp_clip


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
                    self.xp.minimum(loc_l[:, 2:], exp_clip))
                roi_l[:, :2] -= roi_l[:, 2:] / 2
                roi_l[:, 2:] += roi_l[:, :2]

                order = self.xp.argsort(-conf_l)[:self._nms_limit_pre]
                roi_l = roi_l[order]
                conf_l = conf_l[order]

                roi_l[:, :2] = self.xp.maximum(roi_l[:, :2], 0)
                roi_l[:, 2:] = self.xp.minimum(
                    roi_l[:, 2:], self.xp.array(x_shape[2:]))

                mask = (roi_l[:, 2:] - roi_l[:, :2] > 0).all(axis=1)
                roi_l = roi_l[mask]
                conf_l = conf_l[mask]

                indices = utils.non_maximum_suppression(
                    roi_l, self._nms_thresh, limit=self._nms_limit_post)
                roi_l = roi_l[indices]
                conf_l = conf_l[indices]

                roi.append(roi_l)
                conf.append(conf_l)

            roi = self.xp.vstack(roi).astype(np.float32)
            conf = self.xp.hstack(conf).astype(np.float32)

            order = self.xp.argsort(-conf)[:self._nms_limit_post]
            roi = roi[order]

            size = self.xp.sqrt(self.xp.prod(roi[:, 2:] - roi[:, :2], axis=1))
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
