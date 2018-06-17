import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from fpn.roi_align_2d import roi_align_2d


class Head(chainer.Chain):

    _roi_size = 7
    _roi_sample_ratio = 2

    def __init__(self, n_class, scales):
        super().__init__()
        with self.init_scope():
            self.fc1 = L.Linear(1024)
            self.fc2 = L.Linear(1024)
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
            h = F.relu(self.fc1(h))
            h = F.relu(self.fc2(h))

            loc = self.loc(h)
            loc = F.reshape(loc, (loc.shape[0], -1, 4))
            locs.append(loc)

            conf = self.conf(h)
            confs.append(conf)
        return locs, confs
