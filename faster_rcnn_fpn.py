import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

import chainercv
from chainercv import transforms
from chainercv import utils


class FasterRCNNFPNResNet101(chainer.Chain):

    _mean = np.array((122.7717, 115.9465, 102.9801))
    _min_size = 800
    _max_size = 1333
    _stride = 32

    def __init__(self, n_fg_class):
        super().__init__()
        with self.init_scope():
            self.extractor = FPNResNet101()
            self.rpn = RPN()
            self.head = Head(n_fg_class + 1)

    def __call__(self, x):
        hs = self.extractor(x)
        _, _, rois, rpn_hs = self.rpn(hs)

    def predict(self, imgs):
        resized_imgs = []
        for img in imgs:
            _, H, W = img.shape
            scale = self._min_size / min(H, W)
            if scale * max(H, W) > self._max_size:
                scale = self._max_size / max(H, W)
            img = transforms.resize(
                img, (int(H * scale), int(W * scale)))
            img -= self._mean[:, None, None]
            resized_imgs.append(img)

        size = np.array([img.shape[1:] for img in resized_imgs]).max(axis=0)
        size = (np.ceil(size / self._stride) * self._stride).astype(int)
        x = np.zeros((len(imgs), 3, size[0], size[1]), dtype=np.float32)
        for i, img in enumerate(resized_imgs):
            _, H, W = img.shape
            x[i, :, :H, :W] = img

        with chainer.using_config('train', False), chainer.no_backprop_mode():
            self(x)


class FPNResNet101(chainer.Chain):

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.resnet = chainercv.links.ResNet101(n_class=1, arch='fb')
            for i in range(2, 5 + 1):
                setattr(self, 'inner{}'.format(i), L.Convolution2D(256, 1))
                setattr(self, 'outer{}'.format(i),
                        L.Convolution2D(256, 3, pad=1))

        self.resnet.pick = ('res2', 'res3', 'res4', 'res5')
        self.resnet.remove_unused()

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
        return h2, h3, h4, h5


class RPN(chainer.Chain):

    _anchors = (0.5, 1, 2)
    _nms_thresh = 0.7
    _nms_limit_pre = 1000
    _nms_limit_post = 1000
    _roi_resolution = 7

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(256, 3, pad=1)
            self.loc = L.Convolution2D(len(self._anchors) * 4, 1)
            self.conf = L.Convolution2D(len(self._anchors), 1)

    def __call__(self, xs):
        locs = []
        confs = []
        rois = []
        ys = []
        for l, x in enumerate(xs):
            h = F.relu(self.conv(x))

            loc = self.loc(h)
            loc = F.transpose(loc, (0, 2, 3, 1))
            loc = F.reshape(loc, (loc.shape[0], -1, 4))
            locs.append(loc)

            conf = self.conf(h)
            conf = F.transpose(conf, (0, 2, 3, 1))
            conf = F.reshape(conf, (conf.shape[0], -1))
            confs.append(conf)

            _, _, H, W = x.shape
            u, v, ar = np.meshgrid(
                np.arange(H), np.arange(W), self._anchors)
            default_roi = np.stack(
                (u, v, ar, 1 / ar)).reshape((4, -1)).transpose()
            default_roi = self.xp.array(default_roi)

            roi = []
            for i in range(x.shape[0]):
                # loc_i = loc.array[i]
                conf_i = conf.array[i]

                roi_i = default_roi.copy()
                roi_i[:, :2] -= roi_i[:, 2:] / 2
                roi_i[:, 2:] += roi_i[:, :2]

                order = self.xp.argsort(-conf_i)[:self._nms_limit_pre]
                roi_i = roi_i[order]
                conf_i = conf_i[order]

                indices = utils.non_maximum_suppression(
                    roi_i, self._nms_thresh, limit=self._nms_limit_post)
                roi_i = roi_i[indices]

                roi.append(
                    self.xp.hstack((self.xp.ones((len(roi_i), 1)) * i, roi_i)))

            roi = self.xp.vstack(roi).astype(np.float32)
            rois.append(roi)

            y = _roi_pooling_2d(
                x, roi, self._roi_resolution, self._roi_resolution, 1)
            ys.append(y)

        return locs, confs, rois, ys


class Head(chainer.Chain):

    def __init__(self, n_class):
        super().__init__()
        with self.init_scope():
            self.fc6 = L.Linear(1024)
            self.fc7 = L.Linear(1024)
            self.loc = L.Linear(n_class * 4)
            self.conf = L.Linear(n_class)

    def __call__(self, x):
        h = F.relu(self.fc6(x))
        h = F.relu(self.fc7(x))
        loc = self.loc(h)
        conf = self.conf(h)
        return loc, conf


def _upsample(x):
    return F.unpooling_2d(x, 2, cover_all=False)


def _roi_pooling_2d(x, rois, outh, outw, spatial_scale):
    return F.roi_pooling_2d(
        x, rois[:, [0, 2, 1, 4, 3]], outh, outw, spatial_scale)
