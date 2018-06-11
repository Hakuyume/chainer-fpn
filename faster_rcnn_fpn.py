import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

import chainercv
from chainercv import transforms


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
        return self.extractor(x)

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
            h2, h3, h4, h5 = self(x)


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

    _anchors = (0, 1, 2)

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(256, 3, pad=1)
            self.loc = L.Convolution2D(len(self._anchors) * 4, 1)
            self.conf = L.Convolution2D(len(self._anchors), 1)

    def __call__(self, x):
        pass


class Head(chainer.Chain):

    def __init__(self, n_class):
        super().__init__()
        with self.init_scope():
            self.fc6 = L.Linear(1024)
            self.fc7 = L.Linear(1024)
            self.loc = L.Linear(n_class * 4)
            self.conf = L.Linear(n_class)

    def __call__(self, x):
        pass


def _upsample(x):
    return F.unpooling_2d(x, 2, cover_all=False)
