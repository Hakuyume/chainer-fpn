import chainer
import chainer.functions as F
import chainer.links as L

import chainercv


class FasterRCNNFPNResNet101(chainer.Chain):

    def __init__(self, n_fg_class):
        super().__init__()
        with self.init_scope():
            self.extractor = FPNResNet101()
            self.rpn = RPN()
            self.head = Head(n_fg_class + 1)

    def __call__(self, x):
        pass


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

        h5 = self.outer5(h5)
        h4 = self.outer4(h4)
        h3 = self.outer3(h3)
        h2 = self.outer2(h2)
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
