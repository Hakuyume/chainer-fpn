import numpy as np

import chainer
import chainer.functions as F

import chainercv

from fpn.fpn import FPN


def _make_resnet(cls):
    resnet = cls(n_class=1, arch='he')

    resnet.mean = np.array((122.7717, 115.9465, 102.9801))[:, None, None]
    resnet.pool1 = lambda x: F.max_pooling_2d(
        x, 3, stride=2, pad=1, cover_all=False)
    for link in resnet.links():
        if hasattr(link, 'bn'):
            size = len(link.bn.avg_mean)
            del link.bn
            with link.init_scope():
                link.bn = AffineChannel(size)

    return resnet


class FPNResNet50(FPN):

    def __init__(self):
        base = _make_resnet(chainercv.links.ResNet50)
        base.pick = ('res2', 'res3', 'res4', 'res5')
        base.remove_unused()
        super().__init__(base, (1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64))


class FPNResNet101(FPN):

    def __init__(self):
        base = _make_resnet(chainercv.links.ResNet101)
        base.pick = ('res2', 'res3', 'res4', 'res5')
        base.remove_unused()
        super().__init__(base, (1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64))


class AffineChannel(chainer.Link):

    def __init__(self, size):
        super().__init__()
        self.gamma = np.empty(size, dtype=np.float32)
        self.beta = np.empty(size, dtype=np.float32)
        self.register_persistent('gamma')
        self.register_persistent('beta')

    def __call__(self, x):
        return x * self.gamma[:, None, None] + self.beta[:, None, None]
