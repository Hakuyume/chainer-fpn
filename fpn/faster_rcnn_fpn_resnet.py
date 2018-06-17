import numpy as np

import chainer
import chainer.functions as F

import chainercv

from fpn import Head
from fpn import FasterRCNN
from fpn.fpn import FPN
from fpn.rpn import RPN


def _make_fpn(cls):
    base = cls(n_class=1, arch='he')
    base.pick = ('res2', 'res3', 'res4', 'res5')
    base.remove_unused()

    base.mean = np.array((122.7717, 115.9465, 102.9801))[:, None, None]
    base.pool1 = lambda x: F.max_pooling_2d(
        x, 3, stride=2, pad=1, cover_all=False)
    for link in base.links():
        if hasattr(link, 'bn'):
            size = len(link.bn.avg_mean)
            del link.bn
            with link.init_scope():
                link.bn = AffineChannel(size)

    return FPN(base, (1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64))


class FasterRCNNFPNResNet50(FasterRCNN):

    def __init__(self, n_fg_class):
        extractor = _make_fpn(chainercv.links.ResNet50)
        super().__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            head=Head(n_fg_class + 1, extractor.scales),
        )


class FasterRCNNFPNResNet101(FasterRCNN):

    def __init__(self, n_fg_class):
        extractor = _make_fpn(chainercv.links.ResNet101)
        super().__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            head=Head(n_fg_class + 1, extractor.scales),
        )


class AffineChannel(chainer.Link):

    def __init__(self, size):
        super().__init__()
        self.gamma = np.empty(size, dtype=np.float32)
        self.beta = np.empty(size, dtype=np.float32)
        self.register_persistent('gamma')
        self.register_persistent('beta')

    def __call__(self, x):
        return x * self.gamma[:, None, None] + self.beta[:, None, None]
