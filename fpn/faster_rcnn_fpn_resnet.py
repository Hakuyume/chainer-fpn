from fpn import Head
from fpn import FasterRCNN
from fpn.fpn_resnet import FPNResNet101
from fpn.rpn import RPN


class FasterRCNNFPNResNet101(FasterRCNN):

    def __init__(self, n_fg_class):
        extractor = FPNResNet101()
        super().__init__(
            extractor=extractor,
            rpn=RPN(extractor.scales),
            head=Head(n_fg_class + 1, extractor.scales[:-1])
        )
