import argparse

import chainer
from chainer import iterators

from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import COCOBboxDataset
from chainercv.evaluations import eval_detection_coco
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from faster_rcnn_fpn import FasterRCNNFPNResNet101


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('model')
    args = parser.parse_args()

    model = FasterRCNNFPNResNet101(n_fg_class=len(coco_bbox_label_names))
    chainer.serializers.load_npz(args.model, model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    model.use_preset('evaluate')

    dataset = COCOBboxDataset(
        split='minival',
        use_crowded=True,
        return_area=True,
        return_crowded=True)
    iterator = iterators.SerialIterator(
        dataset, 1, repeat=False, shuffle=False)

    in_values, out_values, rest_values = apply_to_iterator(
        model.predict, iterator, hook=ProgressHook(len(dataset)))
    # delete unused iterators explicitly
    del in_values

    pred_bboxes, pred_labels, pred_scores = out_values
    gt_bboxes, gt_labels, gt_area, gt_crowded = rest_values

    result = eval_detection_coco(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_area, gt_crowded)

    print()
    for area in ('all', 'large', 'medium', 'small'):
        print('map ({}):'.format(area),
              result['map/iou=0.50:0.95/area={}/maxDets=100'.format(area)])


if __name__ == '__main__':
    main()
