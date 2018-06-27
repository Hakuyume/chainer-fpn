import argparse

import chainer
from chainer import iterators

from chainercv.datasets import coco_bbox_label_names
from chainercv.datasets import COCOBboxDataset
from chainercv.evaluations import eval_detection_coco
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from fpn import FasterRCNNFPNResNet101
from fpn import FasterRCNNFPNResNet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', choices=('resnet50', 'resnet101'))
    parser.add_argument('--caffe2-mean', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pretrained-model')
    group.add_argument('--snapshot')
    args = parser.parse_args()

    if args.model == 'resnet50':
        model = FasterRCNNFPNResNet50(n_fg_class=len(coco_bbox_label_names),
                                      caffe2_mean=args.caffe2_mean)
    elif args.model == 'resnet101':
        model = FasterRCNNFPNResNet101(n_fg_class=len(coco_bbox_label_names),
                                       caffe2_mean=args.caffe2_mean)

    if args.pretrained_model:
        chainer.serializers.load_npz(args.pretrained_model, model)
    elif args.snapshot:
        chainer.serializers.load_npz(
            args.snapshot, model, path='updater/model:main/model/')

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    model.use_preset('evaluate')

    dataset = COCOBboxDataset(
        split='minival',
        use_crowded=True,
        return_area=True,
        return_crowded=True)
    iterator = iterators.MultithreadIterator(
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
        print('mmap ({}):'.format(area),
              result['map/iou=0.50:0.95/area={}/maxDets=100'.format(area)])


if __name__ == '__main__':
    main()
