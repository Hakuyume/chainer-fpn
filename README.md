# Feature Pyramid Networks for Object Detection

This is an implementation of FPN (Feature Pyramid Networks) using Chainer

## Performance

mmAP on COCO 2014 minival

| backbone | original (Detectron) | ours (inference only) | ours (train & inference) |
|:-:|:-:|:-:|:-:|
| ResNet50 | 36.7 % | 35.7 % | 37.1 % |
| ResNet101 | 39.4 % | 38.2 % | 39.2 % |

## Requirements

- Python 3.6
- Chainer 4.0+
- CuPy 4.0+
- [ChainerCV](https://github.com/chainer/chainercv) (we need to build from master branch)
- ChainerMN 1.3
- [pycocotools](https://github.com/cocodataset/cocoapi)


## Demo
```
$ curl -LO https://github.com/Hakuyume/chainer-fpn/releases/download/assets/faster_rcnn_fpn_resnet50_coco.npz
$ python3 demo.py [--gpu <gpu>] --model resnet50 --pretrained-model faster_rcnn_fpn_resnet50_coco.npz <image>
```

## Training
```
$ mpiexec -n <#gpu> python3 train_coco.py --model resnet50
```
Our experiments were conducted with 8 GPUs.

## Evaluation
```
$ python3 eval_coco.py [--gpu <gpu>] --model resnet50 --pretrained-model faster_rcnn_fpn_resnet50_coco.npz
```
or
```
$ python3 eval_coco.py [--gpu <gpu>] --model resnet50 --snapshot result/snapshot_iter_90000
```

## Convert weights from Detectron

1. Download weights from [Detectron's model zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#end-to-end-faster--mask-r-cnn-baselines).
```
$ curl -L https://s3-us-west-2.amazonaws.com/detectron/35857345/12_2017_baselines/e2e_faster_rcnn_R-50-FPN_1x.yaml.01_36_30.cUF7QR7I/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl -o e2e_faster_rcnn_R-50-FPN_1x.pkl
$ curl -L https://s3-us-west-2.amazonaws.com/detectron/35857890/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_1x.yaml.01_38_50.sNxI7sX7/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl -o e2e_faster_rcnn_R-101-FPN_1x.pkl
```

2. Convert weights.
```
$ python3 detectron2npz.py e2e_faster_rcnn_R-50-FPN_1x.pkl faster_rcnn_fpn_resnet50_coco.npz
```

Note: Since the mean value in Detectron is different from that in ChainerCV,
`--mean=detectron` option should be specified for converted weights.
```
$ python3 eval_coco.py [--gpu <gpu>] --model resnet50 --mean=detectron --pretrained-model faster_rcnn_fpn_resnet50_coco.npz
```

## References
1. Tsung-Yi Lin et al. "Feature Pyramid Networks for Object Detection" CVPR 2017
2. [Detectron](https://github.com/facebookresearch/Detectron)
3. [Mask R-CNN by @wkentaro](https://github.com/wkentaro/chainer-mask-rcnn) (for the implementation of RoIAlign)
