# Feature Pyramid Networks

1. Download pretrained model from [Detectron's model zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md#end-to-end-faster--mask-r-cnn-baselines).
```
$ curl -LO https://s3-us-west-2.amazonaws.com/detectron/35857890/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_1x.yaml.01_38_50.sNxI7sX7/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl
```

2. Convert weights.
```
$ python3 caffe22npz.py model_final.pkl faster_rcnn_fpn_resnet101_coco.npz
```

3. Run inference.
```
$ python3 demo.py faster_rcnn_fpn_resnet101_coco.npz <image>
```
