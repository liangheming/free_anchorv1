# RetinaNet
This is an unofficial pytorch implementation of FreeAnchor object detection as described in [FreeAnchor: Learning to Match Anchors for Visual Object Detection](https://arxiv.org/abs/1909.02466) by Xiaosong Zhang, Fang Wan, Chang Liu, Rongrong Ji, Qixiang Ye.

## requirement
```text
tqdm
pyyaml
numpy
opencv-python
pycocotools
torch >= 1.5
torchvision >=0.6.0
```
## result
we trained this repo on 4 GPUs with batch size 32(8 image per node).the total epoch is 24(),Adam with cosine lr decay is used for optimizing.
finally, this repo achieves 39.5 mAp at 640px(max side) resolution with resnet50 backbone(about 43.18fps).
```shell script
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.395
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.592
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.422
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.207
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.439
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.544
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.326
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.517
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.560
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.348
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.606
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.722
```

## difference from original implement
the main difference is about the input resolution.the original implement use *min_thresh* and *max_thresh* to keep the short
side of the input image larger than *min_thresh* while keep the long side smaller than *max_thresh*.for simplicity we fix the long
side a certain size, then we resize the input image while **keep the width/height ratio**, next we pad the short side.the final
width and height of the input are same.


## training
for now we only support coco detection data.

### COCO
* modify main.py (modify config file path)
```python
from solver.ddp_mix_solver import DDPMixSolver
if __name__ == '__main__':
    processor = DDPMixSolver(cfg_path="your own config path") 
    processor.run()
```
* custom some parameters in *config.yaml*
```yaml
model_name: retinanet
data:
  train_annotation_path: data/annotations/instances_train2017.json
#  train_annotation_path: data/annotations/instances_val2017.json
  val_annotation_path: data/annotations/instances_val2017.json
  train_img_root: data/train2017
#  train_img_root: data/val2017
  val_img_root: data/val2017
  max_thresh: 640
  use_crowd: False
  batch_size: 8
  num_workers: 4
  debug: False
  remove_blank: Ture

model:
  num_cls: 80
  anchor_sizes: [32, 64, 128, 256, 512]
  strides: [8, 16, 32, 64, 128]
  backbone: resnet50
  pretrained: True
  iou_thresh: 0.6
  alpha: 0.5
  gamma: 2.0
  top_k: 50
  beta: 0.11
  reg_weight: 0.75
  block_type: CGR
  dist_train: False
  conf_thresh: 0.05
  nms_iou_thresh: 0.5
  max_det: 300

optim:
  optimizer: Adam
  lr: 0.0001
  milestones: [18,24]
  warm_up_epoch: 0
  weight_decay: 0.0001
  epochs: 24
  sync_bn: True
  amp: True
val:
  interval: 1
  weight_path: weights


gpus: 0,1,2,3
```
* run train scripts
```shell script
nohup python -m torch.distributed.launch --nproc_per_node=4 main.py >>train.log 2>&1 &
```

## TODO
- [x] Color Jitter
- [x] Perspective Transform
- [x] Mosaic Augment
- [x] MixUp Augment
- [x] IOU GIOU DIOU CIOU
- [x] Warming UP
- [x] Cosine Lr Decay
- [x] EMA(Exponential Moving Average)
- [x] Mixed Precision Training (supported by apex)
- [x] Sync Batch Normalize
- [ ] PANet(neck)
- [ ] BiFPN(EfficientDet neck)
- [ ] VOC data train\test scripts
- [ ] custom data train\test scripts
- [ ] MobileNet Backbone support
