# ipcvl2021-mobilenetv3-torch-v2
MobileNetV3 PyTorch implementation (V2)

# 1. MobileNetV3-0.5 classifier
## 1.1. Overview
|       Type      |   Description   |   Notes   |
|:---------------:|:---------------:|:---------:|
| PyTorch Version | 1.9.0a0+2ecb2c7 | with CUDA |
| OS              | Ubuntu Server   |           |
| GPU             | RTX 2080 Ti     | VRAM 11GB |
- 기간: 2021/06/01 ~ 2021/06/22

## 1.2. Hyperparameters & Environment
|        Type       |                Description               |                                      Notes                                      |
|:-----------------:|:----------------------------------------:|:-------------------------------------------------------------------------------:|
| Model             | MobileNetV3                              | width_mult=0.5, classifier_out_features=1000)                                   |
| Model Type        | Classifier                               | ILSVRC2012                                                                      |
| AMP autograd      | True                                     |                                                                                 |
| EMA               | pytorch_ema                              | https://github.com/fadel/pytorch_ema                                            |
| Batch size        | 128                                      | Affects learning rate                                                           |
| BatchNorm         | True                                     | as specified on paper                                                           |
| Input size        | 3 * 224 * 224                            | CHW for pytorch                                                                 |
| Input constraints | FP32, range normalized, image normalized | mean=[0.4547857, 0.4349471, 0.40525291] std=[0.12003352, 0.12323549, 0.1392444] |
| Augmentations     | RandomResizedCrop, RandomHorizontalFlip  | torchvision.transformations                                                     |
| Output size       | 1000                                     | ILSVRC2012                                                                      |
| Optimizer         | Adam                                     | LR=5e-3                                                                         |

## 1.3. Tensorboard result (~24 epcoh only)
![image](https://user-images.githubusercontent.com/5201073/123035256-04d75a00-d426-11eb-8c99-8e5edfa1fc83.png)

## 1.4. Pretrained model
- Trained total 203 epochs (train_loss=2.183716, val_loss=2.088803, val_acc=0.528049)
- Use with `width_mult=0.5, classifier_out_features=1000`
- Download: [mobilenetv3-0.5.zip](https://github.com/jungin500/ipcvl2021-mobilenetv3-torch-v2/releases/download/v1.0-mbv3-0.5/mobilenetv3-0.5.zip) (~32MB)