# [안정인] MobileNetV3 논문 구현 결과 보고

## 1. Code
- Base Model (MobileNetV3): [jungin500/mobilenetv3-torch](https://github.com/jungin500/mobilenetv3-torch)
- Detector (SSDLite): [jungin500/mobilenetv3-ssd-torch@b6f596](https://github.com/jungin500/mobilenetv3-ssd-torch/commit/b6f596a2ab9a7c300203b306fe6fac01dfe6c105)

## 2. 실행 결과
![image](https://user-images.githubusercontent.com/5201073/110898526-be606280-8342-11eb-9f1d-c7c0ccbeda63.png)
![image](https://user-images.githubusercontent.com/5201073/110898626-ecde3d80-8342-11eb-876e-d720d7c1c2d4.png)
![image](https://user-images.githubusercontent.com/5201073/110903487-e05de300-834a-11eb-89b2-d9d9832ec597.png)


## 3-1. 논문: MobileNetV3 (Searching for MobileNetV3)
- 논문링크: https://arxiv.org/abs/1905.02244
   
## 3-2. 참고자료
[1] SSDLite 모델 구조: [MobileNetV2](https://arxiv.org/abs/1801.04381)
[2] torch.nn.Identity 모듈의 구현체: [d-li14/mobilenetv3.pytorch](https://github.com/d-li14/mobilenetv3.pytorch)
[3] SSDLite 모듈 및 Detector 학습 파이프라인: 
  - [qfgaohao/pytorch-ssd](https://github.com/qfgaohao/pytorch-ssd)
  - [jungin500/mobilenetv3-ssd-torch](https://github.com/jungin500/mobilenetv3-ssd-torch/commit/b6f596a2ab9a7c300203b306fe6fac01dfe6c105)

## 4. 구현 및 학습 설명
[1] Base Network인 MobileNetV3 설계 및 해당 모델을 VOC2007 데이터셋으로 1차 학습
  - Criterions
    - Loss: Cross Entropy Loss
    - Optimizer: Adam Optimizer
  - Configurations
    - Batch Size: 256 (GTX 1080 Ti * 1)
    - Epochs: 200
    - Initial Learning Rate: 1e-2 (0.01)
    - Learning Rate Decay Rate: 0.08 / 6 epochs
  - Architectures
    - Dropout on final layer: 0.2 (20%)
  - Final epoch output
    ```
    [EP:0200/0200][1/2][BA:----/0018] Begin one epoch
    [EP:0200/0200][1/2][BA:0001/0018] avg Non-zeros of output: 1000,	Metric (Accuracy on Trainset): 48.04%	Metric (Accuracy on Validset): 82.81%	Loss: 0.573984
    [EP:0200/0200][1/2][BA:0002/0018] avg Non-zeros of output: 1000,	Metric (Accuracy on Trainset): 46.87%	Metric (Accuracy on Validset): 83.98%	Loss: 0.553139
    ...
    [EP:0200/0200][1/2][BA:0017/0018] avg Non-zeros of output: 1000,	Metric (Accuracy on Trainset): 43.75%	Metric (Accuracy on Validset): 82.81%	Loss: 0.566182
    [EP:0200/0200][1/2][BA:0018/0018] avg Non-zeros of output: 1000,	Metric (Accuracy on Trainset): 44.14%	Metric (Accuracy on Validset): 83.98%	Loss: 0.462485
    [EP:0200/0200][2/2] Validation - generating metric
    [EP:0200/0200][2/2] Validation - score: 44.14%
    Base Model saved to: .checkpoints/5373-mbv3-voc-experiment-w1.00-r224-epoch0200-loss0.570-nextlr0.000387-acc0.441400.pth
    [EP:0200/0200] Average loss: 0.570	Average accuracy (Trainset): 83.0%	Average accuracy (Validset): 45.4%
    ```

[2] VOC2007로 Pre-trained된 Weight로 2차 학습 (Fine-Tuning)
  - Criterions
    - Loss: Cross Entropy Loss
    - Optimizer: Adam Optimizer
  - Configurations
    - Batch Size: 256 (GTX 1080 Ti * 1)
    - Epochs: 200
    - Initial Learning Rate: **0.000387**
    - Learning Rate Decay Rate: 0.08 / 6 epochs
  - Architectures
    - Dropout on final layer: 0.2 (20%)
  - Final epoch output
    ```
    [EP:0019/0200][1/2][BA:----/0018] Begin one epoch
    [EP:0019/0200][1/2][BA:0001/0018] avg Non-zeros of output: 1000,	Metric (Accuracy on Trainset): 75.39%	Metric (Accuracy on Validset): 83.20%	Loss: 0.497674
    [EP:0019/0200][1/2][BA:0002/0018] avg Non-zeros of output: 1000,	Metric (Accuracy on Trainset): 75.78%	Metric (Accuracy on Validset): 83.98%	Loss: 0.594128
    ...
    [EP:0019/0200][1/2][BA:0017/0018] avg Non-zeros of output: 1000,	Metric (Accuracy on Trainset): 73.04%	Metric (Accuracy on Validset): 79.68%	Loss: 0.656098
    [EP:0019/0200][1/2][BA:0018/0018] avg Non-zeros of output: 1000,	Metric (Accuracy on Trainset): 73.43%	Metric (Accuracy on Validset): 80.85%	Loss: 0.639312
    [EP:0019/0200][2/2] Validation - generating metric
    [EP:0019/0200][2/2] Validation - score: 73.43%
    Base Model saved to: .checkpoints/fe4f-mbv3-voc-experiment-after200epochs-continue-w1.00-r224-epoch0019-loss0.657-nextlr0.000301-acc0.734300.pth
    [EP:0019/0200] Average loss: 0.657	Average accuracy (Trainset): 81.0%	Average accuracy (Validset): 75.4%
    ```

[3] 최종 Fine-tuned Weight로 SSDLite 학습
  - 레포지토리 []()를 변형하여 Custom MobileNetV3 구현 및 로드
  - Custom MobileNetV3의 Classifier 부분은 제외하고 Base Model만 로드하도록 함
  - Output
    ```
    2021-03-07 23:35:01,962 - root - INFO - Use Cuda.
    2021-03-07 23:35:01,962 - root - INFO - Namespace(balance_data=False, base_net=None, base_net_lr=0.0003, batch_size=32, checkpoint_folder='models/', datasets=['/dataset/VOCdevkit/VOC2007', '/dataset/VOCdevkit/VOC2012'], debug_steps=100, extra_layers_lr=None, freeze_base_net=False, freeze_net=False, gamma=0.1, lr=0.01, mb2_width_mult=1.0, milestones='80,100', momentum=0.9, net='custom-mb3-small-ssd-lite', num_epochs=200, num_workers=4, pretrained_ssd=None, resume='models/custom-mb3-small-ssd-lite-Epoch-25-Loss-5.7945785768570435.pth', scheduler='cosine', t_max=200.0, use_cuda=True, validation_dataset='/dataset/VOCdevkit/VOC2007test', validation_epochs=5, weight_decay=0.0005)
    2021-03-07 23:35:01,963 - root - INFO - Prepare training datasets.
    2021-03-07 23:35:01,963 - root - INFO - No labels file, using default VOC classes.
    2021-03-07 23:35:01,965 - root - INFO - No labels file, using default VOC classes.
    2021-03-07 23:35:01,965 - root - INFO - Stored labels into file models/voc-model-labels.txt.
    2021-03-07 23:35:01,965 - root - INFO - Train dataset size: 16551
    2021-03-07 23:35:01,965 - root - INFO - Prepare Validation datasets.
    2021-03-07 23:35:01,966 - root - INFO - No labels file, using default VOC classes.
    2021-03-07 23:35:01,966 - root - INFO - validation dataset size: 4952
    2021-03-07 23:35:01,966 - root - INFO - Build network.
    2021-03-07 23:35:03,149 - root - INFO - Resume from the model models/custom-mb3-small-ssd-lite-Epoch-25-Loss-5.7945785768570435.pth
    2021-03-07 23:35:03,177 - root - INFO - Took 0.03 seconds to load the model.
    2021-03-07 23:35:03,194 - root - INFO - Learning rate: 0.01, Base net learning rate: 0.0003, Extra Layers learning rate: 0.01.
    2021-03-07 23:35:03,194 - root - INFO - Uses CosineAnnealingLR scheduler.
    2021-03-07 23:35:03,194 - root - INFO - Start training from epoch 0.
    2021-03-07 23:35:26,634 - root - INFO - Epoch: 0, Step: 100, Average Loss: 6.0373, Average Regression Loss 1.9278, Average Classification Loss: 4.1094
    2021-03-07 23:35:48,588 - root - INFO - Epoch: 0, Step: 200, Average Loss: 5.9560, Average Regression Loss 1.8951, Average Classification Loss: 4.0609

    ...

    2021-03-08 05:55:48,510 - root - INFO - Epoch: 199, Step: 400, Average Loss: 5.5020, Average Regression Loss 1.7464, Average Classification Loss: 3.7556
    2021-03-08 05:56:10,045 - root - INFO - Epoch: 199, Step: 500, Average Loss: 5.5089, Average Regression Loss 1.7441, Average Classification Loss: 3.7648
    2021-03-08 05:56:25,630 - root - INFO - Epoch: 199, Validation Loss: 5.2878, Validation Regression Loss 1.7701, Validation Classification Loss: 3.5177
    2021-03-08 05:56:25,657 - root - INFO - Saved model models/custom-mb3-small-ssd-lite-Epoch-199-Loss-5.287805194239462.pth
    ```