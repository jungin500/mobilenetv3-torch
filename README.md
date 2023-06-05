# mobilenetv3-torch
MobileNetV3 PyTorch implementation (V1) - [Check out](https://github.com/jungin500/mobilenetv3-torch-v2/tree/v2) `v2` branch for newer implementation

# Single-batch trials

- Trial 01: `python train.py -g -l 0.01 -p 0.001 -b 512`
    - Single-batch training 
    - 520 epochs
    - Dropout 0.1 (10%)
    - Achieved Acc 90%
    
- Trial 02: Trial 01 + `-l 0.04`
    - Learning rate might be inappropriate (500+ epochs)
     
- Trial 03: Trial 01 + `batch_size=1024, dataset_pct=0.002`
  - 900 epochs
  - Achieved acc 90%+
    
- Trial 04: Triao 03 + 300번마다 0.9
  - 답없음
    
- Trial 05: Trial No Dropout! (Removed dropout layer and inserted some FC layer)
            + `python train.py -g -l 0.02 -p 0.01 -b 1024`, lr_decay=0.9 every 45 epochs
  ```python
  ... (Base Layer End)
  torch.nn.Linear(int(576 * args.width_mult), int(1024 * args.width_mult)),
  torch.nn.Linear(int(1024 * args.width_mult), out_features)
  (End)
  ```
  - 180epochs, 210epochs, 270epochs, 320epochs, 380epochs, ...
           (45)         (45)       (45)       (45)
  - Learning rate 0.9 every 45epochs
  - Try 1: 90%+ on 373 epochs, 100% on 390 epochs
  - Try 2: (Another bunch of batch) 90%+ on 382 epochs, oh, dropped to 1.95% on 385 epochs, recovering, ... recovered on, ...
           600 epochs. 90%, Loss ~0.5, and lr changed to 0.005 and dropped once more to 72.85% on 631 epochs, and again, ...
           99.9% on 715 epochs
           (maybe 0.009 learning rate is bit high on this stage=dropped stage?)
  - Try 3: "Same batch with Try 2" 50% on 450 epochs, unstable slopes here. maybe try one more? stopped at 560 epochs. (84.47%)
  - Try 4: Still unstable with "Try 2-batch". 90% on 412 epochs and finalized on 462 epochs.
           learning rate(final to begin, 45 epochs margin): 0.007, 0.008, 0.009, 0.010, 0.011, 0.012, 0.013, 0.015, 0.016, 0.018, 0.02
    
- Trial 06: lr=0.015, lr_decay=0.92
  - Try 1: "Trial 5 pre-Try 2 batch", and 90%+ on 187 batch, and dropped to 5% at 200 epochs.
           maybe lowering initial lr might be helpful...?
    
- Trial 07: lr=0.011, lr_decay=0.92
  - Try 1: 100%+ on ONLY 129 EPOCHS! ("Trial 5 pre-Try 2 batch") here
  - Try 2: Trial 5 try2 batch
  
- Trial 08: lr=0.009, lr_decay=0.92
  - Try 1: "Trial 5 Try 2 batch", Great convergence on first-45 batch. finalinzed on 111 iteration

- Trial 09: lr=0.008, lr_decay=0.92
  - Try 1: "Trial 5 Try 2 batch", also great convergence. maybe FC-only trained? (or not?) finalized on 98 iteration
  - Try 2: "Trial 5 pre-Try 2 batch", also great convergence, only finalized on... 74 epochs!
    
- Trial 10: Trial 09 + Randomize Dataset on each iteration (Real train) `python train.py -g -l 0.008 -p 0.001 -b 1024`

- Trial 11: after **Training Trial 01**, `python train.py -g -l 0.008 -p 0.005 -b 1024 --no-cache --root-dir C:\ILSVRC2012`
  - Total 5k images (5 batches), same strategy as **Training trials**
  - 


# Training trials
- Trial 01: `python train.py -g -l 0.008 -p 1.0 -b 1024 --no-cache --root-dir C:\ILSVRC2012`
  - Whole dataset training
  - DataLoader `shuffle=True`, transforms below
    ```python
    transforms.RandomResizedCrop((args.input_size, args.input_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    ```
  - Loss doesn't shrink after 10~ epochs, minimum 4.058 and rises to ~4.345
    
- Trial 02: Trial 01 + `Dropout=0.2`
  - Begin: 210307 06:00 (oh, morning ...)
