import os

import torch
from torchvision import transforms
from argparse import ArgumentParser

from Dataloader import ILSVRC2012TaskOneTwoDataset
from Model import MobileNetV3
from ILSVRC2012Preprocessor import LabelReader

from datetime import datetime
from time import time
import math

import torchsummary

if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument('--label-list', default='label.list',
    #                     help='label.list (e.g. File filled with lines containing "n03251766|dryer, drier\\n", ...) ')
    # parser.add_argument('--root-dir', default=r'S:\ILSVRC2012-CLA-DET\ILSVRC',
    #                     help=r'Annotation root directory which contains folder (default: S:\ILSVRC2012-CLA-DET\ILSVRC)')
    # parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size (default: 128)')
    # parser.add_argument('--gpu', '-g', default=False, action='store_true',
    #                     help='Use GPU to train (Not so useful on debugging)')
    # parser.add_argument('--no-cache', default=False, action='store_true',
    #                     help='Do not use cache while loading image data')
    # parser.add_argument('--save-every-epoch', default=False, action='store_true',
    #                     help='Save every epochs weight even if loss conditions are not met')
    # parser.add_argument('--dataset-pct', '-p', type=float, default=1.0,
    #                     help='Dataset usage percentage in 0.0~1.0 range (default: 1.0)')
    # parser.add_argument('--continue-weight', default=None,
    #                     help='(Optional) Continue from weight (e.g. ./weight.pth)')
    # parser.add_argument('--learning-rate', '-l', type=float, default=0.00625,
    #                     help='Learning rate (default: 0.00625)')
    # parser.add_argument('--epochs', '-e', default=200,
    #                     help='Epochs (default: 200)')
    args = parser.parse_args()

    device = torch.device('cuda:0') if args.gpu else torch.device('cpu')

    EPOCHS = 200

    labels = LabelReader(label_file_path=args.label_list).load_label()
    model = torch.nn.Sequential(
        MobileNetV3(size='small', out_features=1000),
        torch.nn.Dropout(p=0.8),
        torch.nn.Linear(1000, 1000)
    ).to(device)

    print("[%s] Loading weight file: %s" % (str(datetime.now()), args.continue_weight))
    model.load_state_dict(torch.load(args.weight))
    result_criterion = torch.nn.LogSoftmax()

    output = model(input)
    result = result_criterion(output)
    optimizer.step()