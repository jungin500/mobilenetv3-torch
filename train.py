import math
import os
from argparse import ArgumentParser
from datetime import datetime
from time import time
import uuid

import torch
import torchsummary
from torchvision import transforms

from Dataloader import ILSVRC2012TaskOneTwoDataset, HDF5Dataset
from ILSVRC2012Preprocessor import LabelReader
from Model import MobileNetV3

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--label-list', default='label.list',
                        help='label.list (e.g. File filled with lines containing "n03251766|dryer, drier\\n", ...) ')
    parser.add_argument('--root-dir', default=r'S:\ILSVRC2012-CLA-DET\ILSVRC',
                        help=r'Annotation root directory which contains folder (default: S:\ILSVRC2012-CLA-DET\ILSVRC)')
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--input-size', '-i', type=int, default=224,
                        help='Image input size (default: 224, candidates: 224, 192, 160, 128)')
    parser.add_argument('--width-mult', '-w', type=float, default=1.0,
                        help='Channel width multiplier (default: 1.0, candidates: 1.0, 0.75, 0.5, 0.25)')
    parser.add_argument('--use-hdf5', default=False, action='store_true',
                        help='Use HDF5 preprocesed dataset')
    parser.add_argument('--hdf5-root', help='HDF5 file root directory')
    parser.add_argument('--gpu', '-g', default=False, action='store_true',
                        help='Use GPU to train (Not so useful on debugging)')
    parser.add_argument('--no-cache', default=False, action='store_true',
                        help='Do not use cache while loading image data')
    parser.add_argument('--save-every-epoch', default=False, action='store_true',
                        help='Save every epochs weight even if loss conditions are not met')
    parser.add_argument('--dataset-pct', '-p', type=float, default=1.0,
                        help='Dataset usage percentage in 0.0~1.0 range (default: 1.0)')
    parser.add_argument('--continue-weight', default=None,
                        help='(Optional) Continue from weight (e.g. ./weight.pth)')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.00625,
                        help='Learning rate (default: 0.00625)')
    parser.add_argument('--live-validation', default=False, action='store_true',
                        help='Validate on evey batch ends')
    parser.add_argument('--summary', '-s', default=False, action='store_true',
                        help='(Optional) summarize model')
    parser.add_argument('--epochs', '-e', default=200, type=int,
                        help='Epochs (default: 200)')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0') if args.gpu else torch.device('cpu')

    if not args.gpu:
        print("WARNING: Using CPU trainer!")

    labels = LabelReader(label_file_path=args.label_list).load_label()

    if args.use_hdf5:
        datasets = HDF5Dataset(
            root_dir=args.hdf5_root,
            device=device
        )

        train_dataloader = datasets
        valid_dataloader = None

        trainset_items = len(datasets)

    else:
        datasets = ILSVRC2012TaskOneTwoDataset(
            labels=labels,
            root_dir=args.root_dir,
            device=device,
            input_size=args.input_size,
            # transform disabled due to dataset architecture change
            transform=transforms.Compose([
                transforms.RandomResizedCrop((args.input_size, args.input_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]),
            use_cache=not args.no_cache,
            dataset_usage_pct=args.dataset_pct
        )

        trainset_items = math.floor(len(datasets) * 0.9)
        validset_items = len(datasets) - trainset_items

        train_datasets, valid_datasets = torch.utils.data.random_split(
            datasets, [trainset_items, validset_items]
        )

        train_dataloader = torch.utils.data.DataLoader(train_datasets,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       pin_memory=False
                                                       )  # do not use num_workers and pin_memory on Windows!

        valid_dataloader = torch.utils.data.DataLoader(valid_datasets,
                                                       batch_size=args.batch_size,
                                                       shuffle=True,
                                                       num_workers=0,
                                                       pin_memory=False
                                                       )  # do not use num_workers and pin_memory on Windows!

    base_model = MobileNetV3(size='small', out_features=1000, width_mult=args.width_mult)
    classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.8),
        torch.nn.Linear(1000, 1000)
    )

    model = torch.nn.Sequential(
        base_model,
        classifier
    )

    if args.continue_weight is not None:
        print("[%s] Loading weight file: %s" % (str(datetime.now()), args.continue_weight))
        model.load_state_dict(torch.load(args.continue_weight))

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=args.learning_rate, momentum=1e-5, weight_decay=1e-5)

    if args.summary:
        gpu = torch.device('cuda:0')
        model.to(gpu)
        torchsummary.summary(model, (3, 128, 128))
        exit(0)

    # Unique run footprint string (could be UUID)
    run_footprint = str(uuid.uuid1())

    first_batch = True
    epoch_avg_loss_cumulative = 0
    epoch_avg_loss_prev = None
    print("[%s] Begin training sequence - initializing dataset (first enumeration)" % (str(datetime.now()),))
    for epoch in range(args.epochs):

        if epoch == 0:
            print("[%s] First Epoch: Warming up epoch" % (str(datetime.now()),))
        elif epoch == 1:
            print("[%s] Epoch timer began" % (str(datetime.now()),))
            begin_time = time()

        # Training time
        running_loss = 0

        for i, data in enumerate(train_dataloader):
            input, label = data

            if first_batch:
                print("[%s] Dataset initialization complete" % str(datetime.now()))
                print("[%s] Begin Training, %d epoches, each %d batches (batch size %d), learning rate %.0e(%.6f)" %
                      (str(datetime.now()), args.epochs, math.ceil(trainset_items / args.batch_size), args.batch_size,
                       args.learning_rate, args.learning_rate))
                if args.save_every_epoch:
                    print("[%s] Saving every epochs" % str(datetime.now()))
                first_batch = False

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label.squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Live-valiation
            if valid_dataloader is not None and args.live_validation:
                metric_sums = 0
                for j, valid_data in enumerate(valid_dataloader):
                    input, label = valid_data

                    output = model(input)
                    output = torch.softmax(output, dim=1)
                    output = torch.argmax(output, dim=1, keepdim=False)

                    diff = (output == label).int()
                    batch_items = diff.shape[0]
                    metric = torch.sum(diff) / batch_items
                    metric = metric.item()
                    metric_sums += metric

                metric_sums /= (j + 1)

                print(
                    f'[{str(datetime.now())}]' + '[epoch %03d, batch %03d] cumulative loss: %.3f current batch loss: %.3f validation accuracy: %.3f' %
                    (epoch + 1, i + 1, running_loss / (i + 1), loss.item(), metric_sums))
            else:
                print(
                    f'[{str(datetime.now())}]' + '[epoch %03d, batch %03d] cumulative loss: %.3f current batch loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / (i + 1), loss.item()))

        running_loss /= i

        print("[%s] Begin valiating sequence" % (str(datetime.now()),))

        # Validation time
        if valid_dataloader is not None and not args.live_validation:
            validation_accuracy_set = 0
            validation_items = 0
            for i, data in enumerate(valid_dataloader):
                input, label = data

                output = model(input)
                output = torch.softmax(output, dim=1)
                output = torch.argmax(output, dim=1, keepdim=False)

                diff = (output == label).int()
                batch_items = diff.shape[0]
                metric = torch.sum(diff)

                validation_accuracy_set += metric
                validation_items += batch_items

            metric = validation_accuracy_set / validation_items
            print("[%s] Validation score: %.5f" % (str(datetime.now()), float(metric.cpu().numpy())))

        if epoch % 3 == 2:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.99
            print("[%s] Updating running rate to %.3f" % (str(datetime.now()), g['lr']))

        if epoch == 1:
            end_time = time()
            delay_secs = end_time - begin_time
            print("[%s] One epoch took %d seconds" % (str(datetime.now()), delay_secs))

        current_epoch_avg_loss = running_loss / (i + 1)

        # Save better results
        if args.save_every_epoch or \
                epoch_avg_loss_prev is None or current_epoch_avg_loss < epoch_avg_loss_prev:
            basepath = '.checkpoints' + os.sep
            savepath = basepath + '%s-mobilenetv3-custom-w%.2f-r%d-epoch%03d-loss%.3f-nextlr%.6f.pth' % \
                       (run_footprint,  args.width_mult, args.input_size, epoch + 1, current_epoch_avg_loss, optimizer.param_groups[0]['lr'])

            if not os.path.isdir(basepath):
                os.mkdir(basepath)

            print(f'[{str(datetime.now())}]' + '[epoch %03d] saved model to: %s' %
                  (epoch + 1, savepath))

            torch.save(model.state_dict(), savepath)

        epoch_avg_loss_cumulative += current_epoch_avg_loss
        epoch_avg_loss_prev = current_epoch_avg_loss

        print(f'[{str(datetime.now())}]' + '[epoch %03d] current epoch average loss: %.3f' %
              (epoch + 1, current_epoch_avg_loss))

    print("Training success")
