import math
import os
from argparse import ArgumentParser
from datetime import datetime
from time import time
import uuid

import torch
import torchsummary
from torchvision import transforms

from imagenet import ImageNet
from voc import VOC
from ILSVRC2012Preprocessor import LabelReader
from Model import MobileNetV3
from output_logger import OutputLogger

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--label-list', default='label.list',
                        help='label.list (e.g. File filled with lines containing "n03251766|dryer, drier\\n", ...) ')
    parser.add_argument('--root-dir', default=r'S:\ILSVRC2012-CLA-DET\ILSVRC',
                        help=r'Annotation root directory which contains folder (default: S:\ILSVRC2012-CLA-DET\ILSVRC), could be .annotations/')
    parser.add_argument('--dataset-type', '-t', default='imagenet', help='Dataset type (default: imagenet, available: imagenet, voc)')
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--input-size', '-i', type=int, default=224,
                        help='Image input size (default: 224, candidates: 224, 192, 160, 128)')
    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of dataset workers (default: 16)')
    parser.add_argument('--width-mult', '-w', type=float, default=1.0,
                        help='Channel width multiplier (default: 1.0, candidates: 1.0, 0.75, 0.5, 0.25)')
    parser.add_argument('--use-hdf5', default=False, action='store_true',
                        help='Use HDF5 preprocesed dataset')
    parser.add_argument('--hdf5-root', help='HDF5 file root directory')
    parser.add_argument('--gpu', '-g', default=False, action='store_true',
                        help='Use GPU to train (Not so useful on debugging)')
    parser.add_argument('--no-cache', default=False, action='store_true',
                        help='Do not use cache while loading image data')
    parser.add_argument('--fast-debug', default=False, action='store_true',
                        help='Minimizing overheads (Disables validation and model saving features)')
    parser.add_argument('--save-every-epoch', default=False, action='store_true',
                        help='Save every epochs weight even if loss conditions are not met')
    parser.add_argument('--dataset-pct', '-p', type=float, default=1.0,
                        help='Dataset usage percentage in 0.0~1.0 range (default: 1.0)')
    parser.add_argument('--continue-weight', default=None,
                        help='(Optional) Continue from weight (e.g. ./weight.pth)')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.00625,
                        help='Learning rate (default: 0.00625)')
    parser.add_argument('--summary', '-s', default=False, action='store_true',
                        help='(Optional) summarize model')
    parser.add_argument('--epochs', '-e', default=200, type=int,
                        help='Epochs (default: 200)')
    parser.add_argument('--print-interval', default=100, type=int,
                        help='Batch process print interval (default: 200)')
    args = parser.parse_args()

    # Unique run footprint string (could be UUID)
    run_uuid = str(uuid.uuid1())
    logger = OutputLogger(run_uuid)

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0') if args.gpu else torch.device('cpu')

    if not args.gpu:
        logger.info("Using CPU trainer!")
    
    if args.dataset_type.lower() == 'imagenet':
        labels = LabelReader(label_file_path=args.label_list).load_label()
        datasets = ImageNet(
            labels=labels,
            root_dir=args.root_dir,
            device=torch.device('cpu'),
            input_size=args.input_size,
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
    elif args.dataset_type.lower() == 'voc':
        labels = []
        with open(args.label_list, 'r') as f:
            items = f.read().rstrip().split("\n")
            for item in items:
                if '|' in item:
                    raise RuntimeError("Requires VOC labels, not a ImageNet one! specify voc.label file with --label-list !")
                labels.append(item.rstrip())
        
        datasets = VOC(
            labels=labels,
            root_dir=args.root_dir,
            device=torch.device('cpu'),
            input_size=args.input_size,
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
        
        # raise RuntimeError("Not Yet Implemented")
    else:
        raise RuntimeError("Dataset type not supported!")

    validset_items = args.batch_size * 1  # one-batch only
    trainset_items = len(datasets) - validset_items

    train_datasets, valid_datasets = torch.utils.data.random_split(
        datasets, [trainset_items, validset_items]
    )

    # On OneGPU (without DataParallel)
    # batch_size=64 -> 75sec
    # batch_size=128 -> 73sec
    # batch_size=256 -> 72sec

    # On Two GPUs (DataParallel)
    # batch_size=64 ->  81sec
    # batch_size=128 -> 72sec
    # batch_size=256 -> 84sec
    # batch_size=512 -> 71sec
    # Disk I/O bottleneck here? none of them?

    train_dataloader = torch.utils.data.DataLoader(train_datasets,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True
                                                   )

    valid_dataloader = torch.utils.data.DataLoader(valid_datasets,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_workers,
                                                   pin_memory=True,
                                                   drop_last=True
                                                   )


    base_model = MobileNetV3(size='small', width_mult=args.width_mult)
    out_features = 1000
    classifier = torch.nn.Sequential(
        torch.nn.AdaptiveAvgPool2d(output_size=1),
        # torch.nn.Conv2d(in_channels=int(576 * args.width_mult), out_channels=int(1024 * args.width_mult), kernel_size=(1, 1), bias=False),
        # torch.nn.Hardswish(inplace=True),
        # torch.nn.Conv2d(in_channels=int(1024 * args.width_mult), out_channels=out_features, kernel_size=(1, 1), bias=False),  # paper output
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(int(576 * args.width_mult), int(1024 * args.width_mult)),
        torch.nn.Dropout(p=0.2),  # paper=0.8 but acc only achieves 10%
        torch.nn.Linear(int(1024 * args.width_mult), out_features)
    )

    model = torch.nn.Sequential(
        base_model,
        classifier
    )

    def xavier_init(m):
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    model.apply(xavier_init)
    if args.gpu:
        model = torch.nn.DataParallel(model)

    if args.continue_weight is not None:
        logger.info("Loading weight file: %s" % args.continue_weight)
        model.load_state_dict(torch.load(args.continue_weight))

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    # optimizer = torch.optim.RMSprop(model.parameters(),
    #                                 lr=args.learning_rate, momentum=1e-5, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.summary:
        gpu = torch.device('cuda:0')
        model.to(gpu)
        torchsummary.summary(model, (3, 128, 128))
        exit(0)

    cpu_device = torch.device('cpu')
    epoch_print_interval = args.print_interval
    
    # We will save one batch of this Validation Dataloader
    # as valid_dataloader is ONLY ONE BATCH!
    saved_validation_set = next(iter(valid_dataloader))

    first_epoch = True
    epoch_avg_loss_cumulative = 0
    epoch_avg_loss_prev = None
    logger.info("Begin training sequence - initializing dataset (first enumeration)")
    for epoch in range(args.epochs):

        if epoch == 0:
            logger.info("Epoch timer began")
            begin_time = time()

        # Training time
        running_loss = torch.tensor(0.0, device=device)
        acc_avg_train = 0
        acc_avg_valid = 0
        acc_avg_iterations = 0

        model.train()
        for i, data in enumerate(train_dataloader):
            if i == 0:
                logger.info("[EP:%04d/%04d][1/2][BA:----/%04d] Begin one epoch" % (epoch + 1, args.epochs, len(train_dataloader)))

            input, label = data

            if first_epoch and i == 0:
                logger.info("Dataset initialization complete")
                logger.info("Begin Training, %d epoches, each %d batches (batch size %d), learning rate %.0e(%.6f)" %
                      (args.epochs, len(train_dataloader), args.batch_size, args.learning_rate, args.learning_rate))
                if args.save_every_epoch:
                    logger.info("Saving every epochs")

            optimizer.zero_grad()
            output = model(input)
            # Because we had to do pin workers!
            if output.device.type != label.device.type:
                label = label.to(device)

            loss = criterion(output, label)
            loss.backward()  # if - loss is 6.907756328582764, then output might be all-zeros.
            optimizer.step()
            running_loss += loss.item()

            # Begin debug validation
            if args.fast_debug or (i == 0 or i % epoch_print_interval == (epoch_print_interval - 1)):
                acc_avg_iterations += 1  # later, we will divide by this value
                
                model.eval()
                output_metric = torch.softmax(output, dim=1)
                output_metric = torch.argmax(output_metric, dim=1, keepdim=False)
                diff = (output_metric == label).int()
                metric = (torch.sum(diff) / output_metric.shape[0])

                non_zeros_count = torch.sum((torch.zeros_like(output) != output).type(torch.int8))
                non_zeros_count = non_zeros_count.detach().cpu().numpy() / output_metric.shape[0]

                metric = metric.to(cpu_device).numpy()
                metric = int(metric.item() * 10000) / 100.0
                acc_avg_train += metric
                
                # Also, REAL validation for each batch! (Optional)
                input, label = saved_validation_set
                output = model(input)
                output = torch.softmax(output, dim=1)
                output = torch.argmax(output, dim=1, keepdim=False)
                label = label.to(device)
                diff = (output == label).int()
                validset_metric = (torch.sum(diff) / output.shape[0])
                validset_metric = validset_metric.to(cpu_device).numpy()
                validset_metric = int(validset_metric.item() * 10000) / 100.0
                acc_avg_valid += validset_metric

                logger.info("[EP:%04d/%04d][1/2][BA:%04d/%04d] avg Non-zeros of output: %d,\tMetric (Accuracy on Trainset): %.2f%%\tMetric (Accuracy on Validset): %.2f%%\tLoss: %.6f" %
                      (epoch + 1, args.epochs, i + 1, len(train_dataloader), non_zeros_count, validset_metric, metric, loss.item()))
                # End for REAL validation of each batch
                
                if validset_metric >= 99.99:
                    basepath = '.checkpoints' + os.sep
                    basemodel_filename = '%s-w%.2f-r%d-epoch%04d-loss%.3f-nextlr%.6f-acc%.6f.pth' % (run_uuid[:4], args.width_mult, args.input_size, epoch + 1, running_loss, optimizer.param_groups[0]['lr'], metric / 100)
                    savepath = basepath + basemodel_filename

                    if not os.path.isdir(basepath):
                        os.mkdir(basepath)

                    torch.save(base_model.state_dict(), basepath + basemodel_filename)
                    logger.info("Metric Overwhelmed! Saved model to %s and Exciting ..." % basepath + basemodel_filename)
                    exit(0)
                model.train()
            # End debug validation
            # One-epoch training end

        running_loss /= (i + 1)
        
        # two metrics are NOT based on each batches
        acc_avg_train /= acc_avg_iterations
        acc_avg_valid /= acc_avg_iterations

        # Validation time
        if args.fast_debug:
            logger.info("[EP:%04d/%04d][2/2] Skipping validation" % (epoch + 1, args.epochs))
        else:
            logger.info("[EP:%04d/%04d][2/2] Validation - generating metric" % (epoch + 1, args.epochs))
            model.eval()
            input, label = saved_validation_set

            output = model(input)
            output = torch.softmax(output, dim=1)
            output = torch.argmax(output, dim=1, keepdim=False)
            if output.device.type != label.device.type:
                label = label.to(device)

            diff = (output == label).int()
            metric = (torch.sum(diff) / output.shape[0])
            metric = metric.to(cpu_device).numpy()
            metric = int(metric.item() * 10000) / 100.0

            logger.info("[EP:%04d/%04d][2/2] Validation - score: %.2f%%" % (epoch + 1, args.epochs, metric))
            model.train()


        # Check Learning rate after Each epoch
        # if epoch % 45 == 0 and epoch != 0:
        # VOC dataset is very-sensitive? do for every epochs
        # VOC might be too small for model!
        if epoch % 5 == 0 and epoch != 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.92
            logger.info("[EP:%04d/%04d] Updating running rate to %.6f" % (epoch + 1, args.epochs, g['lr']))

        if epoch == 0:
            end_time = time()
            delay_secs = end_time - begin_time
            logger.info("[EP:%04d/%04d] One epoch took %d seconds" % (epoch + 1, args.epochs, delay_secs))

        # Save better results
        if not args.fast_debug:
            if args.save_every_epoch or \
                    epoch_avg_loss_prev is None or running_loss < epoch_avg_loss_prev:
                basepath = '.checkpoints' + os.sep
                basemodel_filename = '%s-w%.2f-r%d-epoch%04d-loss%.3f-nextlr%.6f-acc%.6f.pth' % (run_uuid[:4], args.width_mult, args.input_size, epoch + 1, running_loss, optimizer.param_groups[0]['lr'], metric / 100)
                savepath = basepath + basemodel_filename

                if not os.path.isdir(basepath):
                    os.mkdir(basepath)

                torch.save(base_model.state_dict(), basepath + basemodel_filename)
                logger.info('Base Model saved to: %s' % savepath)

        epoch_avg_loss_cumulative += running_loss
        epoch_avg_loss_prev = running_loss

        logger.info('[EP:%04d/%04d] Average loss: %.3f\tAverage accuracy (Trainset): %.1f%%\tAverage accuracy (Validset): %.1f%%' % (epoch + 1, args.epochs, running_loss, acc_avg_train, acc_avg_valid))

        if first_epoch:
            first_epoch = False

    logger.info("Training success")
