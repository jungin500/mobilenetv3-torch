from argparse import ArgumentParser
import datetime

from torch.utils.data import Dataset
from torch import from_numpy
from PIL import Image
import glob

import os
import pickle
from xml.etree import ElementTree

import numpy as np
import torchsummary
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from torch_ema import ExponentialMovingAverage


class LabelReader(object):
    def __init__(self, label_file_path):
        self.label_file_path = label_file_path
        if 'pretrained' in label_file_path:
            print("INFO: Using Pretrained label list! (not custom one)")

    def load_label(self):
        label_map = {}
        # Read label file into label map
        if os.path.isfile(self.label_file_path):
            with open(self.label_file_path, 'r') as f:
                label_name_body = f.read().strip()
                label_name_lines = label_name_body.split("\n")
                for label_entry in tqdm(label_name_lines, desc='레이블 파일 읽기 작업'):
                    synset_name, label_name = label_entry.strip().split("|")
                    label_map[synset_name] = label_name

            print(f"레이블 파일 읽기 완료: 총 {len(list(label_map.keys()))}개 레이블 검색")
            return label_map
        else:
            return None


CACHE_BASE_DIR = "." + os.sep + ".cache"
IMG_PATH_LIST_PKL_FILENAME = CACHE_BASE_DIR + os.sep + 'img_path_list'
IMG_CLASS_LIST_PKL_FILENAME = CACHE_BASE_DIR + os.sep + 'img_class_list'


class ImageNet(torch.utils.data.Dataset):
    def __init__(self, labels, root_dir, transform=None):
        super(ImageNet, self).__init__()

        self.labels = labels
        self.transform = transform

        self.img_path_list = []
        self.img_class_list = []
        self.load_list(root_dir)

    def load_list(self, root_dir):
        label_index = 0
        for label in tqdm(self.labels.keys(), desc='이미지 파일 리스트 읽기 작업'):
            item_dir = os.path.join(root_dir, label)
            file_list = glob.glob(item_dir + os.sep + "*.JPEG")
            self.img_path_list += file_list
            self.img_class_list += [label_index] * len(file_list)
            label_index += 1

        if len(self.img_path_list) != len(self.img_class_list):
            raise RuntimeError(f"이미지 데이터 {len(self.img_path_list)}개와 클래스 데이터 {len(self.img_class_list)}개가 서로 다릅니다!")

        print(f"총 {len(self.img_path_list)}개 이미지 리스트 데이터 및 실효 레이블 {len(list(set(self.img_class_list)))}개 로드 성공")

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        # PIL-version
        image = Image.open(self.img_path_list[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.Tensor([self.img_class_list[idx]]).type(torch.int64).squeeze(dim=0)
        return image, label


class SqueezeExciteModule(nn.Module):
    def __init__(self, expand_size):
        super(SqueezeExciteModule, self).__init__()

        self.se_0_0 = nn.AdaptiveAvgPool2d(output_size=1)
        self.se_0_1 = nn.Flatten()

        self.se_1_0 = nn.Linear(in_features=expand_size, out_features=expand_size)
        self.se_1_1 = nn.ReLU(inplace=True)

        self.se_2_0 = nn.Linear(in_features=expand_size, out_features=expand_size)
        self.se_2_1 = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        x = self.se_0_0(x)
        x = self.se_0_1(x)

        x = self.se_1_0(x)
        x = self.se_1_1(x)

        x = self.se_2_0(x)
        x = self.se_2_1(x)
        x = torch.unsqueeze(x, -1)
        x = torch.unsqueeze(x, -1)
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, dw_kernel_size, expand_size, squeeze_excite,
                 nonlinearity, stride):
        super(Bottleneck, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_size = expand_size
        self.squeeze_excite = squeeze_excite
        self.stride = stride
        self.dw_kernel_size = dw_kernel_size

        if nonlinearity == 'hardswish':
            self.Nonliearity = nn.Hardswish
        elif nonlinearity == 'relu':
            self.Nonliearity = nn.ReLU
        else:
            raise RuntimeError("No such nonlinearity!")

        # 1x1 Conv2d + NL
        self.bottleneck_0_0 = nn.Conv2d(in_channels=in_channels, out_channels=expand_size, kernel_size=(1, 1),
                                        bias=False)
        self.bottleneck_0_1 = nn.BatchNorm2d(num_features=expand_size)
        self.bottleneck_0_2 = self.Nonliearity(inplace=True)

        # Dwise + NL
        self.bottleneck_1_0 = nn.Conv2d(in_channels=expand_size, out_channels=expand_size,
                                        kernel_size=self.dw_kernel_size,
                                        stride=self.stride, padding=self.dw_kernel_size[0] // 2, groups=expand_size,
                                        bias=False)
        self.bottleneck_1_1 = nn.BatchNorm2d(num_features=expand_size)

        # Squeeze-Excite
        if self.squeeze_excite:
            self.squeeze_excite_0 = SqueezeExciteModule(
                expand_size=expand_size
            )
        else:
            self.squeeze_excite_0 = nn.Identity()

        # Final 1x1 Conv2d
        self.bottleneck_final_0 = nn.Conv2d(in_channels=expand_size, out_channels=out_channels, kernel_size=(1, 1),
                                            bias=False)
        self.bottleneck_final_1 = nn.BatchNorm2d(num_features=out_channels)

        # Downsampling first layer
        self.bottleneck_final_2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=(1, 1), stride=self.stride, bias=False)

    def forward(self, x):
        x_0 = self.bottleneck_0_0(x)
        x_0 = self.bottleneck_0_1(x_0)
        x_0 = self.bottleneck_0_2(x_0)

        x_0 = self.bottleneck_1_0(x_0)
        x_0 = self.bottleneck_1_1(x_0)

        x_1 = self.squeeze_excite_0(x_0)
        x_0 = x_0 * x_1

        x_0 = self.bottleneck_final_0(x_0)
        x_0 = self.bottleneck_final_1(x_0)
        x_b = self.bottleneck_final_2(x)
        return x_0.add(x_b)


class MobileNetV3(nn.Module):
    def __init__(self, width_mult=1.0, classifier=True, classifier_out_features=1000):
        super(MobileNetV3, self).__init__()

        conv_0_0 = nn.Conv2d(in_channels=3, out_channels=int(16 * width_mult),
                             kernel_size=(3, 3), stride=2, padding=3 // 2, bias=False)
        conv_0_1 = nn.BatchNorm2d(num_features=int(16 * width_mult))
        conv_0_2 = nn.Hardswish(inplace=True)

        conv_1_0 = Bottleneck(in_channels=int(16 * width_mult),
                              out_channels=int(16 * width_mult), dw_kernel_size=(3, 3),
                              expand_size=16, squeeze_excite=True, nonlinearity='relu', stride=2)
        conv_2_0 = Bottleneck(in_channels=int(16 * width_mult),
                              out_channels=int(24 * width_mult), dw_kernel_size=(3, 3),
                              expand_size=72, squeeze_excite=False, nonlinearity='relu', stride=2)
        conv_3_0 = Bottleneck(in_channels=int(24 * width_mult),
                              out_channels=int(24 * width_mult), dw_kernel_size=(3, 3),
                              expand_size=88, squeeze_excite=False, nonlinearity='relu', stride=1)
        conv_4_0 = Bottleneck(in_channels=int(24 * width_mult),
                              out_channels=int(40 * width_mult), dw_kernel_size=(5, 5),
                              expand_size=96, squeeze_excite=True, nonlinearity='hardswish', stride=2)
        conv_5_0 = Bottleneck(in_channels=int(40 * width_mult),
                              out_channels=int(40 * width_mult), dw_kernel_size=(5, 5),
                              expand_size=240, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        conv_6_0 = Bottleneck(in_channels=int(40 * width_mult),
                              out_channels=int(40 * width_mult), dw_kernel_size=(5, 5),
                              expand_size=240, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        conv_7_0 = Bottleneck(in_channels=int(40 * width_mult),
                              out_channels=int(48 * width_mult), dw_kernel_size=(5, 5),
                              expand_size=120, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        conv_8_0 = Bottleneck(in_channels=int(48 * width_mult),
                              out_channels=int(48 * width_mult), dw_kernel_size=(5, 5),
                              expand_size=144, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        conv_9_0 = Bottleneck(in_channels=int(48 * width_mult),
                              out_channels=int(96 * width_mult), dw_kernel_size=(5, 5),
                              expand_size=288, squeeze_excite=True, nonlinearity='hardswish', stride=2)
        conv_10_0 = Bottleneck(in_channels=int(96 * width_mult),
                               out_channels=int(96 * width_mult), dw_kernel_size=(5, 5),
                               expand_size=576, squeeze_excite=True, nonlinearity='hardswish', stride=1)
        conv_11_0 = Bottleneck(in_channels=int(96 * width_mult),
                               out_channels=int(96 * width_mult), dw_kernel_size=(5, 5),
                               expand_size=576, squeeze_excite=True, nonlinearity='hardswish', stride=1)

        conv_12_0 = nn.Conv2d(in_channels=int(96 * width_mult), out_channels=int(576 * width_mult),
                              kernel_size=(1, 1), bias=False)
        conv_12_1 = nn.Hardswish(inplace=True)
        conv_12_2 = nn.BatchNorm2d(num_features=int(576 * width_mult))

        self.features = nn.Sequential(
            conv_0_0,
            conv_0_1,
            conv_0_2,
            conv_1_0,
            conv_2_0,
            conv_3_0,
            conv_4_0,
            conv_5_0,
            conv_6_0,
            conv_7_0,
            conv_8_0,
            conv_9_0,
            conv_10_0,
            conv_11_0,
            conv_12_0,
            conv_12_1,
            conv_12_2
        )

        if classifier:
            self.classifiers = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten(start_dim=1),
                nn.Linear(int(576 * width_mult), int(1024 * width_mult)),
                nn.Dropout(p=0.2),
                nn.Linear(int(1024 * width_mult), classifier_out_features)
            )
        else:
            self.classifiers = nn.Identity()

    def forward(self, x):
        x = self.features(x)
        x = self.classifiers(x)
        return x



class VOC(Dataset):
    def __init__(self, root_dir, transform=None):
        super(VOC, self).__init__()
        self.transform = transform

        self.img_path_list = []
        self.img_class_list = []
        self.labels = []

        if os.path.isfile('dataset.pkl'):
            print("Loading annotations from cache")
            self.img_path_list, self.img_class_list, self.labels = pickle.load(open('dataset.pkl', 'rb'))
        else:
            self.load_list(root_dir)
            print("Saved annotations to  cache")

        self.img_class_list = from_numpy(self.img_class_list)

    def load_list(self, root_dir):
        annotation_dir = os.path.join(root_dir, 'Annotations')
        images_dir = os.path.join(root_dir, 'JPEGImages')

        label_map = {}
        for xml_filename in tqdm(glob.glob(os.path.join(annotation_dir, '*.xml')), desc='클래스 리스트 생성 작업'):
            with open(xml_filename, 'r') as f:
                xml_body = f.read()
                root = ElementTree.fromstring(xml_body)
                for item in root.findall('object'):
                    label_map[item.find('name').text] = True
        self.labels = list(sorted(label_map.keys()))

        for xml_filename in tqdm(glob.glob(os.path.join(annotation_dir, '*.xml')), desc='이미지 어노테이션 읽기 작업'):
            with open(xml_filename, 'r') as f:
                xml_body = f.read()
                root = ElementTree.fromstring(xml_body)
                image_file_path = os.path.join(images_dir, root.find('filename').text)
                for item in root.findall('object'):
                    label_index = self.labels.index(item.find('name').text)
                    assert label_index != -1

                    self.img_path_list.append(image_file_path)
                    self.img_class_list.append(label_index)

        self.img_class_list = np.array(self.img_class_list, dtype=np.int64)
        pickle.dump((self.img_path_list, self.img_class_list, self.labels), open('dataset.pkl', 'wb'))

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_path_list[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = self.img_class_list[idx]
        return image, label


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpu', '-g', default=False, action='store_true', help='Enables GPU')
    parser.add_argument('--epochs', '-e', default=200, type=int, help='Epochs (default: 200)')
    parser.add_argument('--num-workers', '-p', default=0, type=int, help='num_workers (default: 0)')
    parser.add_argument('--continue-weight', '-c', default=None, type=str, help='load weight and continue training')
    parser.add_argument('--init-lr', '-ilr', default=False, action='store_true', help='Sets default learning rate rather than weight value')
    args = parser.parse_args()

    # CUDA-related stuffs
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    model = MobileNetV3(width_mult=1.0, classifier=True, classifier_out_features=1000).float().to(device)
    torchsummary.summary(model, input_size=(3, 224, 224), device=device.type)

    labels = LabelReader(label_file_path='imagenet_label.list').load_label()
    datasets = ImageNet(
        labels=labels,
        root_dir=r'C:\Dataset\ILSVRC\Data\CLS-LOC\train',
        transform=transforms.Compose([
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                # mean=[0.485, 0.456, 0.406],
                # std=[0.229, 0.224, 0.225]
                # VOC-mean, stds
                mean=[0.4547857, 0.4349471, 0.40525291],
                std=[0.12003352, 0.12323549, 0.1392444]
            )
        ])
    )
    # datasets = VOC(
    #     root_dir=r'C:\Dataset\VOCdevkit\VOC2008',
    #     transform=transforms.Compose([
    #         transforms.RandomResizedCrop((224, 224)),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize(
    #             mean=[0.4547857, 0.4349471, 0.40525291],
    #             std=[0.12003352, 0.12323549, 0.1392444]
    #         )
    #     ])
    # )

    validset_items = int(len(datasets) * 0.1)
    trainset_items = len(datasets) - validset_items
    train_dataset, valid_dataset = torch.utils.data.random_split(
        datasets, [trainset_items, validset_items]
    )

    dataloader_extras = {'shuffle': True, 'num_workers': args.num_workers, 'pin_memory': True, 'drop_last': False}
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=384, **dataloader_extras)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=max(1, 192), **dataloader_extras)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.009375, weight_decay=1e-5)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.9999)

    total_epochs = args.epochs
    summary_writer = torch.utils.tensorboard.SummaryWriter(
        log_dir=os.path.join('logs', '%s-lr%.6f'% (
                datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
                optimizer.param_groups[0]['lr']
            )
        )
    )

    if args.continue_weight:
        if not os.path.isfile(args.continue_weight):
            print("Weight file %s not found!" % args.continue_weight)
            exit(-1)

        checkpoint = torch.load(args.continue_weight)
        c_epoch = checkpoint['epoch'] + 1
        c_model_state_dict = checkpoint['model_state_dict']
        c_optimizer_state_dict = checkpoint['optimizer_state_dict']
        c_loss = checkpoint['loss']

        model.load_state_dict(c_model_state_dict)
        optimizer.load_state_dict(c_optimizer_state_dict)
        epoch_range = range(c_epoch, total_epochs)
        print("Continuing epoch %03d, learning rate %.4f" % (c_epoch, optimizer.param_groups[0]['lr']))

        if args.init_lr:
            print("Reinitializing lr to %.4f" % args.learning_rate)
            for g in optimizer.param_groups:
                g['lr'] = args.learning_rate
    else:
        epoch_range = range(total_epochs)

    run_name = "MobileNetV3-Large-%s" % (datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    epoch_val_accuracies = []
    for epoch in epoch_range:
        tr = tqdm(enumerate(train_dataloader), total=len(train_dataloader), ncols=160,
                  desc='[Epoch %04d/%04d] Spawning Workers' % (epoch + 1, total_epochs))
        summary_writer.add_scalar('Hyperparameters/Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        accuracies = []
        losses = []
        model.train(True)
        for i, (image, label) in tr:
            if device.type != label.device.type:
                image = image.to(device)
                label = label.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            ema.update(model.parameters())

            with torch.no_grad():
                accuracy = torch.mean(torch.eq(torch.argmax(output, dim=1), label).int().float())
                accuracies.append(accuracy.item())
                losses.append(loss.item())
                summary_writer.add_scalar('Training/Batch Loss', losses[-1], epoch * len(train_dataloader) + i)
                summary_writer.add_scalar('Training/Batch Accuracy', accuracies[-1], epoch * len(train_dataloader) + i)

                tr.set_description("[Epoch %04d/%04d][Image Batch %04d/%04d] Training Loss: %.4f Training Accuracy: %.4f" %
                                   (epoch + 1, total_epochs, i, len(train_dataloader), np.mean(losses), np.mean(accuracies)))

        train_loss_value = np.mean(losses)
        train_accuracy_value = np.mean(accuracies)

        summary_writer.add_scalar('Training/Epoch Loss', train_loss_value, epoch)
        summary_writer.add_scalar('Training/Epoch Accuracy', train_accuracy_value, epoch)

        vl = tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), ncols=160,
                  desc='[Epoch %04d/%04d] Spawning Workers' % (epoch + 1, total_epochs))

        accuracies = []
        losses = []
        model.train(False)
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        for i, (image, label) in vl:
            if device.type != label.device.type:
                image = image.to(device)
                label = label.to(device)

            output = model(image)
            loss = criterion(output, label)
            accuracy = torch.mean(torch.eq(torch.argmax(output, dim=1), label).int().float())

            accuracies.append(accuracy.item())
            losses.append(loss.item())

            vl.set_description("[Epoch %04d/%04d][Image Batch %04d/%04d] Validation Loss: %.4f, Accuracy: %.4f" %
                        (epoch + 1, total_epochs, i, len(valid_dataloader), np.mean(losses), np.mean(accuracies)))
        ema.restore(model.parameters())

        for i in range(len(losses)):
            summary_writer.add_scalar('Validation/Batch Loss', losses[i], epoch * len(valid_dataloader) + i)
            summary_writer.add_scalar('Validation/Batch Accuracy', accuracies[i], epoch * len(valid_dataloader) + i)

        val_loss_value = np.mean(losses)
        val_acc_value = np.mean(accuracies)

        summary_writer.add_scalar('Validation/Epoch Loss', val_loss_value, epoch)
        summary_writer.add_scalar('Validation/Epoch Accuracy', val_acc_value, epoch)

        if epoch % 3 == 0 and epoch != 0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] * 0.99
            print("[Epoch %04d/%04d] Reducing learning rate to %.6f" % (epoch + 1, args.epochs, g['lr']))

        if not epoch_val_accuracies or np.max(epoch_val_accuracies) < val_acc_value:
            if not os.path.isdir('.checkpoints'):
                os.mkdir('.checkpoints')
            save_filename = os.path.join('.checkpoints',
                 '%s-epoch%04d-train_loss%.6f-val_loss%.6f-val_acc%.6f.zip' % (run_name, epoch, train_loss_value, val_loss_value, val_acc_value))
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, save_filename)
            print("[Epoch %04d/%04d] Saved checkpoint %s" % (epoch + 1, args.epochs, save_filename))

        epoch_val_accuracies.append(val_acc_value)
