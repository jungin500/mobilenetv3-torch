from argparse import ArgumentParser

import h5py
import torch
from torchvision import transforms
from tqdm import tqdm
from time import time

from Dataloader import ILSVRC2012TaskOneTwoDataset
from ILSVRC2012Preprocessor import LabelReader

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--label-list', default='label.list',
                        help='label.list (e.g. File filled with lines containing "n03251766|dryer, drier\\n", ...) ')
    parser.add_argument('--root-dir', default=r'S:\ILSVRC2012-CLA-DET\ILSVRC',
                        help=r'Annotation root directory which contains folder (default: S:\ILSVRC2012-CLA-DET\ILSVRC)')
    parser.add_argument('--files', '-f', type=int, default=200,
                        help='Files to save (Will be a \'epoch\' to implemetation')
    parser.add_argument('--batch-size', type=int, default=128, help='Images per batch (default: 128)')
    parser.add_argument('--input-size', '-i', type=int, default=224,
                        help='Image input size (default: 224, candidates: 224, 192, 160, 128)')
    parser.add_argument('--dataset-pct', '-p', type=float, default=1.0,
                        help='Dataset usage percentage in 0.0~1.0 range (default: 1.0)')
    args = parser.parse_args()

    labels = LabelReader(label_file_path=args.label_list).load_label()

    dataset = ILSVRC2012TaskOneTwoDataset(
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
        use_cache=True,
        dataset_usage_pct=args.dataset_pct
    )

    batch_size = args.batch_size
    batch_count = (len(dataset) // batch_size)
    dataset_len = batch_size * batch_count  # could not be equal to dataset original size

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=0)

    print('Total %d images in dataset, using %d images' % (len(dataset), dataset_len))

    # Build h5py using Dataset items!
    for iteration in range(args.files):
        # Each H5 file is approximately ~1.2GB (not compressed)
        with h5py.File(
                r'D:\dataset-ilsvrc2012-taskonetwo-20210227-%ditems-compressed-%03d.hdf5' % (
                dataset_len, iteration),
                'w') as h5py_file:


            h5py_image_dataset = h5py_file.create_dataset('images', (dataset_len, 3, args.input_size, args.input_size),
                                                          chunks=(1, 3, args.input_size, args.input_size), dtype='f',
                                                          compression="gzip", compression_opts=9)
            h5py_labels_dataset = h5py_file.create_dataset('labels', (dataset_len, 1), dtype='f', compression="gzip",
                                                           compression_opts=9)

            PERFORMANCE_MEASUREMENT = False

            begin_time = time()
            for i, data in tqdm(enumerate(dataloader), desc="Dataset iteration [%d/%d]" % (iteration + 1, args.files),
                                total=dataset_len // batch_size):

                time_diff = time() - begin_time
                if PERFORMANCE_MEASUREMENT: print("\nOne batch loaded (%d images), took %d seconds" % (batch_size, time_diff))
                begin_time = time()

                if i >= (dataset_len // batch_size):
                    break

                image, label = data
                image = image.squeeze().numpy().astype('float32')
                label = label.numpy().astype('int64')

                time_diff = time() - begin_time
                if PERFORMANCE_MEASUREMENT: print("Numpy conversion took %d seconds" % (time_diff,))
                begin_time = time()

                h5py_image_dataset[batch_size * i:batch_size * (i + 1)] = image
                h5py_labels_dataset[batch_size * i:batch_size * (i + 1)] = label

                time_diff = time() - begin_time
                if PERFORMANCE_MEASUREMENT: print("Image writing took %d seconds\n" % (time_diff,))
                begin_time = time()
