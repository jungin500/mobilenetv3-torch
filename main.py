import torch
from torchvision import transforms
from argparse import ArgumentParser

from Dataloader import ILSVRC2012TaskOneTwoDataset
from Model import MobileNetV3
from ILSVRC2012Preprocessor import LabelReader

import torchsummary

if __name__ == '__main__':
    parser = ArgumentParser()
    # parser.add_argument('--annotation-root', help='Annotation root directory (e.g. ILSVRC/Annotations/DET/train/ILSVRC2013_train)')
    # parser.add_argument('--label-names', default='label.list', help='label.list filename (e.g. File filled with lines containing "n03251766|dryer, drier\\n", ...) ')
    parser.add_argument('--batch-size', '-b', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--no-cache', default=False, action='store_true', help='Do not use cache while loading image datas')
    args = parser.parse_args()

    torch.multiprocessing.set_start_method('spawn')

    labels = LabelReader(label_file_path='label.list').load_label()
    datasets = ILSVRC2012TaskOneTwoDataset(
        labels=labels,
        # root_dir=r'S:\ILSVRC2012-CLA-DET\Miniset',
        root_dir='/mnt/s/ILSVRC2012-CLA-DET/Miniset',
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]),
        use_cache=not args.no_cache,
        dataset_usage_pct=1.00
    )

    dataloader = torch.utils.data.DataLoader(datasets, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)  # do not use num_workers on Windows!

    model = torch.nn.Sequential(
        MobileNetV3(size='small', out_features=3),
        torch.nn.Softmax()
    )
    model.to('cuda')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.1, momentum=1e-5, weight_decay=1e-5)
    torchsummary.summary(model, (3, 224, 224))

    EPOCHS = 10

    for epoch in range(EPOCHS):
        running_loss = 0
        for i, data in enumerate(dataloader):
            input, label = data

            optimizer.zero_grad()
            output = model(input)
            loss = criterion(output, label.squeeze())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0

    print("Training success")
