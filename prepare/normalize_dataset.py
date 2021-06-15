import cv2
from prepare import VOC, LabelReader, ImageNet
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    # dataset = VOC(root_dir=r'C:\Dataset\VOCdevkit\VOC2008')
    labels = LabelReader(label_file_path='imagenet_label.list').load_label()
    dataset = ImageNet(labels=labels, root_dir=r'C:\Dataset\ILSVRC\Data\CLS-LOC\train')
    image_list_nodupe = list(dict.fromkeys(dataset.img_path_list))

    values = np.zeros((3, len(image_list_nodupe)))

    image_index = 0
    for image_path in tqdm(image_list_nodupe, desc='Calculate average of image'):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_by_channels = cv2.split(image)

        values[0, image_index] = np.mean(image_by_channels[0])
        values[1, image_index] = np.mean(image_by_channels[1])
        values[2, image_index] = np.mean(image_by_channels[2])

        image_index += 1

    mean = np.mean(values, axis=1) / 255
    std = np.std(values, axis=1) / 255

    print("Mean: ", mean)
    print("StdDev: ", std)