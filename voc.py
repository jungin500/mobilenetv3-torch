import glob
import math
import os
import pickle

import h5py
import torch
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm

CACHE_BASE_DIR = "." + os.sep + ".cache"
IMG_PATH_LIST_PKL_FILENAME = CACHE_BASE_DIR + os.sep + 'img_path_list'
IMG_CLASS_LIST_PKL_FILENAME = CACHE_BASE_DIR + os.sep + 'img_class_list'


class VOC(torch.utils.data.Dataset):
    def __init__(self, labels, root_dir, device, input_size, transform=None, use_cache=True, dataset_usage_pct=1.0):
        super(VOC, self).__init__()

        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform
        self.device = device
        self.input_size = input_size

        img_path_list_filename_pct = IMG_PATH_LIST_PKL_FILENAME + '_' + str(dataset_usage_pct) + '.pkl'
        img_class_list_filename_pct = IMG_CLASS_LIST_PKL_FILENAME + '_' + str(dataset_usage_pct) + '.pkl'

        print(f"데이터셋 중 {int(dataset_usage_pct * 1000) / 10}%를 사용합니다.")

        if use_cache and os.path.isfile(img_path_list_filename_pct) and os.path.isfile(img_class_list_filename_pct):
            try:
                for _ in tqdm(range(1), desc='이미지 파일 리스트 읽기 작업 (Using cache)'):
                    with open(img_path_list_filename_pct, 'rb') as f:
                        self.img_path_list = pickle.load(f)
                    with open(img_class_list_filename_pct, 'rb') as f:
                        self.img_class_list = pickle.load(f)
            except Exception as e:
                print(e)
                self.load_list(
                    dataset_usage_pct=dataset_usage_pct,
                    save_cache=True,
                    img_path_pkl_name=img_path_list_filename_pct,
                    img_class_pkl_name=img_class_list_filename_pct
                )
        else:
            self.load_list(
                dataset_usage_pct=dataset_usage_pct,
                save_cache=True,
                img_path_pkl_name=img_path_list_filename_pct,
                img_class_pkl_name=img_class_list_filename_pct
            )

        # print(f"리스트 초반부 10개:", self.img_class_list[:10])
        # print(f"리스트 마지막 10개:", self.img_class_list[-10:])

    def load_list(self, dataset_usage_pct, save_cache=True, img_path_pkl_name=None, img_class_pkl_name=None):
        self.img_path_list = []
        self.img_class_list = []

        label_index = 0
        for annotation_item in tqdm(glob.glob(self.root_dir + os.sep + '*.anot'), desc='이미지 파일 리스트 부분 읽기 작업'):
            with open(annotation_item, 'r') as f:
                annotation_body = [k.rstrip() for k in f.read().strip().split('\n')]
                filename = annotation_body[0]
                image_width, image_height = annotation_body[1].split(' ')
                objects = annotation_body[2:]
                object_whs = [[float(v) for v in r.split(' ')[2:4]] for r in objects]
                object_volumes = [r[0] * r[1] for r in object_whs]
                max_volume_object_idx = np.argmax(object_volumes, axis=0)  # we will only use that object
                
                # for object_line in objects:
                object_line = objects[max_volume_object_idx]
                
                object_pickles = object_line.split(' ')
                xc, yc, width, height = [float(k) for k in object_pickles[:4]]
                one_hot = [float(k) for k in object_pickles[4:]]

                # But we only need one_hot ones!
                class_id = np.array(one_hot).argmax(axis=0).item()
                self.img_path_list.append(filename)
                self.img_class_list.append(class_id)
            label_index += 1

        if save_cache:
            if img_path_pkl_name is None or img_class_pkl_name is None:
                print("저장할 캐시 이름이 비어있습니다. 저장하지 않습니다.")
                return

            print("캐시 작성 중 ...")
            if not os.path.exists(CACHE_BASE_DIR):
                os.mkdir(CACHE_BASE_DIR)
            with open(img_path_pkl_name, 'wb') as f:
                pickle.dump(self.img_path_list, f, pickle.HIGHEST_PROTOCOL)
            with open(img_class_pkl_name, 'wb') as f:
                pickle.dump(self.img_class_list, f, pickle.HIGHEST_PROTOCOL)

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
        if self.device.type != image.device.type:
            image = image.to(self.device)
        if self.device.type != label.device.type:
            label = label.to(self.device)
        return image, label

        # OpenCV version (minimizing CPU usage) ? seconds
        # if self.transform is not None:
        #     raise RuntimeError("Transform Not Supported!")
        # image = cv2.imread(self.img_path_list[idx], cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = cv2.resize(image, (self.input_size, self.input_size), cv2.INTER_CUBIC)
        # image = np.transpose(image, [2, 0, 1])
        # image = torch.tensor(image, device=self.device, dtype=torch.float) / 255.
        # label = torch.tensor(self.img_class_list[idx], device=self.device)
        #
        # return image, label

        # # OpenCV version 2 (can support transforms - but extra copy incurs here!) 25 seconds
        # image = cv2.imread(self.img_path_list[idx], cv2.IMREAD_COLOR)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(image)
        # if self.transform is not None:
        #     image = self.transform(image)
        # if self.device.type != 'cpu':
        #     image = image.to(self.device)
        # label = torch.tensor(self.img_class_list[idx], device=self.device)
        # label = torch.unsqueeze(label, 0)
        # return image, label
