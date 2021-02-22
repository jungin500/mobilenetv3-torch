# -*- coding: utf-8 -*-

import os
from glob import glob
from xml.etree import ElementTree
from tqdm import tqdm
from argparse import ArgumentParser


class LabelReader(object):
    def __init__(self, label_file_path):
        self.label_file_path = label_file_path

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


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--annotation-root', help='Annotation root directory (e.g. ILSVRC/Annotations/DET/train/ILSVRC2013_train)')
#     parser.add_argument('--label-names', default='label.list', help='label.list filename (e.g. File filled with lines containing "n03251766|dryer, drier\\n", ...) ')
#     args = parser.parse_args()
#
#     reader = LabelReader(args.label_names)
#     label_map = reader.load_label()
#
#     if label_map is None:
#         raise RuntimeError("라벨 파일이 존재하지 않습니다!")
#
#     PARSED_ANNOTATION_DIR = '.' + os.sep + '.annotations'
#     if not os.path.exists(PARSED_ANNOTATION_DIR):
#         os.mkdir(PARSED_ANNOTATION_DIR)
#
#     # fetch XML file names
#     xml_file_list = glob(args.annotation_root + os.sep + '*.xml')
#
#     for xml_filename in tqdm(xml_file_list, desc='(2/2) 레이블 매칭 및 신뢰도 작업'):
#         with open(xml_filename, 'r') as f:
#             xml_body = f.read()
#
#             # Parse XML file
#             root = ElementTree.fromstring(xml_body)
#             image_path = os.sep.join(xml_filename.split(os.sep)[:-2]) + os.sep + 'JPEGImages' + os.sep + root.find(
#                 'filename').text
#             image_width =int(root.find('size').find('width').text)
#             image_height = int(root.find('size').find('height').text)
#
#             object_list = root.findall('object')
#             object_label_list = [item.find('name').text for item in object_list]
#             xyminmax_list = [item.find('bndbox') for item in object_list]
#             xyminmax_list = [
#                 [int(float(x)) for x in [obj.find('xmin').text, obj.find('xmax').text, obj.find('ymin').text, obj.find('ymax').text]]
#                 for obj in xyminmax_list]
#
#             xycenter_wh_list = [
#                 # Effecitively each are (x_center, y_center, width, height)
#                 ((obj[0] + obj[1]) / 2, (obj[2] + obj[3]) / 2, obj[1] - obj[0], obj[3] - obj[2])
#                 for obj in xyminmax_list
#             ]
#
#             xycenter_wh_list_norm = [
#                 # Effectively each are normalized to (0, 1]. center xy position are also normalized
#                 (obj[0] / image_width, obj[1] / image_height, obj[2] / image_width, obj[3] / image_height)
#                 for obj in xycenter_wh_list
#             ]
#
#             object_list_len = len(object_list)
#
#             xml_filename_nopath = xml_filename.split(os.sep)[-1]
#             with open(PARSED_ANNOTATION_DIR + os.sep + xml_filename_nopath.replace('.xml', '.anot'), 'w') as wf:
#                 wf.write(image_path + '\n')
#                 wf.write(f'{image_width} {image_height}\n')
#                 for object_idx in range(object_list_len):
#                     object_name = object_label_list[object_idx]
#                     xycenter_wh_norm = list(xycenter_wh_list_norm[object_idx])
#
#                     object_one_hot_list = [0.0] * label_count
#                     object_one_hot_list[label_list.index(object_name)] = 1.
#
#                     wf.write(' '.join([str(x) for x in xycenter_wh_norm]) + ' ')
#                     wf.write(' '.join([str(x) for x in object_one_hot_list]) + '\n')
#                 wf.flush()