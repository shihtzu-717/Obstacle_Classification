import os
import cv2
import pickle
import random
import traceback
import argparse
import glob
import numpy as np

from enum import Enum
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor

RawData = namedtuple('RawData', 'data_set, label, image_path, idx, class_id, bbox')
CropData = namedtuple('CropData', 'data_set, label, image_path')

class Padding(Enum):
    BBOX = 'BBOX'
    PIXEL = 'PIXEL'
    IMAGE = 'IMAGE'
    FIX = 'FIX'
    FIX2 = 'FIX2'
    
    def __str__(self):
        return self.value    

def xywh2xyxy(bbox, width, height):
    x, y, w, h = bbox
    x1 = int(np.clip(x - w / 2, 0, 1) * width)
    y1 = int(np.clip(y - h / 2, 0, 1) * height)
    x2 = int(np.clip(x + w / 2, 0, 1) * width)
    y2 = int(np.clip(y + h / 2, 0, 1) * height)
    return x1, y1, x2, y2

def crop_image(image_path, bbox, padding, padding_size, use_shift, use_bbox, output_path=None, imsave=True):
    try:
        image = cv2.imread(str(image_path))
        height, width, _ = image.shape
        x1, y1, x2, y2 = xywh2xyxy(bbox, width, height)
        if use_bbox:
            image = cv2.rectangle(
                image,
                (x1 - 1, y1 - 1),
                (x2, y2),
                (0, 0, 255),
                1
            )
        if padding_size > 0:
            if padding == Padding.BBOX or padding =='BBOX':
                pad_x = int(padding_size * (x2 - x1))
                pad_y = int(padding_size * (y2 - y1))
            elif padding == Padding.PIXEL or padding =='PIXEL':
                pad_x = int(padding_size)
                pad_y = int(padding_size)
            elif padding == Padding.IMAGE or padding =='IMAGE':
                pad_x = int(padding_size * width)
                pad_y = int(padding_size * height)
            elif padding == Padding.FIX or padding =='FIX':
                pad_x = max(int((padding_size - (x2 - x1)) / 2), 0)
                pad_y = max(int((padding_size - (y2 - y1)) / 2), 0)
            elif padding == Padding.FIX2 or padding =='FIX2':
                padding_size = max(padding_size, x2 - x1, y2 - y1)
                pad_x = max(int((padding_size - (x2 - x1)) / 2), 0)
                pad_y = max(int((padding_size - (y2 - y1)) / 2), 0)
            else:
                raise Exception(f'Unknown padding {padding}')
            if use_shift:
                x1 -= pad_x
                y1 -= pad_y
                x2 += pad_x
                y2 += pad_y
                if x1 < 0:
                    x2 -= x1
                    x1 = 0
                if width < x2:
                    x1 -= x2 - width
                    x2 = width
                if y1 < 0:
                    y2 -= y1
                    y1 = 0
                if height < y2:
                    y1 -= y2 - height
                    y2 = height
            else:
                x1 = np.clip(x1 - pad_x, 0, width)
                y1 = np.clip(y1 - pad_y, 0, height)
                x2 = np.clip(x2 + pad_x, 0, width)
                y2 = np.clip(y2 + pad_y, 0, height)
        image = image[y1:y2, x1:x2]
        if imsave:
            cv2.imwrite(str(output_path), image)
            return True
        else:
            return image
    except:
        traceback.print_exc()
        return False

def get_image_path(data_root, data, is_train, is_output):
    if is_output:
        image_path = data.image_path.with_name(f'{data.idx}_{data.image_path.name}')
    else:
        image_path = data.image_path
    if is_train:
        return data_root / data.label / data.data_set / image_path
    else:
        return data_root / data.data_set / data.label / image_path

def find_image(anno):
    ext_list = ['.jpg', '.png']
    for ext in ext_list:
        image = anno.with_suffix(ext)
        if image.exists():
            return image
    raise Exception(f'image not found: {anno}')

def find_annotation(img):
    annot = Path(img)
    annot = annot.with_suffix('.txt')
    if annot.exists():
        return annot
    raise Exception(f'image not found: {img}')

def read_annotation(annotation):
    with annotation.open('r', encoding='utf-8') as rf:
        for line_idx, line in enumerate(rf.readlines()):
            line = line.strip()
            data = line.split()
            if len(data) == 0:
                continue
            class_id = int(data[0])
            bbox = [float(v) for v in data[1:5]]
            yield line_idx, class_id, bbox

def make_dataset_file(dataset_file):
    data_list = []
    with open(dataset_file, 'r') as rf:
        data_lines = [i.strip() for i in rf.readlines()]
    for image_path in data_lines:
        data_root = '/'.join(image_path.split('/')[:-3])
        annot_path = image_path.replace('images', 'annotations')
        annot_path = find_annotation(annot_path)
        image_path = Path(image_path)
        image_path = image_path.relative_to(data_root)

        for line_idx, class_id, bbox in read_annotation(annot_path):
            label = annot_path.parts[-3]
            raw_data = RawData(
                data_root, label, image_path,
                line_idx, class_id, bbox
            )
            data_list.append(raw_data)
    return data_list

def make_crop_dataset_file(dataset_file):
    data_list = []
    with open(dataset_file, 'r') as rf:
        data_lines = [i.strip() for i in rf.readlines()]
    for image_path in data_lines:
        data_root = '/'.join(image_path.split('/')[:-3])
        image_path = Path(image_path)
        image_path = image_path.relative_to(data_root)
        label = image_path.parts[-3]
        raw_data = RawData(data_root, label, image_path, 0, 5, [])
        data_list.append(raw_data)
    return data_list

def make_crop_dataset_list(data_root, label_list=None, split_info=None):
    data_list = defaultdict(list)
    data_root = Path(data_root).resolve()
    for image_path in split_info.keys():
        image_path = Path(image_path)
        try:
            label = image_path.parts[-3]
        except:
            label = 'pred'
        try:                
            split_tag = split_info[image_path]
        except:
            print(f'{image_path} is not exist.')
            continue
        # print(data_root, label, image_path, split_tag)
        # crop_data = CropData(data_root, label, image_path)
        # data_list[split_tag].append(crop_data)
        raw_data = RawData(data_root, label, image_path, 0, 5, [])
        data_list[split_tag].append(raw_data)
    return data_list

def make_list(data_root, label_list=None, split_info=None):
    data_list = defaultdict(list)
    # print('dataset\tlabel\tclass_id\ttest_cnt\tval_cnt\ttrain_cnt')
    if split_info is None:
        for annotation in data_root.glob('*.txt'):
            image_path = find_image(annotation)
            image_path = image_path.relative_to(data_root)
            annotation = annotation.relative_to(data_root)
            for line_idx, class_id, bbox in read_annotation(annotation):
                raw_data = (
                    data_root, image_path, 
                    line_idx, class_id, bbox
                )
                data_list['data'].append(raw_data)

    elif label_list is None:
        for annotation in data_root.glob('**/*.txt'):
            # annotation = annotation.relative_to(data_root)
            try:
                image_path = find_image(annotation)
            except:
                newanno = Path(str(annotation).replace('annotations', 'images'))
                image_path = find_image(newanno)

            image_path = image_path.relative_to(data_root)
            try:                
                split_tag = split_info[image_path]
            except:
                print(f'{image_path} is not exist.')
                continue
            # image_path = image_path.relative_to(data_root)
            # split_tag = split_info[annotation.relative_to(data_root)]
            # annotation = annotation.relative_to(data_root)
            for line_idx, class_id, bbox in read_annotation(annotation):
                label = annotation.parts[-3]
                raw_data = RawData(
                    data_root, label, image_path, 
                    line_idx, class_id, bbox
                )
                data_list[split_tag].append(raw_data)

    else:
        for data_sub_root in data_root.glob('*'):
            if not data_sub_root.is_dir():
                continue
            for label in label_list:
                annotation_root = data_sub_root / label
                count_dict = defaultdict(lambda: defaultdict(int))
                for annotation in annotation_root.glob('**/*.txt'):
                    image_path = find_image(annotation)
                    image_path = image_path.relative_to(annotation_root)
                    split_tag = split_info[annotation.relative_to(data_root)]
                    for line_idx, class_id, bbox in read_annotation(annotation):
                        raw_data = RawData(
                            data_sub_root.name, label, image_path,
                            line_idx, class_id, bbox
                        )
                        count_dict[raw_data.class_id][split_tag] += 1
                        data_list[split_tag].append(raw_data)
                for class_id, count in count_dict.items():
                    test_cnt = count['test']
                    val_cnt = count['val']
                    train_cnt = count['train']
                    print(f'{data_sub_root.name}\t{label}\t{class_id}\t{test_cnt}\t{val_cnt}\t{train_cnt}')
    return data_list

def split_data(data_root, test_ratio, val_ratio, label_list=None, file_write=False, use_cropimg=False):
    split_info_path = data_root / f'split_info_{test_ratio}_{val_ratio}'
    
    train_ratio = 1 - (test_ratio + val_ratio)
    split_population = 'test', 'val', 'train'
    split_weights = test_ratio, val_ratio, train_ratio

    if use_cropimg:
        split_info = []
        types = ('*.jpg', '*.jpeg', '*.png') # the tuple of file types
        for files in types:
            # glob("./**/*.jpg", recursive=True)
            split_info.extend(glob.glob(str(data_root/ '**' / files), recursive=True))
        dict_val = []
        cls_lists = []
        for cls in os.listdir(data_root):
            cls_list = [Path(i).relative_to(data_root) for i in split_info if Path(i).relative_to(data_root).parts[0]==cls]
            dict_val.extend([random.choices(split_population, split_weights)[0] for i in range(len(cls_list))])
            cls_lists.extend(cls_list)
        split_info = dict(zip(cls_lists, dict_val))
        data_list = make_crop_dataset_list(data_root, label_list, split_info)

    else:
        if split_info_path.exists():
            print(f'load {split_info_path}')
            with split_info_path.open('rb') as rf:
                split_info = pickle.load(rf)
        else:
            split_info = []
            types = ('*.jpg', '*.jpeg', '*.png') # the tuple of file types
            for files in types:
                # glob("./**/*.jpg", recursive=True)
                split_info.extend(glob.glob(str(data_root/ '**' / files), recursive=True))
            
            dict_val = []
            cls_lists = []
            for cls in os.listdir(data_root):
                cls_list = [Path(i).relative_to(data_root) for i in split_info if Path(i).relative_to(data_root).parts[0]==cls]
                dict_val.extend([random.choices(split_population, split_weights)[0] for i in range(len(cls_list))])
                cls_lists.extend(cls_list)
            split_info = dict(zip(cls_lists, dict_val))
            # split_info = defaultdict(lambda: random.choices(split_population, split_weights)[0], split_info)

        data_list = make_list(data_root, label_list, split_info)
        if file_write:
            print(f'make {split_info_path}')
            with split_info_path.open('wb') as wf:
                split_info = dict(split_info)
                pickle.dump(split_info, wf)
    return data_list

def preprocess_data(data_list, padding, padding_size, use_shift, use_bbox,
                    use_class, data_root, output_root, is_train):
    print(f'preprocess {output_root}')
    with ProcessPoolExecutor() as executor:
        future_list = []
        for data in tqdm(data_list):
            if data.class_id not in use_class:
                continue
            image_path = get_image_path(data_root, data, False, False)
            output_path = get_image_path(output_root, data, is_train, True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image_future = executor.submit(
                crop_image, image_path, data.bbox,
                padding, padding_size, use_shift, use_bbox,
                output_path
            )
            future_list.append(image_future)
        for future in tqdm(future_list):
            future.result()

def main():
    parser = argparse.ArgumentParser('preprocess pothole data')
    parser.add_argument('--name', default='test', type=str)
    parser.add_argument('--data_root', default='/data/pothole_data/raw', type=str)
    parser.add_argument('--output_root', default='/data/pothole_data/out', type=str)
    parser.add_argument('--label_list', default=['positive', 'negative'], nargs='+', type=str)
    parser.add_argument('--padding', default=Padding.BBOX, choices=list(Padding), type=Padding)
    parser.add_argument('--padding_size', default=0.0, type=float)
    parser.add_argument("--use_bbox", action='store_true')
    parser.add_argument("--use_shift", action='store_true')
    parser.add_argument('--use_class', default=[0, 1, 2, 3], nargs='+', type=int)
    parser.add_argument('--val_ratio', default=0.1, type=float)
    parser.add_argument('--test_ratio', default=0.1, type=float)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root) / args.name
    label_list = args.label_list
    padding = args.padding
    padding_size = args.padding_size
    use_bbox = args.use_bbox
    use_shift = args.use_shift
    use_class = args.use_class
    val_ratio = args.val_ratio
    test_ratio = args.test_ratio

    data_list = split_data(
        data_root, test_ratio, val_ratio, label_list)
    
    for data_name, data_sub_list in data_list.items():
        is_train = data_name != 'test'
        preprocess_data(
            data_sub_list,
            padding, padding_size, use_shift, use_bbox, use_class,
            data_root,
            output_root / data_name,
            is_train
        )

if __name__ == "__main__":
    main()
