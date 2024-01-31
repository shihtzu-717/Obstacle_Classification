import os
import cv2
import pickle
import random
import traceback
import argparse

import numpy as np

from enum import Enum
from tqdm import tqdm
from pathlib import Path
from glob import glob
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor

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

def crop_image(image_path, bbox, padding, padding_size, use_shift, use_bbox):
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
        return image
    except:
        traceback.print_exc()
        return False
    

def find_annotation(img):
    annot_path = img.replace('images', 'annotations')
    annot = Path(annot_path)
    annot = annot.with_suffix('.txt')
    if annot.exists():
        return annot
    raise Exception(f'image not found: {img}')

def read_annotation(annotation, use_class):
    res = []
    with annotation.open('r', encoding='utf-8') as rf:
        for line in rf.readlines():
            line = line.strip()
            data = line.split()
            if len(data) == 0:
                continue
            class_id = int(data[0])
            if not class_id in use_class:
                continue
            bbox = [float(v) for v in data[1:5]]
            res.append([class_id, bbox])
    return res

def main():
    parser = argparse.ArgumentParser('preprocess pothole data')
    parser.add_argument('--data_root', '-p', default='/data/pothole_data/raw', type=str)
    parser.add_argument('--output_root', '-o', default='/data/pothole_data/out', type=str)
    parser.add_argument('--label_list', default=['positive', 'negative'], nargs='+', type=str)
    parser.add_argument('--padding', default=Padding.PIXEL, choices=list(Padding), type=Padding)
    parser.add_argument('--padding_size', default=100.0, type=float)
    parser.add_argument("--use_bbox", default=False, action='store_true')
    parser.add_argument("--use_shift", default=True, action='store_true')
    parser.add_argument('--use_class', default=[0], nargs='+', type=int)
    args = parser.parse_args()

    print(args)

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    padding = args.padding
    padding_size = args.padding_size
    use_bbox = args.use_bbox
    use_shift = args.use_shift
    use_class = args.use_class

    images_list = glob(str(data_root / '**' / 'images' / '*.jpg'), recursive=True) + glob(str(data_root / '**' / 'images' / '*.png'), recursive=True)
    print(len(images_list))
    for idx, img in tqdm(enumerate(images_list)):
        annot_path = find_annotation(img)
        bbox = read_annotation(annot_path, use_class)
        if len(bbox) == 0:
            continue
        for b in bbox:
            output_path = output_root / f'{Path(img).stem}_{idx}{Path(img).suffix}'
            crop_img = crop_image(img, b[1], padding, padding_size, use_shift, use_bbox)
            crop_img = cv2.resize(crop_img, (224, 224))
            cv2.imwrite(str(output_path), crop_img)



if __name__ == "__main__":
    main()
