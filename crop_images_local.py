import glob
import os
import cv2
import traceback
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

def xywh2xyxy(bbox, width, height):
    x, y, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    x1 = int(np.clip(x - w / 2, 0, 1) * width)
    y1 = int(np.clip(y - h / 2, 0, 1) * height)
    x2 = int(np.clip(x + w / 2, 0, 1) * width)
    y2 = int(np.clip(y + h / 2, 0, 1) * height)
    return x1, y1, x2, y2

def crop_image(image_path, bbox, YOLO_FORMAT, padding, padding_size, use_shift, use_bbox=False, output_path=None, imsave=False):
    try:
        image = cv2.imread(str(image_path))
        height, width, _ = image.shape
        if YOLO_FORMAT:
            x1, y1, x2, y2 = xywh2xyxy(bbox, width, height)
        else:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if use_bbox:
            image = cv2.rectangle(
                image,
                (x1 - 1, y1 - 1),
                (x2, y2),
                (0, 0, 255),
                1
            )
        if padding_size > 0:
            # if padding == Padding.BBOX or padding =='BBOX':
            if padding =='BBOX':
                pad_x = int(padding_size * (x2 - x1))
                pad_y = int(padding_size * (y2 - y1))
            # elif padding == Padding.PIXEL or padding =='PIXEL':
            elif padding =='PIXEL':
                pad_x = int(padding_size)
                pad_y = int(padding_size)
            # elif padding == Padding.IMAGE or padding =='IMAGE':
            elif padding =='IMAGE':
                pad_x = int(padding_size * width)
                pad_y = int(padding_size * height)
            # elif padding == Padding.FIX or padding =='FIX':
            elif padding =='FIX':
                pad_x = max(int((padding_size - (x2 - x1)) / 2), 0)
                pad_y = max(int((padding_size - (y2 - y1)) / 2), 0)
            # elif padding == Padding.FIX2 or padding =='FIX2':
            elif padding =='FIX2':
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
            image = cv2.resize(image, (224, 224))
            cv2.imwrite(str(output_path), image)

            return True
        else:
            return image
    except:
        print("Please Check the Image File")
        traceback.print_exc()
        return False

def main(path, output_path):
    # crop_img = preprocess_data.crop_image(
    #     image_path=image_path,
    #     bbox=bbox,
    #     YOLO_FORMAT=YOLO_FORMAT,
    #     padding=opsdict['pad'],
    #     padding_size=float(opsdict['padsize']),
    #     use_shift=str2bool(opsdict['shift']),
    #     use_bbox=str2bool(opsdict['box']),
    # )
    img_path = os.path.join(path, 'images')
    file_list = glob.glob(os.path.join(path, 'images', '*.jpg')) + glob.glob(os.path.join(path, 'images', '*.png'))
    for f in tqdm(file_list):
        img_fn = os.path.basename(f)
        if '.jpg' in img_fn:
            annot_fn = img_fn.replace('jpg', 'txt')
        elif '.png' in img_fn:
            annot_fn = img_fn.replace('png', 'txt')
        annot_path = os.path.join(path, 'annotations', annot_fn)
        with open(annot_path, 'r') as rf:
            readlines = [i.strip() for i in rf.readlines()]

        num=0
        org_img_fn = img_fn
        readlines = list(filter(None, readlines))
        for line in readlines:
            ele = (line).split()
            if ele[0] != "0":
                continue
            else:
                # img_fn = org_img_fn.split('.')[0]+"_"+str(num)+'.jpg'
                img_fn = os.path.splitext(org_img_fn)[0]+"_"+str(num)+'.jpg'
                crop_img = crop_image(
                    image_path=f,
                    bbox=ele[1:5],
                    YOLO_FORMAT=True,
                    padding='PIXEL', # padding=opsdict['pad'],
                    padding_size=100.0, # padding_size=float(opsdict['padsize']),
                    use_shift=True, # use_shift=str2bool(opsdict['shift']),
                    use_bbox=False, # use_bbox=str2bool(opsdict['box']),
                    output_path=os.path.join(output_path, img_fn),
                    imsave=True
                )
                num+=1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", "-p", type=str, required=True, help="user define base root directory")
    parser.add_argument("--output_path", "-o", type=str, help="user define output root directory")
    args = parser.parse_args()

    path = args.base_path
    output_path = os.path.join(path, 'crop_images')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    main(path, output_path)