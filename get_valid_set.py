import os
import shutil
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', required=True, type=str)
parser.add_argument('--output_path', '-o', required=True, type=str)
args = parser.parse_args()

txt_path = args.path
output_path = args.output_path
data_list = []
with open(txt_path, 'r') as f:
    data_list += [i.strip() for i in f.readlines()]

if not os.path.exists(output_path):
    os.makedirs(os.path.join(output_path, 'images'))
    os.makedirs(os.path.join(output_path, 'annotations'))

for img_path in tqdm(data_list):
    annot_path = img_path.replace('images', 'annotations')
    annot_path = annot_path.replace('.jpg', '.txt')
    annot_path = annot_path.replace('.png', '.txt')
    shutil.copy(img_path, os.path.join(output_path, 'images'))
    shutil.copy(annot_path, os.path.join(output_path, 'annotations'))