import os
import shutil
from glob import glob
from tqdm import tqdm

use_classes = ["1", "2"]

# base_path = '../../nasdata/base_trainset/RiaaS/annotations/**/*.txt'
# output_path = '../../nasdata/base_pothole/RiaaS/'

base_path = '../../nasdata/base_trainset/KoreaEX/annotations/**/*.txt'
output_path = '../../nasdata/base_pothole/KoreaEX/'

if not os.path.exists(output_path):
    os.makedirs(os.path.join(output_path, 'images'))
    os.makedirs(os.path.join(output_path, 'annotations'))


base_annot_list = glob(base_path, recursive=True)

pothole_data_list = []
for annot_path in tqdm(base_annot_list, desc="Check the pothole data"):
    lines = []
    with open(annot_path, 'r') as rf:
        lines = [i.strip() for i in rf.readlines()]
    
    for line in lines:
        ele = line.split()
        if ele[0] in use_classes:
            pothole_data_list.append(annot_path)
            break
        else:
            break

for pothole_annot in tqdm(pothole_data_list, desc='Copy pothole data'):
    fn_annot = os.path.basename(pothole_annot)
    images_path = pothole_annot.replace('annotations', 'images')
    if os.path.exists(images_path.replace('.txt', '.jpg')):
        images_path = images_path.replace('.txt', '.jpg')
    else:
        images_path.replace('.txt', '.png')
    shutil.copy(pothole_annot, os.path.join(output_path, 'annotations'))
    shutil.copy(images_path, os.path.join(output_path, 'images'))

    lines = []
    with open(os.path.join(output_path, 'annotations', fn_annot), 'r') as rf:
        lines = [i.strip() for i in rf.readlines()]
    new_lines = []
    for line in lines:
        ele = line.split()
        if ele[0] in use_classes:
            ele[0] = "0"
            new_lines.append(' '.join(ele) + '\n')
        
    with open(os.path.join(output_path, 'annotations', fn_annot), 'w') as wf:
        wf.writelines(new_lines)

