import os
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', '-p', required=True, type=str)
args = parser.parse_args()

txt_path = args.path
valid_data_list = []
with open(os.path.join(txt_path, 'valid.txt'), 'r') as f:
    valid_data_list += [i.strip() for i in f.readlines()]

valid_data_set = set(valid_data_list)
total_data_path_set = set(glob("../nasdata/trainset/*/*/images/*.jpg") + glob("../nasdata/trainset/*/*/images/*.png"))

train_data_set = total_data_path_set - set(valid_data_list)

with open(os.path.join(txt_path, 'train.txt'), 'w') as f:
    for i in train_data_set:
        f.write(i+'\n')
