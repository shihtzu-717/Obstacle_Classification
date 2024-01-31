import os
import random
from glob import glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data_path', '-p', required=True, type=str)
parser.add_argument('--output_path', '-o', required=True, type=str)
parser.add_argument('--mode', '-m', default='w', type=str)
parser.add_argument('--file_name', '-fn', required=True, type=str)
args = parser.parse_args()

data_path = args.data_path
output_path = args.output_path
m = args.mode

data_list = glob(os.path.join(data_path, '**', 'images', '*'), recursive=True)
for i in data_list:
    if 'desktop.ini' in i:
        print(i)
        data_list.remove(i)

train_data_list = random.sample(data_list, round(len(data_list)*0.8))
valid_data_list = list(set(data_list) - set(train_data_list))

with open(os.path.join(output_path, f'{args.file_name}_train.txt'), m) as f:
    for i in train_data_list:
        f.write(i+'\n')

with open(os.path.join(output_path, f'{args.file_name}_valid.txt'), m) as f:
    for i in valid_data_list:
        f.write(i+'\n')

with open(os.path.join(output_path, f'{args.file_name}_total.txt'), m) as f:
    for i in train_data_list + valid_data_list:
        f.write(i+'\n')
