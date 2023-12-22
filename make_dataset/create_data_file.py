import os
import random
from glob import glob


origin_train_list = []
with open('../results/230822_dataset_2/train_2.txt', 'r') as f:
    origin_train_list = [i.strip() for i in f.readlines()]

origin_valid_list = []
with open('../results/230822_dataset_2/test_2.txt', 'r') as f:
    origin_valid_list = [i.strip() for i in f.readlines()]

crop_data_list = glob('../../nasdata/crop_trainset_2/detection_data/**/*.jpg', recursive=True) + glob('../../nasdata/crop_trainset_2/detection_data/**/*.png', recursive=True)

origin_train_fn_list = [os.path.basename(i) for i in origin_train_list]
origin_valid_fn_list = [os.path.basename(i) for i in origin_valid_list]
crop_data_fn_list = [os.path.basename(i) for i in crop_data_list]
crop_data_fn_dict = {}

for data_path in crop_data_list:
    i = os.path.basename(data_path)
    img_type = os.path.splitext(i)[1]
    key_path = '_'.join(i.split('_')[:-1])+img_type
    if crop_data_fn_dict.get(key_path) == None:
        crop_data_fn_dict[key_path] = [data_path]
    elif len(crop_data_fn_dict.get(key_path)) > 0:
        tmp = crop_data_fn_dict.get(key_path)
        crop_data_fn_dict[key_path] = tmp + [data_path]
# print(crop_data_fn_dict)

crop_data_fn_keys_set = set(crop_data_fn_dict.keys())
crop_data_fn_keys_train_set = crop_data_fn_keys_set & set(origin_train_fn_list)
crop_data_fn_keys_valid_set = crop_data_fn_keys_set & set(origin_valid_fn_list)
print(len(crop_data_fn_keys_train_set), len(set(origin_train_fn_list)))
print(len(crop_data_fn_keys_valid_set), len(set(origin_valid_fn_list)))

train_set = []
for i in crop_data_fn_keys_train_set:
    train_set += crop_data_fn_dict.get(i)

valid_set = []
for i in crop_data_fn_keys_valid_set:
    valid_set += crop_data_fn_dict.get(i)

# print(len(train_set))
# print(len(valid_set))

with open('../results/230822_dataset_2/tmp_train.txt', 'w') as f:
    for i in train_set:
        f.write(i+'\n')
        
with open('../results/230822_dataset_2/tmp_valid.txt', 'w') as f:
    for i in valid_set:
        f.write(i+'\n')
