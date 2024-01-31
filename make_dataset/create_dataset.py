
### 지난 학습에서 사용한 train, valid set을 파일 이름으로 가져와서 현재 학습에 그대로 사용
### 현재 dataset에서 클래스가 바뀐 데이터를 적용해서 바꿈
import shutil
import os
import random
from glob import glob

train_data_txt = '/home/daree/obstacle_code/results/240122/dataset_txt/20240122_case01_train.txt'
valid_data_txt = '/home/daree/obstacle_code/results/240122/dataset_txt/20240122_case01_valid.txt'

case_num = '07'
date_dir = '0125'

new_dataset_dir = f'/home/daree/nasdata/obstacle_data/trainset/2024{date_dir}/testcase_{case_num}'
new_train_data_txt = f'/home/daree/obstacle_code/results/24{date_dir}/dataset_txt/2024{date_dir}_case{case_num}_train.txt'
new_valid_data_txt = f'/home/daree/obstacle_code/results/24{date_dir}/dataset_txt/2024{date_dir}_case{case_num}_valid.txt'
new_total_data_txt = f'/home/daree/obstacle_code/results/24{date_dir}/dataset_txt/2024{date_dir}_case{case_num}_total.txt'

new_dataset_list = glob(f'{new_dataset_dir}/**/*.jpg', recursive=True) + glob(f'{new_dataset_dir}/**/*.png', recursive=True)

new_dataset_list_fn = []
for i in new_dataset_list:
    fn = os.path.basename(i)
    new_dataset_list_fn.append(fn)

new_dataset_dict = {}
for i in new_dataset_list:
    fn = os.path.basename(i)
    new_dataset_dict[fn] = i

with open(train_data_txt, 'r') as f:
    train_data_list = [i.strip() for i in f.readlines()]

with open(valid_data_txt, 'r') as f:
    valid_data_list = [i.strip() for i in f.readlines()]

new_train_data_list = []
new_train_data_list_fn = []
for i in train_data_list:
    fn = os.path.basename(i)
    if not new_dataset_dict.get(fn):
        continue
    new_train_data_list.append(new_dataset_dict[fn])
    new_train_data_list_fn.append(fn)

new_valid_data_list = []
new_valid_data_list_fn = []
for i in valid_data_list:
    fn = os.path.basename(i)
    if not new_dataset_dict.get(fn):
        continue
    new_valid_data_list.append(new_dataset_dict[fn])
    new_valid_data_list_fn.append(fn)


add_data = list(set(new_dataset_list_fn) - set(new_train_data_list_fn + new_valid_data_list_fn))
add_train_data_list = random.sample(add_data, round(len(add_data)*0.8))
add_valid_data_list = list(set(add_data) - set(train_data_list))

for fn in add_train_data_list:
    new_train_data_list.append(new_dataset_dict[fn])
for fn in add_valid_data_list:
    new_valid_data_list.append(new_dataset_dict[fn])

print(f'    dataset_len: {len(train_data_list)+ len(valid_data_list)} = train: {len(train_data_list)}, valid: {len(valid_data_list)}')
print(f'new_dataset_len: {len(new_dataset_list)} = train: {len(new_train_data_list)}, valid: {len(new_valid_data_list)}')

with open(new_train_data_txt, 'w') as f:
    for i in new_train_data_list:
        if not os.path.exists(i):
            print(f'Not Exist Data: {i}')
        else:
            f.write(i+'\n')

with open(new_valid_data_txt, 'w') as f:
    for i in new_valid_data_list:
        f.write(i+'\n')

with open(new_total_data_txt, 'w') as f:
    for i in new_train_data_list + new_valid_data_list:
        f.write(i+'\n')

