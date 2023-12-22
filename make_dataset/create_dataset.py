
### 지난 학습에서 사용한 train, valid set을 파일 이름으로 가져와서 현재 학습에 그대로 사용
### 현재 dataset에서 클래스가 바뀐 데이터를 적용해서 바꿈
import shutil
import os
from glob import glob

train_data_txt = '/home/daree/obstacle_code/results/231121/231121_train.txt'
valid_data_txt = '/home/daree/obstacle_code/results/231121/231121_valid.txt'

case_num = '04'
date_dir = '1221'

new_dataset_dir = f'/home/daree/nasdata/obstacle_data/trainset/2023{date_dir}/testcase_{case_num}'
new_train_data_txt = f'/home/daree/obstacle_code/results/23{date_dir}/dataset_txt/2023{date_dir}_case{case_num}_train.txt'
new_valid_data_txt = f'/home/daree/obstacle_code/results/23{date_dir}/dataset_txt/2023{date_dir}_case{case_num}_valid.txt'
new_total_data_txt = f'/home/daree/obstacle_code/results/23{date_dir}/dataset_txt/2023{date_dir}_case{case_num}_total.txt'

new_dataset_list = glob(f'{new_dataset_dir}/**/*.jpg', recursive=True) + glob(f'{new_dataset_dir}/**/*.png', recursive=True)
new_dataset_dict = {}
for i in new_dataset_list:
    fn = os.path.basename(i)
    new_dataset_dict[fn] = i


with open(train_data_txt, 'r') as f:
    train_data_list = [i.strip() for i in f.readlines()]

with open(valid_data_txt, 'r') as f:
    valid_data_list = [i.strip() for i in f.readlines()]

print(f'train_data_list: {len(train_data_list)}')
print(f'valid_data_list: {len(valid_data_list)}')
print(f'new_dataset_list: {len(new_dataset_list)}')

new_train_data_list = []
for i in train_data_list:
    fn = os.path.basename(i)
    if not new_dataset_dict.get(fn):
        continue
    new_train_data_list.append(new_dataset_dict[fn])

new_valid_data_list = []
for i in valid_data_list:
    fn = os.path.basename(i)
    if not new_dataset_dict.get(fn):
        continue
    new_valid_data_list.append(new_dataset_dict[fn])

with open(new_train_data_txt, 'w') as f:
    for i in new_train_data_list:
        f.write(i+'\n')

with open(new_valid_data_txt, 'w') as f:
    for i in new_valid_data_list:
        f.write(i+'\n')

with open(new_total_data_txt, 'w') as f:
    for i in new_train_data_list + new_valid_data_list:
        f.write(i+'\n')