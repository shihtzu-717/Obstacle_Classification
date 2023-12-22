import os

train_txt_file = '../results/230822_dataset_2/train+GAN+detection_except_augmentation+seoul_pos.txt'
valid_txt_file = '../results/230822_dataset_2/valid+GAN+seoul_pos.txt'


with open(train_txt_file, 'r') as f:
    train_list = [i.strip() for i in f.readlines()]

with open(valid_txt_file, 'r') as f:
    valid_list = [i.strip() for i in f.readlines()]

train_set = set()
for i in train_list:
    tmp = '_'.join((i.split('_')[:-1]))
    type = (i.split('_')[-1]).split('.')[-1]
    fn = tmp + '.' + type
    if not os.path.exists(fn):
        fn = fn.replace('.jpg', '.png')
        print(fn)
        train_set.add(fn)
    else:
        train_set.add(fn)

valid_set = set()
for i in valid_list:
    tmp = '_'.join((i.split('_')[:-1]))
    type = (i.split('_')[-1]).split('.')[-1]
    fn = tmp + '.' + type
    if not os.path.exists(fn):
        fn = fn.replace('.jpg', '.png')
        valid_set.add(fn)
    else:
        valid_set.add(fn)

org_train_txt_file = '../results/230822_dataset_2/org_train+GAN+detection_except_augmentation+seoul_pos.txt'
org_valid_txt_file = '../results/230822_dataset_2/org_valid+GAN+seoul_pos.txt'

with open(org_train_txt_file, 'w') as f:
    for i in train_set:
        f.write(i+'\n')

with open(org_valid_txt_file, 'w') as f:
    for i in valid_set:
        f.write(i+'\n')