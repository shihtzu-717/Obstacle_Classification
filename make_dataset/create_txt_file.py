from glob import glob
import os


data_dir_path = '../../nasdata/crop_trainset_2/manhole_set02_neg/negative/images'
data_list = glob(os.path.join(data_dir_path, '*.jpg')) + glob(os.path.join(data_dir_path, '*.png'))
data_fn_list = [os.path.basename(i) for i in data_list]

with open('../results/231005_manhole/manhole_set02_neg.txt', 'w') as f:
    for i in data_fn_list:
        f.write(os.path.join('/home/daree/nasdata/crop_trainset_2/manhole_set02_neg/negative/images', i)+'\n')

# total_list = []
# with open('../results/231005_manhole/manhole_set01.txt', 'r') as f:
#     total_list = [i.strip() for i in f.readlines()]
# total_fn_list = [os.path.basename(i) for i in total_list]

# total_fn_set = set(total_fn_list)
# data_fn_set = set(data_fn_list)

# for i in data_fn_set - total_fn_set:
#     if os.path.exists(os.path.join(data_dir_path, i)):
#         print(os.path.join('/home/daree/nasdata/crop_trainset_2/manhole_set01/negative/images', i))
#     else:
#         print("NOT EXISIT", print(os.path.join(data_dir_path, i)))