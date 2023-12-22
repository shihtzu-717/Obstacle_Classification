from glob import glob
import os

train_list = []
with open('train.txt', 'r') as f:
    train_list = [i.strip() for i in f.readlines()]

val_list = []
with open('valid.txt', 'r') as f:
    val_list = [i.strip() for i in f.readlines()]


critical_list = (glob('data/set*/positive/critical/images/*.jpg') + glob('data/set*/positive/critical/images/*.png'))
normal_list = (glob('data/set*/positive/normal/images/*.jpg') + glob('data/set*/positive/normal/images/*.png'))

train_fn_set = set([os.path.basename(i) for i in train_list])
val_fn_set = set([os.path.basename(i) for i in val_list])
critical_fn_set = set([os.path.basename(i) for i in critical_list])
normal_fn_set = set([os.path.basename(i) for i in normal_list])

trainset_critical_data = list(train_fn_set & critical_fn_set)
valset_critical_data = list(val_fn_set & critical_fn_set)

trainset_normal_data = list(train_fn_set & normal_fn_set)
valset_normal_data = list(val_fn_set & normal_fn_set)

with open('trainset_critical_pos.txt', 'w') as f:
    for i in trainset_critical_data:
        f.write('/home/daree/nasdata/critical_data/14th_critical/positive/images/'+i+'\n')

with open('valset_critical_pos.txt', 'w') as f:
    for i in valset_critical_data:
        f.write('/home/daree/nasdata/critical_data/14th_critical/positive/images/'+i+'\n')

with open('trainset_normal_pos.txt', 'w') as f:
    for i in trainset_normal_data:
        f.write('/home/daree/nasdata/critical_data/13th_normal/positive/images/'+i+'\n')

with open('valset_normal_pos.txt', 'w') as f:
    for i in valset_normal_data:
        f.write('/home/daree/nasdata/critical_data/13th_normal/positive/images/'+i+'\n')