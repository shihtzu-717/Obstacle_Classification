import os 
import glob

from packaging import version

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

def tabulate_events(dpath):
    dirs = sorted(os.listdir(dpath))
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in dirs]
    tags = summary_iterators[0].Tags()['scalars']
    for n, it in enumerate(summary_iterators):
        if it.Tags()['scalars'] != tags:
            summary_iterators.pop(n)
            dirs.pop(n)

    out = defaultdict(list)
    for tag in tags:
        for events in [acc.Scalars(tag) for acc in summary_iterators]:
            out[tag].append([e.value for e in events])
    return out, dirs


def to_csv(dpath):
    d, dirs = tabulate_events(dpath)
    tags, values = zip(*d.items())
    # col = [si for i, si in enumerate(dirs[0].split('_')) if i%2==0]
    # col.extend(['dir']) 

    col = []

    data = []
    for ii, dir in enumerate(dirs):
        # val = [si for i, si in enumerate(dir.split('_')) if i%2==1]
        # val.extend([dir])
        val = []
        for idx, tag in enumerate(tags):
            tag = tag.split('/')[-1]
            if idx in [5, 8]:
                n_val = values[idx]
                if ii == 0:
                    col.extend([f'{tag}_last_val', f'{tag}_ave_val', f'{tag}_max_val']) 
                val.extend([n_val[ii][-1], sum(n_val[ii])/len(n_val[ii]), max(n_val[ii])])
        data.append(val)
    df = pd.DataFrame(np.array(data), columns=col)
    df.to_csv(get_file_path(dpath, tag))
            

def get_file_path(dpath, tag):
    # file_name = tag.replace("/", "_") + '.csv'
    file_name = dpath.split('/')[-1] + '.csv'
    folder_path = os.path.join('csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


if __name__ == '__main__':
    path_list = glob.glob("./log/*")
    for i in tqdm(path_list):
        to_csv(i)
 

#  pad_FIX_padsize_50.0_box_True_shift_True_ratio_0.95
# pad_FIX_padsize_100.0_box_False_shift_False_ratio_0.7
# classifier/code/log/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_0.7_tratio_0.95_nbclss_2