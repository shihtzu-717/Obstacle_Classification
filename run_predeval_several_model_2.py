import os
import sys
import glob
from pathlib import Path

base = """python main.py \
            --model convnext_base \
            --lr 1e-6 \
            --use_cropimg=False \
            --auto_resume=False \
            --drop_path 0.2 \
            --layer_decay 0.8 \
            --test_val_ratio 0.0 1.0 \
            --use_class 5 \
            --pred_eval True \
            --pred True \
            --path_type true --txt_type false \
            --eval_data_path ../nasdata/obstacle_data/testset/snow \
            --pred_save True \
            --pred_save_with_conf False \
            --use_cropimg True \
            --conf 0.0 \
            --several_models True \
            --sample_images False \
            --use_type False"""

# pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_13_case01
# pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_13_case03
# pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_14_case02
# pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_14_case04
# pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_14_case05
# pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_14_case06

models = [  
    'results/240122/pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_15_case05/checkpoint-best.pth'
]

class_names = [
    'results/240122/dataset_txt/class_name_case05.txt',
]

output_dir_path = "../res/testset/snow"
graph_save_dir = "../res/testset/snow"

for m in models:
    if not os.path.exists(m):
        print(f"Check the models name {m}")
        sys.exit()



ckpt = ' '.join(models)
cln = ' '.join(class_names)

os.system(f"""{base} \
            --resume {ckpt} \
            --pred_eval_name {graph_save_dir} \
            --pred_save_path {output_dir_path} \
            --cls_name_txt {cln}""")
