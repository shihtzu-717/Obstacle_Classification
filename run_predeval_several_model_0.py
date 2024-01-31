import os
import sys
import glob
from pathlib import Path

# base = """CUDA_VISIBLE_DEVICES=1 python main.py \
#             --lr 1e-6 \
#             --use_cropimg=False \
#             --auto_resume=False \
#             --drop_path 0.2 \
#             --layer_decay 0.8 \
#             --test_val_ratio 0.0 0.2 \
#             --nb_classes 2 \
#             --use_softlabel True \
#             --use_class 0 \
#             --pred True \
#             --pred_eval True \
#             --eval_data_path '../nasdata/trainset/01st_data' \
#             --eval_not_include_neg True"""

base = """CUDA_VISIBLE_DEVICES=0 python main.py \
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
            --eval_data_path ../nasdata/obstacle_data/test/20231012_set_1_crop_images \
            --pred_save True \
            --pred_save_with_conf True \
            --use_cropimg True \
            --conf 0.0 \
            --several_models True \
            --sample_images False"""


models = [  
            'results/231130/pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_9_case12/checkpoint-best.pth',
            'results/231130/pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_10_case05/checkpoint-best.pth',
            'results/231130/pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_10_case09/checkpoint-best.pth',
            'results/231130/pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_10_case10/checkpoint-best.pth',
            'results/231130/pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_10_case11/checkpoint-best.pth',
            'results/231130/pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_11_case02/checkpoint-best.pth',

        ]

class_names = [
    'results/231130/dataset_txt/class_name_case12.txt',
    'results/231130/dataset_txt/class_name_case05.txt',
    'results/231130/dataset_txt/class_name_case09.txt',
    'results/231130/dataset_txt/class_name_case10.txt',
    'results/231130/dataset_txt/class_name_case11.txt',
    'results/231130/dataset_txt/class_name_case02.txt',
]
    

for m in models:
    if not os.path.exists(m):
        print(f"Check the models name {m}")
        sys.exit()

output_dir_path = "../res/231130_data/20231012_set_1_crop_images/"
graph_save_dir = "../res/231130_graph/20231012_set_1_crop_images/"

ckpt = ' '.join(models)
cln = ' '.join(class_names)

os.system(f"""{base} \
            --resume {ckpt} \
            --pred_eval_name {graph_save_dir} \
            --pred_save_path {output_dir_path} \
            --cls_name_txt {cln}""")