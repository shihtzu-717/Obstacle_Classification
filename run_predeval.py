import os
import sys
import glob
from pathlib import Path

# base = """CUDA_VISIBLE_DEVICES=2 python main.py \
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

base = """python main.py \
            --model convnext_base \
            --lr 1e-6 \
            --use_cropimg=False \
            --auto_resume=False \
            --drop_path 0.2 \
            --layer_decay 0.8 \
            --test_val_ratio 0.0 1.0 \
            --use_softlabel True \
            --use_class 5 \
            --pred_eval True \
            --pred True \
            --path_type true --txt_type false \
            --eval_data_path ../nasdata/tmp_obstacle \
            --pred_save True \
            --pred_save_with_conf False \
            --use_cropimg False \
            --conf 0.0 \
            --several_models False"""


models = [
            'results/tmp/pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_9_tmp/checkpoint-best.pth',
        ]

for m in models:
    if not os.path.exists(m):
        print(f"Check the models name {m}")
        sys.exit()

# org_output_dir_path = "../res/230822_dataset_2_data/dataset_20230719"
# org_graph_save_dir = "../res/230822_dataset_2_graph/dataset_20230719"

output_dir_path = "../res/tmp"
graph_save_dir = "../res/tmp"


for ckpt in models:
    nb_cls = 9
    name = ckpt.split('/')[-2]+ '_'
    ops = Path(ckpt).parts[-2]
    opsdict = dict(zip([str(d) for i,d in enumerate(str(ops).split('_')) if i%2==0], 
                       [str(d) for i, d in enumerate(ops.split('_')) if i%2==1]))

    # if "checkpoint-valid_min_loss.pth" in ckpt:
    #     output_dir_path = os.path.join(output_dir_path, 'loss_model')
    #     graph_save_dir = os.path.join(graph_save_dir, 'loss_model')
    # elif "checkpoint-best.pth" in ckpt:
    #     output_dir_path = os.path.join(output_dir_path, 'best_model')
    #     graph_save_dir = os.path.join(graph_save_dir, 'best_model')

    if not os.path.exists(output_dir_path):
        os.makedirs(Path(output_dir_path), exist_ok=True)
    
    if not os.path.exists(graph_save_dir):
        os.makedirs(Path(graph_save_dir), exist_ok=True)

    print(output_dir_path, graph_save_dir)

    os.system(f"""{base} \
            --resume {ckpt} \
            --padding {opsdict['pad']} \
            --padding_size {opsdict['padsize']} \
            --use_bbox {opsdict['box']} \
            --pred_eval_name {graph_save_dir}/{name} \
            --pred_save_path {output_dir_path}/{name} \
            --nb_classes {nb_cls} \
            --use_shift {opsdict['shift']}""")

