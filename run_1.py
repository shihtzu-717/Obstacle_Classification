import os

## model setting ##
# padding = ['FIX', 'PIXEL', 'FIX2']
# padding_size = [0, 50 ,100, 150, 200, 224, 256, 336, 384, 448]
# use_bbox = ['False', 'True']
# use_shift = ['False', 'True']
# soft_label_ratio = [0.9, 0.8, 0.7, 0.6]
# target_label_ratio = [1, 0.98, 0.96, 0.94, 0.92, 0.90 ]
# warmup = [5]

# padding = ['PIXEL']
# padding_size = [100]
# use_bbox = ['False']
# use_shift = ['True']
# soft_label_ratio = [0.7, 0.8, 0.9]
# target_label_ratio = [0.92, 0.95, 1]
# nb_classes = [2]
# soft_type = [1, 2]

padding = ['PIXEL']
padding_size = [100]
use_bbox = ['False']
use_shift = ['True']
# soft_label_ratio = [0.7]
# target_label_ratio = [0.95]
soft_label_ratio = [1.0]
target_label_ratio = [1.0]
nb_classes = [2]
soft_type = [1, 2]


# base = """CUDA_VISIBLE_DEVICES=2 python main.py \
#             --model convnext_base --drop_path 0.2 --input_size 224 \
#             --batch_size 256 --lr 5e-5 --update_freq 2 \
#             --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
#             --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
#             --finetune checkpoint/convnext_base_22k_224.pth \
#             --path_type true --txt_type false \
#             --data_path '/home/daree/nasdata/trainset/01st_data' \
#                 '/home/daree/nasdata/trainset/02nd_data' \
#                 '/home/daree/nasdata/trainset/03rd_data' \
#                 '/home/daree/nasdata/trainset/04th_data' \
#                 '/home/daree/nasdata/trainset/05th_data' \
#                 '/home/daree/nasdata/trainset/06th_data' \
#                 '/home/daree/nasdata/trainset/07th_data' \
#                 '/home/daree/nasdata/trainset/08th_data' \
#                 '/home/daree/nasdata/trainset/09th_data' \
#                 '/home/daree/nasdata/trainset/10th_data' \
#                 '/home/daree/nasdata/trainset/11th_data' \
#                 '/home/daree/nasdata/trainset/12th_data' \
#             --model_ema false --model_ema_eval false \
#             --data_set image_folder \
#             --use_cropimg=False \
#             --auto_resume=False \
#             --test_val_ratio 0.0 0.2 \
#             --split_file_write=False \
#             --use_cropimg False \
#             --save_ckpt True \
#             --lossfn BCE \
#             --use_class 0"""

base = """CUDA_VISIBLE_DEVICES=2 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --batch_size 256 --lr 5e-5 --update_freq 2 \
            --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --path_type false --txt_type true \
            --train_txt_path results/230710_set1-12/train.txt \
            --valid_txt_path results/230710_set1-12/valid.txt \
            --model_ema false --model_ema_eval false \
            --data_set image_folder \
            --use_cropimg=False \
            --auto_resume=False \
            --test_val_ratio 0.0 0.2 \
            --split_file_write=False \
            --use_cropimg False \
            --save_ckpt True \
            --lossfn BCE \
            --use_class 0"""
# org_output_dir_name = "after_set1-3"
org_output_dir_name = "230710_set1-12"

# org_output_dir_name = "230706_set1-12_class-balance"
log_dir = "log_230710_set1-12"
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
 
for pad in padding:
    for pad_size in padding_size:
        for bbox in use_bbox:
            for shift in use_shift:
                for target_ratio in target_label_ratio:
                    for soft_ratio in soft_label_ratio:
                        for ncls in nb_classes:
                            for st in soft_type: 
                                use_softlabel = True if ncls == 2 else False
                                name = f'pad_{pad}_padsize_{pad_size:.1f}_box_{bbox}_shift_{shift}_sratio_{soft_ratio}_tratio_{target_ratio}_nbclss_{ncls}'
                                output_dir_name = org_output_dir_name
                                if ncls == 4:
                                    name += f'_soft-type_{st}'
                                    output_dir_name += f"/4-class-soft_type-{st}"
                                if ncls == 2:
                                    output_dir_name += f"/2-class"
                                    if st == 2:
                                        continue
                                if not os.path.isdir(os.getcwd() + f'/{log_dir}/' + name):
                                    os.system(f"""{base} \
                                            --padding {pad}\
                                            --padding_size {pad_size}\
                                            --use_bbox {bbox}\
                                            --use_shift {shift}\
                                            --output_dir results/{output_dir_name}/{name} \
                                            --soft_label_ratio {soft_ratio} \
                                            --label_ratio {target_ratio} \
                                            --nb_classes {ncls} \
                                            --log_dir {log_dir} \
                                            --log_name {name} \
                                            --use_softlabel {use_softlabel} \
                                            --soft_type {st}""")
                                else:
                                    print(f"Check the {os.getcwd() + f'/{log_dir}/' + name} ")

