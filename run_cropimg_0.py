import os

org_output_dir_name = "240125"
log_dir = "log_240125"

base1 = """CUDA_VISIBLE_DEVICES=0 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --batch_size 128 --lr 5e-5 --update_freq 2 \
            --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0.0 --mixup 0.0 \
            --path_type false --txt_type true \
            --train_txt_path results/240125/dataset_txt/20240125_case01_train.txt \
            --valid_txt_path results/240125/dataset_txt/20240125_case01_valid.txt \
            --model_ema false --model_ema_eval false \
            --data_set image_folder \
            --auto_resume=False \
            --test_val_ratio 0.0 0.2 \
            --split_file_write=False \
            --save_ckpt True \
            --lossfn BCE \
            --use_class 5 \
            --use_cropimg True"""

name1 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_13_case01'
cls_name_txt1 = 'results/240125/dataset_txt/class_name_case01.txt'
os.system(f"""{base1} \
            --padding PIXEL \
            --padding_size 100.0 \
            --use_bbox False \
            --use_shift True \
            --output_dir results/{org_output_dir_name}/{name1} \
            --nb_classes 13 \
            --log_dir {log_dir} \
            --log_name {name1} \
            --cls_name_txt {cls_name_txt1}""")



##########################################################################################################
base2 = """CUDA_VISIBLE_DEVICES=0 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --batch_size 128 --lr 5e-5 --update_freq 2 \
            --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0.0 --mixup 0.0 \
            --path_type false --txt_type true \
            --train_txt_path results/240125/dataset_txt/20240125_case02_train.txt \
            --valid_txt_path results/240125/dataset_txt/20240125_case02_valid.txt \
            --model_ema false --model_ema_eval false \
            --data_set image_folder \
            --auto_resume=False \
            --test_val_ratio 0.0 0.2 \
            --split_file_write=False \
            --save_ckpt True \
            --lossfn BCE \
            --use_class 5 \
            --use_cropimg True"""

name2 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_14_case02'
cls_name_txt2 = 'results/240125/dataset_txt/class_name_case02.txt'
os.system(f"""{base2} \
            --padding PIXEL \
            --padding_size 100.0 \
            --use_bbox False \
            --use_shift True \
            --output_dir results/{org_output_dir_name}/{name2} \
            --nb_classes 14 \
            --log_dir {log_dir} \
            --log_name {name2} \
            --cls_name_txt {cls_name_txt2}""")



##########################################################################################################
base3 = """CUDA_VISIBLE_DEVICES=0 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --batch_size 128 --lr 5e-5 --update_freq 2 \
            --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0.0 --mixup 0.0 \
            --path_type false --txt_type true \
            --train_txt_path results/240125/dataset_txt/20240125_case03_train.txt \
            --valid_txt_path results/240125/dataset_txt/20240125_case03_valid.txt \
            --model_ema false --model_ema_eval false \
            --data_set image_folder \
            --auto_resume=False \
            --test_val_ratio 0.0 0.2 \
            --split_file_write=False \
            --save_ckpt True \
            --lossfn BCE \
            --use_class 5 \
            --use_cropimg True"""

name3 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_13_case03'
cls_name_txt3 = 'results/240125/dataset_txt/class_name_case03.txt'
os.system(f"""{base3} \
            --padding PIXEL \
            --padding_size 100.0 \
            --use_bbox False \
            --use_shift True \
            --output_dir results/{org_output_dir_name}/{name3} \
            --nb_classes 13 \
            --log_dir {log_dir} \
            --log_name {name3} \
            --cls_name_txt {cls_name_txt3}""")



##########################################################################################################
base4 = """CUDA_VISIBLE_DEVICES=0 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --batch_size 128 --lr 5e-5 --update_freq 2 \
            --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0.0 --mixup 0.0 \
            --path_type false --txt_type true \
            --train_txt_path results/240125/dataset_txt/20240125_case04_train.txt \
            --valid_txt_path results/240125/dataset_txt/20240125_case04_valid.txt \
            --model_ema false --model_ema_eval false \
            --data_set image_folder \
            --auto_resume=False \
            --test_val_ratio 0.0 0.2 \
            --split_file_write=False \
            --save_ckpt True \
            --lossfn BCE \
            --use_class 5 \
            --use_cropimg True"""

name4 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_nbclss_14_case04'
cls_name_txt4 = 'results/240125/dataset_txt/class_name_case04.txt'
os.system(f"""{base4} \
            --padding PIXEL \
            --padding_size 100.0 \
            --use_bbox False \
            --use_shift True \
            --output_dir results/{org_output_dir_name}/{name4} \
            --nb_classes 14 \
            --log_dir {log_dir} \
            --log_name {name4} \
            --cls_name_txt {cls_name_txt4}""")


