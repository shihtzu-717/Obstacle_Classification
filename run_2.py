import os

base = """CUDA_VISIBLE_DEVICES=0 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --batch_size 128 --lr 1e-5 --update_freq 2 \
            --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --path_type true --txt_type false \
            --data_path '/home/daree/nasdata/tmp_obstacle' \
            --model_ema false --model_ema_eval false \
            --data_set image_folder \
            --use_cropimg=False \
            --auto_resume=False \
            --test_val_ratio 0.0 0.2 \
            --split_file_write=False \
            --use_cropimg False \
            --save_ckpt True \
            --lossfn BCE \
            --use_class 5"""

org_output_dir_name = "tmp"
log_dir = "tmp"
name1 = f'pad_PIXEL_padsize_100.0_box_False_shift_True_sratio_1.0_tratio_1.0_nbclss_9_tmp'
os.system(f"""{base} \
            --padding PIXEL \
            --padding_size 100.0 \
            --use_bbox False \
            --use_shift True \
            --output_dir results/{org_output_dir_name}/{name1} \
            --soft_label_ratio 1.0 \
            --label_ratio 1.0 \
            --nb_classes 9 \
            --log_dir {log_dir} \
            --log_name {name1} \
            --use_softlabel False \
            --soft_type 1""")
