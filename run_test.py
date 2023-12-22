import os

org_output_dir_name = "231130"
log_dir = "log_231130"

base1 = """CUDA_VISIBLE_DEVICES=0 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --batch_size 128 --lr 5e-5 --update_freq 2 \
            --epochs 200 --warmup_epochs 20 --weight_decay 1e-8 \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
            --path_type false --txt_type true \
            --train_txt_path results/231130/dataset_txt/20231130_case01_train.txt \
            --valid_txt_path results/231130/dataset_txt/20231130_case01_valid.txt \
            --model_ema false --model_ema_eval false \
            --data_set image_folder \
            --auto_resume=False \
            --test_val_ratio 0.0 0.2 \
            --split_file_write=False \
            --save_ckpt True \
            --lossfn BCE \
            --use_class 5 \
            --use_cropimg True"""

name1 = f'tmp'
cls_name_txt = 'results/231130/dataset_txt/class_name_case01.txt'
os.system(f"""{base1} \
            --padding PIXEL \
            --padding_size 100.0 \
            --use_bbox False \
            --use_shift True \
            --output_dir results/{org_output_dir_name}/{name1} \
            --nb_classes 12 \
            --log_dir {log_dir} \
            --log_name {name1} \
            --cls_name_txt {cls_name_txt}""")