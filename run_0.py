import os

## model setting ##
# padding = ['FIX', 'PIXEL', 'FIX2']
# padding_size = [0, 50 ,100, 150, 200, 224, 256, 336, 384, 448]
# use_bbox = ['False', 'True']
# use_shift = ['False', 'True']
# soft_label_ratio = [0.9, 0.8, 0.7, 0.6]
# target_label_ratio = [1, 0.98, 0.96, 0.94, 0.92, 0.90 ]
# warmup = [0]

# padding = ['FIX2']
# padding_size = [384]
# use_bbox = ['False']
# use_shift = ['True']
# target_label_ratio = [0.92, 0.95, 1]
# soft_label_ratio = [0.7, 0.8, 0.9]
# nb_classes = [2, 4]
# soft_type = [1, 2]
# org_output_dir_name = "/230601-train_set1-2-3_test_set4_epoch_150"


# base = """CUDA_VISIBLE_DEVICES=0 python main.py \
#             --model convnext_base --drop_path 0.2 --input_size 224 \
#             --batch_size 128 --lr 5e-5 --update_freq 2 \
#             --epochs 50 --weight_decay 1e-8 \
#             --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
#             --finetune checkpoint/convnext_base_22k_224.pth \
#             --data_path '/home/daree/nasdata/ambclass_update/1st_data' '/home/daree/nasdata/ambclass_update/2nd_data' '/home/daree/nasdata/ambclass_update/3rd_data'\
#             --eval_data_path data \
#             --model_ema true --model_ema_eval true \
#             --data_set image_folder \
#             --warmup_epochs 15 \
#             --use_cropimg=False \
#             --auto_resume=False \
#             --test_val_ratio 0.0 0.2 \
#             --split_file_write=False \
#             --use_cropimg False \
#             --save_ckpt True \
#             --lossfn BCE \
#             --use_class 0"""


padding = ['FIX2']
padding_size = [384]
use_bbox = ['False']
use_shift = ['True']
target_label_ratio = [0.92]
soft_label_ratio = [0.7]
nb_classes = [4]
soft_type = [1]

org_output_dir_name = "test"


base = """CUDA_VISIBLE_DEVICES=0 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --batch_size 256 --lr 5e-5 --update_freq 2 \
            --epochs 50 --weight_decay 1e-8 \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --data_path '/home/daree/nasdata/ambclass_update/1st_data'\
            --model_ema true --model_ema_eval true \
            --data_set image_folder \
            --warmup_epochs 15 \
            --use_cropimg=False \
            --auto_resume=False \
            --test_val_ratio 0.0 0.2 \
            --split_file_write=False \
            --use_cropimg False \
            --save_ckpt True \
            --lossfn BCE \
            --use_class 0"""


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
                                if not os.path.isdir(os.getcwd() + '/log/' + name):
                                    os.system(f"""{base} \
                                            --padding {pad}\
                                            --padding_size {pad_size}\
                                            --use_bbox {bbox}\
                                            --use_shift {shift}\
                                            --output_dir results/{output_dir_name}/{name} \
                                            --soft_label_ratio {soft_ratio} \
                                            --label_ratio {target_ratio} \
                                            --nb_classes {ncls} \
                                            --log_name {name} \
                                            --use_softlabel={use_softlabel} \
                                            --soft_type {st}""")
