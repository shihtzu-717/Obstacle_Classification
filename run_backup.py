import os

base = "python -m torch.distributed.launch --nproc_per_node=2 main.py --lr 1e-6 --use_cropimg=False"
epochs = 50
set1_model = "/home/unmanned/results/set_1/b/pad_p_100_bbox_cls012_yp10n1/checkpoint-best.pth"       
set1_2_model = "/home/unmanned/results/set_2/b/pad_f2_336_shift_cls012_yp24n1tp1/checkpoint-best.pth"

## model setting ##
models = [set1_model, set1_2_model]
inputsize = [224, 224]
padding = ['PIXEL', 'FIX2']
padding_size = [100, 336]
use_bbox = ['True', 'False']
use_shift = ['False', 'True']
## model setting ##


total_model= "/home/daree/code/results/im224_b128_wu0/checkpoint-best.pth"

set1_data = "/home/daree/data/pothole_data/set_1/base_cls012/test/yolo"
set1_raw_data = "/home/daree/nas/dataset1/images"

set2_data = "/home/daree/data/pothole_data/set_2/train/base_cls012/test/yolo"
set2_raw_data = "/home/daree/nas/dataset2/images"

set3_data = "/home/daree/nas/dataset3"
set3_sample = "/home/daree/code/samples"
raw_data = '/home/daree/data/pothole_data/raw/'

datas = [set1_raw_data, set2_raw_data]

########################### fullsize image set 사용 ###########################
## model setting ##
padding = ['FIX'] # 'PIXEL', 'FIX2', 
padding_size = [0, 50 ,100, 150, 200, 224, 256, 336, 384, 448]
use_bbox = ['False', 'True']
use_shift = ['False', 'True']
soft_label_ratio = [0.7]

## model setting ##


base = """CUDA_VISIBLE_DEVICES=1 python main.py \
            --model convnext_base --drop_path 0.2 --input_size 224 \
            --batch_size 128 --lr 5e-5 --update_freq 2 \
            --warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
            --layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
            --finetune checkpoint/convnext_base_22k_224.pth \
            --data_path /home/daree/nas/ambclss/1st_data \
            --eval_data_path /home/daree/nas/ambclss/1st_data \
            --model_ema true --model_ema_eval true \
            --data_set image_folder \
            --nb_classes 2 \
            --use_cropimg=False \
            --auto_resume=False \
            --test_val_ratio 0.1 0.1 \
            --use_cropimg False \
            --save_ckpt False \
            --use_class 0"""

for pad in padding:
    for pad_size in padding_size:
        for bbox in use_bbox:
            for shift in use_shift:
                for ratio in soft_label_ratio:
                    name = f'pad_{pad}_padsize_{pad_size:.1f}_box_{bbox}_shift_{shift}_ratio_{ratio}'
                    if not os.path.isdir(os.getcwd() + '/log/' + name):
                        os.system(f"""{base} \
                                --padding {pad}\
                                --padding_size {pad_size}\
                                --use_bbox {bbox}\
                                --use_shift {shift}\
                                --output_dir results/b/{name} \
                                --soft_label_ratio {ratio} \
                                --log_name {name}""")
########################### test raw image 사용 ###########################



########################### test crop image 사용 ###########################
# base = "python main.py --lr 1e-6 --use_cropimg=True --auto_resume=False --drop_path 0.2 --layer_decay 0.8 --data_set image_folder"
# inputsize = [336, 336]
# data = [set1_data, set2_data]
# for i, ckpt in enumerate([set1_model]):
#     os.system(f"""{base} \
#             --resume {ckpt} \
#             --input_size {inputsize[i]} \
#             --eval_data_path {data[1]}\
#             --data_path {data[1]}\
#             --eval True""")
########################### test crop image 사용 ###########################

########################### test raw image 사용 ###########################
# base = """CUDA_VISIBLE_DEVICES=0 python /home/daree/code/main.py \
#             --lr 1e-6 \
#             --use_cropimg=False \
#             --auto_resume=False \
#             --drop_path 0.2 \
#             --layer_decay 0.8 \
#             --test_val_ratio 1.0 0.0 \
#             --use_class 0 \
#             --eval True"""

# for i, ckpt in enumerate(models):
#     os.system(f"""{base} \
#             --resume {ckpt} \
#             --input_size {inputsize[i]} \
#             --eval_data_path {set3_sample}\
#             --data_path {set3_sample}\
#             --padding {padding[i]}\
#             --padding_size {padding_size[i]}\
#             --use_bbox {use_bbox[i]}\
#             --use_shift {use_shift[i]}""")
########################### test raw image 사용 ###########################


########################### pred evaluation ###########################
# base = """CUDA_VISIBLE_DEVICES=0 python /home/daree/code/main.py \
#             --lr 1e-6 \
#             --use_cropimg=False \
#             --auto_resume=False \
#             --drop_path 0.2 \
#             --layer_decay 0.8 \
#             --test_val_ratio 1.0 0.0 \
#             --use_class 0 \
#             --pred_eval True \
#             --pred True"""

# for j, data in enumerate(datas):
#     for i, ckpt in enumerate(models):
#         name = f'Mset{i+1}_Dset{j+1}_'
#         os.system(f"""{base} \
#                 --resume {ckpt} \
#                 --input_size {inputsize[i]} \
#                 --eval_data_path {data} \
#                 --data_path {data} \
#                 --padding {padding[i]} \
#                 --padding_size {padding_size[i]} \
#                 --use_bbox {use_bbox[i]} \
#                 --pred_eval_name {name} \
#                 --use_shift {use_shift[i]}""")
########################### pred evaluation ###########################
       



