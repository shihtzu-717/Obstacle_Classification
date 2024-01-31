if [ $# -eq 1 ]; then
    model_name=$1
    auto_augment=rand-m9-mstd0.5-inc1
elif [ $# -eq 2 ]; then
    model_name=$2
    auto_augment=rand-m9-mstd0.5-inc1
elif [ $# -eq 3 ]; then
    model_name=$2
    auto_augment=$3
elif [ $# -ne 3 ]; then
    echo "Usage: $0 data_name model_name auto_augment"
    exit 1
fi
data_name=$1
mkdir -p results/b/${model_name}

echo "train_base ${data_name} ${model_name} ${auto_augment}"
python -m torch.distributed.launch --nproc_per_node=2 main.py \
--model convnext_base --drop_path 0.2 --input_size 224 \
--batch_size 128 --lr 5e-5 --update_freq 2 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--aa ${auto_augment} \
--finetune checkpoint/convnext_base_22k_224.pth \
--data_path /data/pothole_data/out/${data_name}/train \
--eval_data_path /data/pothole_data/out/${data_name}/val \
--output_dir results/b/${model_name} \
--model_ema true --model_ema_eval true \
--data_set image_folder \
--nb_classes 2 \
--log_dir results/b/${model_name}/log &> results/b/${model_name}/train.out

find results/b/${model_name} -type f -regex '.*checkpoint-[0-9]+[.]pth' -delete

echo "eval_base ${data_name} ${model_name} ${auto_augment}"
python main.py \
--model convnext_base \
--eval true \
--resume results/b/${model_name}/checkpoint-best.pth \
--input_size 224 \
--drop_path 0.2 \
--layer_decay 0.8 \
--data_set image_folder \
--nb_classes 2 \
--data_path /data/pothole_data/out/${data_name}/train \
--eval_data_path /data/pothole_data/out/${data_name}/test/train &> results/b/${model_name}/eval_train.out

python main.py \
--model convnext_base \
--eval true \
--resume results/b/${model_name}/checkpoint-best.pth \
--input_size 224 \
--drop_path 0.2 \
--layer_decay 0.8 \
--data_set image_folder \
--nb_classes 2 \
--data_path /data/pothole_data/out/${data_name}/train \
--eval_data_path /data/pothole_data/out/${data_name}/test/yolo &> results/b/${model_name}/eval_yolo.out