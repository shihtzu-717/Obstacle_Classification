if [ $# -ne 1 ]; then
    echo "Usage: $0 param"
    exit 1
fi
data_name=$1
mkdir -p results/xl/${data_name}

echo "train_xlarge ${data_name}"
python -m torch.distributed.launch --nproc_per_node=2 main.py \
--model convnext_xlarge --drop_path 0.4 --input_size 224 \
--batch_size 64 --lr 5e-5 --update_freq 4 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--layer_decay 0.8 --head_init_scale 0.001 --cutmix 0 --mixup 0 \
--finetune checkpoint/convnext_xlarge_22k_224.pth \
--data_path /data/pothole_data/out/${data_name}/train \
--eval_data_path /data/pothole_data/out/${data_name}/val \
--output_dir results/xl/${data_name} \
--model_ema true --model_ema_eval true \
--data_set image_folder \
--nb_classes 2 \
--log_dir results/xl/${data_name}/log &> results/xl/${data_name}/train.out

find results/xl/${data_name} -type f -regex '.*checkpoint-[0-9]+[.]pth' -delete

echo "eval_base ${data_name}"
python main.py \
--model convnext_xlarge \
--eval true \
--resume results/xl/${data_name}/checkpoint-best.pth \
--input_size 224 \
--drop_path 0.4 \
--layer_decay 0.8 \
--data_set image_folder \
--nb_classes 2 \
--data_path /data/pothole_data/out/${data_name}/train \
--eval_data_path /data/pothole_data/out/${data_name}/test/train &> results/xl/${data_name}/eval_train.out

python main.py \
--model convnext_xlarge \
--eval true \
--resume results/xl/${data_name}/checkpoint-best.pth \
--input_size 224 \
--drop_path 0.4 \
--layer_decay 0.8 \
--data_set image_folder \
--nb_classes 2 \
--data_path /data/pothole_data/out/${data_name}/train \
--eval_data_path /data/pothole_data/out/${data_name}/test/yolo &> results/xl/${data_name}/eval_yolo.out