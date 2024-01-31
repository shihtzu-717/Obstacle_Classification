if [ $# -eq 1 ]; then
    model_name=$1
elif [ $# -eq 2 ]; then
    model_name=$2
    auto_augment=rand-m9-mstd0.5-inc1
elif [ $# -ne 2 ]; then
    echo "Usage: $0 data_name model_name auto_augment"
    exit 1
fi
data_name=$1
echo "test_base ${data_name} ${model_name}"
python main.py \
--model convnext_base \
--eval true \
--resume results/set_1/b/${model_name}/checkpoint-best.pth \
--input_size 224 \
--drop_path 0.2 \
--layer_decay 0.8 \
--data_set image_folder \
--nb_classes 2 \
--data_path /data/pothole_data/out_test/${data_name}/test/yolo \
--eval_data_path /data/pothole_data/out_test/${data_name}/test/yolo &> results/set_1/b/${model_name}/test_yolo.out