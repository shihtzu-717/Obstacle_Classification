if [ $# -ne 4 ]; then
    echo "Usage: $0 param1 param2 param3 param4"
    exit 1
fi
input_dir=/data/pothole_data/out/$1
target_dir="/data/pothole_data/out/$1_yp$2n$3tp$4"

rm -rf ${target_dir}
echo "make ${target_dir}"
mkdir -p ${target_dir}/train/positive ${target_dir}/val/positive
ln -s ../$1/test ${target_dir}/test

for ((data_idx=0 ; data_idx < $2 ; data_idx++));
do
    ln -s ../../../$1/train/positive/yolo ${target_dir}/train/positive/yolo_${data_idx}
    ln -s ../../../$1/val/positive/yolo ${target_dir}/val/positive/yolo_${data_idx}
done
for ((data_idx=0 ; data_idx < $3 ; data_idx++));
do
    ln -s ../../$1/train/negative ${target_dir}/train/negative_${data_idx}
    ln -s ../../$1/val/negative ${target_dir}/val/negative_${data_idx}
done
for ((data_idx=0 ; data_idx < $4 ; data_idx++));
do
    ln -s ../../../$1/train/positive/train ${target_dir}/train/positive/train_${data_idx}
    ln -s ../../../$1/val/positive/train ${target_dir}/val/positive/train_${data_idx}
done