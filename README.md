# [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)

Official PyTorch implementation of **ConvNeXt**, from the following paper:

[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545). CVPR 2022.\
[Zhuang Liu](https://liuzhuang13.github.io), [Hanzi Mao](https://hanzimao.me/), [Chao-Yuan Wu](https://chaoyuan.org/), [Christoph Feichtenhofer](https://feichtenhofer.github.io/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/) and [Saining Xie](https://sainingxie.com)\
Facebook AI Research, UC Berkeley\
[[`arXiv`](https://arxiv.org/abs/2201.03545)][[`video`](https://www.youtube.com/watch?v=QzCjXqFnWPE)]

--- 

<p align="center">
<img src="https://user-images.githubusercontent.com/8370623/180626875-fe958128-6102-4f01-9ca4-e3a30c3148f9.png" width=100% height=100% 
class="center">
</p>

We propose **ConvNeXt**, a pure ConvNet model constructed entirely from standard ConvNet modules. ConvNeXt is accurate, efficient, scalable and very simple in design.

## Catalog
- [x] ImageNet-1K Training Code  
- [x] ImageNet-22K Pre-training Code  
- [x] ImageNet-1K Fine-tuning Code  
- [x] Downstream Transfer (Detection, Segmentation) Code
- [x] Image Classification [\[Colab\]](https://colab.research.google.com/drive/1CBYTIZ4tBMsVL5cqu9N_-Q3TBprqsfEO?usp=sharing) and Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/convnext)
- [x] Fine-tune on CIFAR with Weights & Biases logging [\[Colab\]](https://colab.research.google.com/drive/1ijAxGthE9RENJJQRO17v9A7PTd1Tei9F?usp=sharing)



<!-- ✅ ⬜️  -->

## Results and Pre-trained Models
### ImageNet-1K trained models

| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| ConvNeXt-T | 224x224 | 82.1 | 28M | 4.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth) |
| ConvNeXt-S | 224x224 | 83.1 | 50M | 8.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth) |
| ConvNeXt-B | 224x224 | 83.8 | 89M | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth) |
| ConvNeXt-B | 384x384 | 85.1 | 89M | 45.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth) |
| ConvNeXt-L | 224x224 | 84.3 | 198M | 34.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth) |
| ConvNeXt-L | 384x384 | 85.5 | 198M | 101.0G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth) |

### ImageNet-22K trained models

| name | resolution |acc@1 | #params | FLOPs | 22k model | 1k model |
|:---:|:---:|:---:|:---:| :---:| :---:|:---:|
| ConvNeXt-T | 224x224 | 82.9 | 29M | 4.5G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth)   | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_224.pth)
| ConvNeXt-T | 384x384 | 84.1 | 29M | 13.1G |     -          | [model](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth)
| ConvNeXt-S | 224x224 | 84.6 | 50M | 8.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth)   | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_224.pth)
| ConvNeXt-S | 384x384 | 85.8 | 50M | 25.5G |     -          | [model](https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_1k_384.pth)
| ConvNeXt-B | 224x224 | 85.8 | 89M | 15.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth)   | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth)
| ConvNeXt-B | 384x384 | 86.8 | 89M | 47.0G |     -          | [model](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth)
| ConvNeXt-L | 224x224 | 86.6 | 198M | 34.4G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth)  | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth)
| ConvNeXt-L | 384x384 | 87.5 | 198M | 101.0G |    -         | [model](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth)
| ConvNeXt-XL | 224x224 | 87.0 | 350M | 60.9G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth) | [model](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth)
| ConvNeXt-XL | 384x384 | 87.8 | 350M | 179.0G |  -          | [model](https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth)


### ImageNet-1K trained models (isotropic)
| name | resolution |acc@1 | #params | FLOPs | model |
|:---:|:---:|:---:|:---:| :---:|:---:|
| ConvNeXt-S | 224x224 | 78.7 | 22M | 4.3G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_iso_small_1k_224_ema.pth) |
| ConvNeXt-B | 224x224 | 82.0 | 87M | 16.9G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_iso_base_1k_224_ema.pth) |
| ConvNeXt-L | 224x224 | 82.6 | 306M | 59.7G | [model](https://dl.fbaipublicfiles.com/convnext/convnext_iso_large_1k_224_ema.pth) |


## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Evaluation
We give an example evaluation command for a ImageNet-22K pre-trained, then ImageNet-1K fine-tuned ConvNeXt-B:

Single-GPU
```
python main.py --model convnext_base --eval true \
--resume https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```
Multi-GPU
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model convnext_base --eval true \
--resume https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth \
--input_size 224 --drop_path 0.2 \
--data_path /path/to/imagenet-1k
```

This should give 
```
* Acc@1 85.820 Acc@5 97.868 loss 0.563
```

- For evaluating other model variants, change `--model`, `--resume`, `--input_size` accordingly. You can get the url to pre-trained models from the tables above. 
- Setting model-specific `--drop_path` is not strictly required in evaluation, as the `DropPath` module in timm behaves the same during evaluation; but it is required in training. See [TRAINING.md](TRAINING.md) or our paper for the values used for different models.

## Training
See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions.

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library, [DeiT](https://github.com/facebookresearch/deit) and [BEiT](https://github.com/microsoft/unilm/tree/master/beit) repositories.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}
```

---
---
## Execution
1. [main.py](main.py) 
    - 전체 code 실행
    - get_args_parser( )의 parameter 수정

    ```
    --data_path : Train data  위치 --> list 형태로 입력
    --eval_data_path : Validation data 위치 --> string 형태로 입력
    --nb_classes : class 수 지정 --> amb class를 포함하는 경우 4로 지정
    --use_softlabel : soft label을 사용할 경우 True 설정
    --soft_label_ratio : soft label의 target ratio(amb_pos : amb_neg)
    --label_ratio : pos,neg target ratio
    # Image Crop & Padding --> data set image load와 관련된 항목
    ```

2. [engine.py](engine.py)
    - train, validation, prediction(check result)과 관련된 코드
    - prediction(evaluation) 모듈(새로 추가)
      - prediction 모듈은 train 결과를 불러와 evaluation
      - args.pred가 True로 설정되면 prediction진입 
      - 결과 image구분과 graph 결과 생성

3. [preprocess_data.py](preprocess_data.py) & [datasets.py](datasets.py)
    - dataset 생성과 image crop 및 전처리 관련 코드
    - 두 파일내에 있는 function 및 class를 활용
    - split_data
      - 데이터 셋을 train, val, test으로 구분하는 모듈
      - 기존에 split파일이 있으면 새로 목록을 구성하지 않음
      - tset, val 비율에 따라 파일명 생성
    - make_list
      - dataloader를 통해서 불러올 raw데이터 list생성

4. [softLabelLoss.py](softLabelLoss.py)
    - 다양한 loss를 적용하기 위한 code
    - args.np_class 가 2 혹은 4일 경우 target을 수정
    - args.use_sortlabel 이 True일 경우 4class -> 2class로 변경

5. [load_board.py](load_board.py)
    - log에 저장되어 있는 tensorboard 저장파일 파싱
    - csv로 결과 저장
    - path에 log파일이 저장되어 있는 폴더 이름 수정 후 실행
    - 추가로 필요한 부분이 있는 경우 col, val 부분을 수정
    - csv 폴더에 csv파일 생성

6. [run.py](run_0.py)
    - 루틴한 실험을 진행하기 위한 코드
    - base : 변동이 없는 parameter 설정
    - 변동이 있는 parameter는 list형태로 지정하고 for문으로 실행