# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import torch
import preprocess_data
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
import utils

from typing import Iterable, Optional
from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from timm.models import create_model
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from pathlib import Path
from torch import nn


def create_images_with_conf(image_path, re, label, pred_save_path, use_cropimg, args):
    background_color = (255, 255, 255)  # 흰색 배경
    width, height = 2560, 1440  # 배경의 가로와 세로 크기
    background = np.ones((height, width, 3), dtype=np.uint8) * background_color
    if not args.use_cropimg:
        crop_img = preprocess_data.crop_image(
                image_path = image_path, 
                bbox = re[4], 
                padding = args.padding, 
                padding_size = args.padding_size, 
                use_shift = args.use_shift, 
                use_bbox = args.use_bbox, 
                imsave = args.imsave
            )
        crop_img = cv2.resize(crop_img, (args.input_size, args.input_size))

    image = cv2.imread(image_path)
    ih, iw, ic = image.shape
    if not use_cropimg:
        x, y, w, h = float(re[4][0])*iw, float(re[4][1])*ih, float(re[4][2])*iw, float(re[4][3])*ih
        x1 = int(round(x-w/2))
        y1 = int(round(y-h/2))
        x2 = int(round(x+w/2))
        y2 = int(round(y+h/2))
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, color=[0, 255, 255])

    # 이미지 중앙 위치 계산
    bg_height, bg_width, _ = background.shape
    img_height, img_width, _ = image.shape
    x = (bg_width - img_width) // 2
    y = (bg_height - img_height) // 2

    # 이미지를 배경 중앙에 추가
    background[y:y+img_height, x:x+img_width] = image

    # 텍스트 추가
    text_top = f"predicted class: {label},  {re[2]:.2f}%"
    text_bottom = f"target: {re[3]}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2  # 폰트 크기를 2로 변경
    text_color = (0, 0, 0)  # 검은색 텍스트
    text_thickness = 3

    # 텍스트 위치 계산
    text_top_size, _ = cv2.getTextSize(text_top, font, font_scale, text_thickness)
    text_bottom_size, _ = cv2.getTextSize(text_bottom, font, font_scale, text_thickness)
    text_top_x = (bg_width - text_top_size[0]) // 2
    text_top_y = y - 20  # 이미지 중앙 위쪽에 20px 떨어진 위치
    text_bottom_x = (bg_width - text_bottom_size[0]) // 2
    text_bottom_y = y + img_height + text_bottom_size[1] + 20  # 이미지 중앙 아래쪽에 20px 떨어진 위치

    # 텍스트를 배경에 추가
    # cv2.putText(background, text_top, (text_top_x, text_top_y), font, font_scale, text_color, text_thickness, cv2.LINE_AA)
    # cv2.putText(background, text_bottom, (text_bottom_x, text_bottom_y), font, font_scale, text_color, text_thickness, cv2.LINE_AA)
    background_um = cv2.UMat(background)
    cv2.putText(background_um, text_top, (text_top_x, text_top_y), font, font_scale, text_color, text_thickness, cv2.LINE_AA)
    cv2.putText(background_um, text_bottom, (text_bottom_x, text_bottom_y), font, font_scale, text_color, text_thickness, cv2.LINE_AA)

    # # UMat을 다시 일반 이미지로 변환
    background = background_um.get()

    # 이미지 데이터 타입 변환
    background = np.asarray(background, dtype=np.uint8)

    # 이미지 저장
    fn = (os.path.basename(image_path)).split('.')[0]+'_'+re[5]+'.jpg'
    output_path = str(Path(pred_save_path) / label / 'inference' / fn)
    cv2.imwrite(output_path, background)

    if re[3] == label:
        true_data_path = str(Path(pred_save_path) / label / 'true_data')
        shutil.copy(image_path, os.path.join(true_data_path, 'images'))
        annot_path = image_path.replace('images', 'annotations')
        annot_path = annot_path.replace('.jpg', '.txt')
        annot_path = annot_path.replace('.png', '.txt')
        if not use_cropimg:
            shutil.copy(annot_path, os.path.join(true_data_path, 'annotations'))
            cv2.imwrite(os.path.join(true_data_path, 'crop_img', fn), crop_img)
        cv2.imwrite(os.path.join(true_data_path, 'inference', fn), background)
    else:
        false_data_path = str(Path(pred_save_path) / label / 'false_data')
        shutil.copy(image_path, os.path.join(false_data_path, 'images'))
        annot_path = image_path.replace('images', 'annotations')
        annot_path = annot_path.replace('.jpg', '.txt')
        annot_path = annot_path.replace('.png', '.txt')
        if not use_cropimg:
            shutil.copy(annot_path, os.path.join(false_data_path, 'annotations'))
            cv2.imwrite(os.path.join(false_data_path, 'crop_img', fn), crop_img)
        cv2.imwrite(os.path.join(false_data_path, 'inference', fn), background)

def softmax(x):
    exp_x = torch.exp(x - torch.max(x))
    softmax_x = exp_x / torch.sum(exp_x)
    return softmax_x

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False, use_softlabel=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    optimizer.zero_grad()

    # for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = batch[0]
        targets = batch[-1]

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else: # full precision
            output, outvect = model(samples, onlyfc=False)
            loss = criterion(output, targets)
  
        loss_value = loss.item()

        if not math.isfinite(loss_value): # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else: # full precision
            loss /= update_freq
            loss.backward()
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            if use_softlabel:
                targets = torch.tensor([0 if i==2 or i==0 else 1 for i in targets]).to(device)
            class_acc = (output.max(-1)[-1] == targets).float().mean()*100
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            if use_amp:
                log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion=torch.nn.CrossEntropyLoss(), use_amp=False, use_softlabel=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header): # 학습할 때는 data가 data_loader_val임 
        images = batch[0].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)
        if use_softlabel:
            target = torch.tensor([0 if i==2 or i==0 else 1 for i in target]).to(device)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)
        
        acc1, acc2 = accuracy(output, target, topk=(1, 2)) # top5는 의미 없어 2로 변경

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)

        for class_name, class_id in data_loader.dataset.class_to_idx.items():
            if use_softlabel:
                class_id = 0 if class_id==2 or class_id==0 else 1 
                class_name = 'negative' if class_name == 'amb_neg' else class_name
                class_name = 'positive' if class_name == 'amb_pos' else class_name

            mask = (target == class_id)
            target_class = torch.masked_select(target, mask)
            data_size = target_class.shape[0]
            if data_size > 0:
                mask = mask.unsqueeze(1).expand_as(output)
                output_class = torch.masked_select(output, mask)
                if use_softlabel:
                    output_class = output_class.view(-1, 2)
                else:
                    output_class = output_class.view(-1, len(data_loader.dataset.class_to_idx))
                acc1_class, acc2_class = accuracy(output_class, target_class, topk=(1, 2)) # top5는 의미 없어 2로 변경
                metric_logger.meters[f'acc1_{class_name}'].update(acc1_class.item(), n=data_size)
                metric_logger.meters[f'acc2_{class_name}'].update(acc2_class.item(), n=data_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@2 {top2.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top2=metric_logger.acc2, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def prediction(args, device):
    import sys
    import random
    from preprocess_data import make_dataset_file
    from datasets import PotholeDataset, get_split_data
    from sklearn.metrics import precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay, classification_report

    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    totorch = transforms.ToTensor()

    # 모델 생성 train한 모델과 같은 모델을 생성해야 함.
    model = create_model(
        args.model, 
        pretrained=False, 
        num_classes=args.nb_classes, 
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        head_init_scale=args.head_init_scale,
    )
    model.to(device)

    # Trained Model
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model,
        optimizer=None, loss_scaler=None, model_ema=None)
    model.eval()

    # Data laod     
    data_list = []
    result = []
    # sets = get_split_data(data_root=Path(args.eval_data_path), 
    #                               test_r=args.test_val_ratio[0], 
    #                               val_r=args.test_val_ratio[1], 
    #                               file_write=args.split_file_write,
    #                               label_list = args.label_list) 
    # data_list = sets['test'] if len(sets['test']) > 0 else sets['val']
    if args.path_type:
        for path in args.eval_data_path:
            settmp = get_split_data(data_root=Path(path), 
                                    test_r=args.test_val_ratio[0], 
                                    val_r=args.test_val_ratio[1], 
                                    file_write=args.split_file_write,
                                    label_list = args.label_list, 
                                    use_cropimg = args.use_cropimg)
            data_list += settmp['val']
            
    elif args.txt_type:
        if args.valid_txt_path == "":
            print("Please Check the valid_txt_path")
            sys.exit(1)
        data_list = make_dataset_file(args.valid_txt_path)

        
    random.shuffle(data_list)  # Data list shuffle
    tonorm = transforms.Normalize(mean, std)  # Transform 생성

    idx = 0
    for data in tqdm(data_list, desc='Image Cropping... '):
        if data.class_id not in args.use_class:
            continue
        if args.use_cropimg:
            crop_img = cv2.imread(str(data[0] / data.image_path))
        else:
            crop_img = preprocess_data.crop_image(
                image_path = data[0] / data.image_path, 
                bbox = data.bbox, 
                padding = args.padding, 
                padding_size = args.padding_size, 
                use_shift = args.use_shift, 
                use_bbox = args.use_bbox, 
                imsave = args.imsave
            )

        # File 이름에 label이 있는지 확인
        spltnm = str(data[1]).split('_')
        target = int(spltnm[0][1]) if spltnm[0][0] == 't' else -1

        # label이 따로 있는 경우 아래 4가지 label로 지정
        if target == -1:
            if data[1] == 'amb_neg':
                target = 0 # amb_neg
            elif data[1] == 'amb_pos':
                target = 1 # amb_pos
            elif data[1] == 'negative':
                target = 2 # neg
            elif data[1] == 'positive':
                target = 3 # pos
            else:
                target =-1

        crop_img = cv2.resize(crop_img, (args.input_size, args.input_size))
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        pil_image=Image.fromarray(crop_img)
        input_tensor = totorch(pil_image).to(device)
        input_tensor = input_tensor.unsqueeze(dim=0)
        input_tensor = tonorm(input_tensor)
        
        # model output 
        output_tensor = model(input_tensor) 
        pred, conf = int(torch.argmax(output_tensor).detach().cpu().numpy()), float((torch.max(output_tensor)).detach().cpu().numpy())
        # softmax = nn.Softmax()
        # probs = softmax(output_tensor)

        # output = np.squeeze(output_tensor)
        probs = softmax(output_tensor) # softmax 통과

        probs_max = ((torch.max(probs)).detach().cpu().numpy())*100

        if (args.conf > probs_max) and (pred == 1): # amb_pos 인 경우 
            pred = 0
        elif (args.conf > probs_max) and (pred == 3): # positive 인 경우 
            pred = 2

        result.append((pred, probs_max, target, data[0] / data.image_path, data.label, data.bbox, str(idx)))
        idx += 1
        
    ##################################### save result image & anno #####################################

    if args.pred_save:
        import os
        if args.use_softlabel == False and args.nb_classes == 4 : 
            os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'images', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'annotations', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'images', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'annotations', exist_ok=True)
        os.makedirs(Path(args.pred_save_path) /'negative' / 'images', exist_ok=True)
        os.makedirs(Path(args.pred_save_path) /'negative' / 'annotations', exist_ok=True)
        os.makedirs(Path(args.pred_save_path) /'positive' / 'images', exist_ok=True)
        os.makedirs(Path(args.pred_save_path) /'positive' / 'annotations', exist_ok=True)
        if args.pred_save_with_conf:
            if args.use_softlabel == False and args.nb_classes == 4 : 
                os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'inference', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'inference', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'true_data' / 'images', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'true_data' / 'annotations', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'true_data' / 'inference', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'false_data' / 'images', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'false_data' / 'annotations', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'false_data' / 'inference', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'true_data' / 'images', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'true_data' / 'annotations', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'true_data' / 'inference', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'false_data' / 'images', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'false_data' / 'annotations', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'false_data' / 'inference', exist_ok=True)
                if not args.use_cropimg:
                    os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'true_data' / 'crop_img', exist_ok=True)
                    os.makedirs(Path(args.pred_save_path) /'amb_neg' / 'false_data' / 'crop_img', exist_ok=True)
                    os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'true_data' / 'crop_img', exist_ok=True)
                    os.makedirs(Path(args.pred_save_path) /'amb_pos' / 'false_data' / 'crop_img', exist_ok=True)

            os.makedirs(Path(args.pred_save_path) /'negative' / 'inference', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'positive' / 'inference', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'negative' / 'true_data' / 'images', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'negative' / 'true_data' / 'annotations', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'negative' / 'true_data' / 'inference', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'negative' / 'false_data' / 'images', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'negative' / 'false_data' / 'annotations', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'negative' / 'false_data' / 'inference', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'positive' / 'true_data' / 'images', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'positive' / 'true_data' / 'annotations', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'positive' / 'true_data' / 'inference', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'positive' / 'false_data' / 'images', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'positive' / 'false_data' / 'annotations', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'positive' / 'false_data' / 'inference', exist_ok=True)
            if not args.use_cropimg:
                os.makedirs(Path(args.pred_save_path) /'negative' / 'true_data' / 'crop_img', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'negative' / 'false_data' / 'crop_img', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'positive' / 'true_data' / 'crop_img', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'positive' / 'false_data' / 'crop_img', exist_ok=True)

        if args.use_softlabel and args.nb_classes == 4:
            amb_neg = [(x[3], 'amb_neg', x[1], x[4], x[5], x[6]) for x in result if x[0]==0] + [(x[3], 'negative', x[1], x[4], x[5], x[6]) for x in result if x[0]==2]
            amb_pos = [(x[3], 'amb_pos', x[1], x[4], x[5], x[6]) for x in result if x[0]==1] + [(x[3], 'positive', x[1], x[4], x[5], x[6]) for x in result if x[0]==3]

            # amb_neg = []
            # amb_pos = []
            # for x in result:
            #     if x[0] == 0:
            #         amb_neg.append((x[3], 'amb_neg', x[1], x[4], x[5], x[6]))
            #     elif x[0] == 1:
            #         if x[1] >= args.conf:
            #             amb_pos.append((x[3], 'amb_pos', x[1], x[4], x[5], x[6]))
            #         else:
            #             amb_neg.append((x[3], 'amb_neg', x[1], x[4], x[5], x[6]))
            #     elif x[0] == 2:
            #         amb_neg.append((x[3], 'negative', x[1], x[4], x[5], x[6]))
            #     elif x[0] == 3:
            #         if x[1] >= args.conf:
            #             amb_pos.append((x[3], 'positive', x[1], x[4], x[5], x[6]))
            #         else:
            #             amb_neg.append((x[3], 'negative', x[1], x[4], x[5], x[6]))
                    
        else:
            amb_neg = [(x[3], 'amb_neg', x[1], x[4], x[5], x[6]) for x in result if x[0]==0]
            amb_pos = [(x[3], 'amb_pos', x[1], x[4], x[5], x[6]) for x in result if x[0]==1]
            neg = [(x[3], 'negative', x[1], x[4], x[5], x[6]) for x in result if x[0]==2]
            pos = [(x[3], 'positive', x[1], x[4], x[5], x[6]) for x in result if x[0]==3]

            # amb_neg = []
            # amb_pos = []
            # neg = []
            # pos = []
            # for x in result:
            #     if x[0] == 0:
            #         amb_neg.append((x[3], 'amb_neg', x[1], x[4], x[5], x[6]))
            #     elif x[0] == 1:
            #         if x[1] >= args.conf:
            #             amb_pos.append((x[3], 'amb_pos', x[1], x[4], x[5], x[6]))
            #         else:
            #             amb_neg.append((x[3], 'amb_neg', x[1], x[4], x[5], x[6]))
            #     elif x[0] == 2:
            #         neg.append((x[3], 'negative', x[1], x[4], x[5], x[6]))
            #     elif x[0] == 3:
            #         if x[1] >= args.conf:
            #             pos.append((x[3], 'positive', x[1], x[4], x[5], x[6]))
            #         else:
            #             neg.append((x[3], 'negative', x[1], x[4], x[5], x[6]))

            # amb_neg = [(x[3], 'amb_neg', x[1], x[4], x[5], x[6]) for x in result if x[0]==0 and x[1] > 90.0]
            # amb_pos = [(x[3], 'amb_pos', x[1], x[4], x[5], x[6]) for x in result if x[0]==1 and x[1] > 90.0]
            # neg = [(x[3], 'negative', x[1], x[4], x[5], x[6]) for x in result if x[0]==2 and x[1] > 90.0]
            # pos = [(x[3], 'positive', x[1], x[4], x[5], x[6]) for x in result if x[0]==3 and x[1] > 90.0]
        
        
        with open(Path(args.pred_save_path)/"conf_avg.txt", 'w') as f:
            f.write('')
        an_sum = 0
        an_min = 100.0
        an_max = 0
        for an in tqdm(amb_neg, desc='Class_0 images copying... '):
            img_path = str(an[0])
            an_conf = float(an[2])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            # shutil.copy(an[0], Path(args.pred_save_path) /'amb_neg' / 'images')
            # shutil.copy(annot_path, Path(args.pred_save_path) / 'amb_neg' / 'annotations')
            if args.nb_classes == 2 or (args.use_softlabel and args.nb_classes == 4):
                shutil.copy(an[0], Path(args.pred_save_path) /'negative' / 'images')
                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / 'negative' / 'annotations')
            else:
                shutil.copy(an[0], Path(args.pred_save_path) /'amb_neg' / 'images')
                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / 'amb_neg' / 'annotations')
            if args.pred_save_with_conf:
                if args.nb_classes == 2 or (args.use_softlabel and args.nb_classes == 4):
                    create_images_with_conf(img_path, an, 'negative', args.pred_save_path, args.use_cropimg, args)
                else:
                    create_images_with_conf(img_path, an, 'amb_neg', args.pred_save_path, args.use_cropimg, args)
                    
            an_sum = an_sum + an_conf
            if an_max < an_conf:
                an_max = an_conf
            if an_min > an_conf:
                an_min = an_conf
        try:
            print(f"Class_0 AVG: {an_sum / len(amb_neg):.2f}%")
            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write(f"Class_0 CNT: {len(amb_neg)}, ")
                f.write(f"Class_0 AVG: {an_sum / len(amb_neg):.2f}%, ")
                f.write(f"Class_0 MAX: {an_max:.2f}%, ")
                f.write(f"Class_0 MIN: {an_min:.2f}%\n")
        except ZeroDivisionError:
            print("No Class_0 Data")

        ap_sum = 0
        ap_min = 100.0
        ap_max = 0
        for ap in tqdm(amb_pos, desc='Class_1 images copying... '):
            img_path = str(ap[0])
            ap_conf = float(ap[2])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            if args.nb_classes == 2 or (args.use_softlabel and args.nb_classes == 4):
                shutil.copy(ap[0], Path(args.pred_save_path) / 'positive' / 'images')
                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / 'positive' / 'annotations')
            else:
                shutil.copy(ap[0], Path(args.pred_save_path) / 'amb_pos' / 'images')
                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / 'amb_pos' / 'annotations')
            # shutil.copy(ap[0], Path(args.pred_save_path) / 'amb_pos' / 'images')
            # shutil.copy(annot_path, Path(args.pred_save_path) / 'amb_pos' / 'annotations')
            if args.pred_save_with_conf:
                if args.nb_classes == 2 or (args.use_softlabel and args.nb_classes == 4):
                    create_images_with_conf(img_path, ap, 'positive', args.pred_save_path, args.use_cropimg, args)
                else:
                    create_images_with_conf(img_path, ap, 'amb_pos', args.pred_save_path, args.use_cropimg, args)
                    
            ap_sum = ap_sum + ap_conf
            if ap_max < ap_conf:
                ap_max = ap_conf
            if ap_min > ap_conf:
                ap_min = ap_conf
        try:
            print(f"Class_1 AVG: {ap_sum / len(amb_pos):.2f}%")
            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write(f"Class_1 CNT: {len(amb_pos)}, ")
                f.write(f"Class_1 AVG: {ap_sum / len(amb_pos):.2f}%, ")
                f.write(f"Class_1 MAX: {ap_max:.2f}%, ")
                f.write(f"Class_1 MIN: {ap_min:.2f}%\n")
        except ZeroDivisionError:
                print("No Class_1 Data")
        
        if (args.use_softlabel == False) and (args.nb_classes == 4):
            n_sum = 0
            n_min = 100.0
            n_max = 0
            for n in tqdm(neg, desc='Class_2 images copying... '):
                img_path = str(n[0])
                n_conf = float(n[2])
                annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                shutil.copy(n[0], Path(args.pred_save_path) /'negative' / 'images')
                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / 'negative' / 'annotations')
                if args.pred_save_with_conf:
                    create_images_with_conf(img_path, n, 'negative', args.pred_save_path, args.use_cropimg, args)

                n_sum = n_sum + n_conf
                if n_max < n_conf:
                    n_max = n_conf
                if n_min > n_conf:
                    n_min = n_conf
            try:
                print(f"Class_2 AVG: {n_sum / len(neg):.2f}%")
                with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                    f.write(f"Class_2 CNT: {len(neg)}, ")
                    f.write(f"Class_2 AVG: {n_sum / len(neg):.2f}%, ")
                    f.write(f"Class_2 MAX: {n_max:.2f}%, ")
                    f.write(f"Class_2 MIN: {n_min:.2f}%\n")
            except ZeroDivisionError:
                print("No Class_2 Data")

            p_sum = 0
            p_min = 100.0
            p_max = 0
            for p in tqdm(pos, desc='Class_3 images copying... '):
                img_path = str(p[0])
                p_conf = float(p[2])
                annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                shutil.copy(p[0], Path(args.pred_save_path) / 'positive' / 'images')
                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / 'positive' / 'annotations')
                if args.pred_save_with_conf:
                    create_images_with_conf(img_path, p, 'positive', args.pred_save_path, args.use_cropimg, args)
                    
                p_sum = p_sum + p_conf
                if p_max < p_conf:
                    p_max = p_conf
                if p_min > p_conf:
                    p_min = p_conf
            try:
                print(f"Class_3 AVG: {p_sum / len(pos):.2f}%")
                with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                    f.write(f"Class_3 CNT: {len(pos)}, ")
                    f.write(f"Class_3 AVG: {p_sum / len(pos):.2f}%, ")
                    f.write(f"Class_3 MAX: {p_max:.2f}%, ")
                    f.write(f"Class_3 MIN: {p_min:.2f}%\n")
            except ZeroDivisionError:
                print("No Class_3 Data")
        
    ##################################### save result image & anno #####################################

    ##################################### save evalutations #####################################
    if args.pred_eval:
        if np.sum(np.array(result)[...,2]) < 0:
            conf_TN = [x[1] for x in result if (x[0]==0)]
            conf_TP = [x[1] for x in result if (x[0]==1)]
            conf_FN = []
            conf_FP = []

            # index set    
            itn = [i for i in range(len(result)) if (result[i][0]==0)]
            itp = [i for i in range(len(result)) if (result[i][0]==1)]

            # histogram P-N 
            plt.hist((conf_TN, conf_TP), label=('Negative', 'Positive'),histtype='bar', bins=50)
            plt.xlabel('Confidence')
            plt.ylabel('Conunt')
            plt.legend(loc='upper left')
            # plt.savefig('image/'+args.pred_eval_name+'hist_PN.png')
            plt.savefig(args.pred_eval_name+'hist_PN.png')
            plt.close()

        else:
            y_pred = [i[0] for i in result]
            y_target = [i[2] for i in result]

            # y_pred = []
            # y_target = []
            # for i in result:
            #     if args.conf >= i[1] and i[0] == 1: # conf가 특정 값 이하이고 positive로 예측한 경우
            #         print(i[0], i[1])
            #         i[0] = 0
            #         print(i[0], i[1])
            #         y_pred.append(i[0])
            #         y_target.append(i[2])
            #     else:
            #         y_pred.append(i[0])
            #         y_target.append(i[2])

            pos_val = 3

            # 4class to 2class 변경
            if args.use_softlabel:
                y_pred = [0 if i==2 or i==0 else 1 for i in y_pred]
                y_target = [0 if i==2 or i==0 else 1 for i in y_target]
                pos_val = 1

            # 4class to 3class 변경
            if args.four_to_three:
                org_y_pred = [i[0] for i in result] 
                org_y_target = [i[2] for i in result]
                y_pred = []
                y_target = []

                for i in org_y_pred:
                    if i == 0 or i == 1:
                        y_pred.append(0)
                    elif i == 2:
                        y_pred.append(1)
                    else:
                        y_pred.append(2)
            
                for i in org_y_target:
                    if i == 0 or i == 1:
                        y_target.append(0)
                    elif i == 2:
                        y_target.append(1)
                    else:
                        y_target.append(2)
                pos_val = 2

            # precision recall 계산
            precision = precision_score(y_target, y_pred, average= "macro")
            recall = recall_score(y_target, y_pred, average= "macro")
            cm = confusion_matrix(y_target, y_pred)
            cm_display = ConfusionMatrixDisplay(cm).plot()
            if (args.use_softlabel == False) and (args.nb_classes == 4):
                cls_report = classification_report(y_target, y_pred, target_names=["amb_neg", "amb_pos", "Negative", "Positive"])

            else:
                try:
                    cls_report = classification_report(y_target, y_pred, target_names=["Negative", "Positive"])
                except:
                    cls_report = classification_report(y_target, y_pred, target_names=["Class_0"])
            
            plt.title('Precision: {0:.4f}, Recall: {1:.4f}'.format(precision, recall))
            # plt.savefig('image/'+args.pred_eval_name+'cm.png')
            plt.savefig(args.pred_eval_name+'cm.png')
            plt.close()
            print(cm)
            print('정밀도(Precision): {0:.4f}, 재현율(Recall): {1:.4f}\n'.format(precision, recall))
            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write('정밀도(Precision): {0:.4f}, 재현율(Recall): {1:.4f}\n'.format(precision, recall))
                f.write(cls_report)
                f.write('F1-score : {0:.4f}\n'.format(2 * (precision * recall) / (precision + recall)))
                f.write(str(cm))
            print(cls_report)
            print('F1-score : {0:.4f}\n'.format(2 * (precision * recall) / (precision + recall)))

            if args.eval_not_include_neg:
                not_include_neg_list = []
                for i in result:
                    if i[2] != 2:
                        y_pred = [i[0] for i in result]
                        y_target = [i[2] for i in result]
                        not_include_neg_list.append(i)
                result = not_include_neg_list

            # collect data 
            conf_TN = [x[1] for p, t, x in zip(y_pred, y_target, result) if p==t and p!=pos_val] 
            conf_TP = [x[1] for p, t, x in zip(y_pred, y_target, result) if p==t and p==pos_val] 
            conf_FN = [x[1] for p, t, x in zip(y_pred, y_target, result) if p!=t and p!=pos_val] 
            conf_FP = [x[1] for p, t, x in zip(y_pred, y_target, result) if p!=t and p==pos_val] 
            
            true_neg_over_conf = 0
            true_pos_over_conf = 0
            false_neg_over_conf = 0
            false_pos_over_conf = 0
            true_neg_under_conf = 0
            true_pos_under_conf = 0
            false_neg_under_conf = 0
            false_pos_under_conf = 0

            for i in conf_TP:
                if i >= args.conf:
                    true_pos_over_conf+=1
                else:
                    true_pos_under_conf+=1

            for i in conf_TN:
                if i >= args.conf:
                    true_neg_over_conf+=1
                else:
                    true_neg_under_conf+=1

            for i in conf_FP:
                if i >= args.conf:
                    false_pos_over_conf+=1
                else:
                    false_pos_under_conf+=1

            for i in conf_FN:
                if i >= args.conf:
                    false_neg_over_conf+=1
                else:
                    false_neg_under_conf+=1

            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write(f"\n\nTP: {true_pos_over_conf+true_pos_under_conf}, True Positive data over {args.conf}%: {true_pos_over_conf}, True Positive data under {args.conf}%: {true_pos_under_conf}\n")
                f.write(f"TN: {true_neg_over_conf+true_neg_under_conf}, True Neagtive data over {args.conf}%: {true_neg_over_conf}, True Neagtive data under {args.conf}%: {true_neg_under_conf}\n")
                f.write(f"FP: {false_pos_over_conf+false_pos_under_conf}, False Positive data over {args.conf}%: {false_pos_over_conf}, False Positive data under {args.conf}%: {false_pos_under_conf}\n")
                f.write(f"FN: {false_neg_over_conf+false_neg_under_conf}, False Neagtive data over {args.conf}%: {false_neg_over_conf}, False Neagtive data under {args.conf}%: {false_neg_under_conf}\n")

            print(f"TP: {true_pos_over_conf+true_pos_under_conf}, True Positive data over {args.conf}%: {true_pos_over_conf}, True Positive data under {args.conf}%: {true_pos_under_conf}")
            print(f"TN: {true_neg_over_conf+true_neg_under_conf}, True Neagtive data over {args.conf}%: {true_neg_over_conf}, True Neagtive data under {args.conf}%: {true_neg_under_conf}")
            print(f"FP: {false_pos_over_conf+false_pos_under_conf}, False Positive data over {args.conf}%: {false_pos_over_conf}, False Positive data under {args.conf}%: {false_pos_under_conf}")
            print(f"FN: {false_neg_over_conf+false_neg_under_conf}, False Neagtive data over {args.conf}%: {false_neg_over_conf}, False Neagtive data under {args.conf}%: {false_neg_under_conf}")

            # get index 
            itn = [i for i in range(len(result)) if (y_pred[i]==y_target[i] and y_pred[i]!=pos_val)]
            itp = [i for i in range(len(result)) if (y_pred[i]==y_target[i] and y_pred[i]==pos_val)]
            ifn = [i for i in range(len(result)) if (y_pred[i]!=y_target[i] and y_pred[i]!=pos_val)]
            ifp = [i for i in range(len(result)) if (y_pred[i]!=y_target[i] and y_pred[i]==pos_val)]
            
            # histogram T-F 
            plt.hist(((conf_TN+conf_TP),(conf_FN+conf_FP)), label=('True', 'False'),histtype='bar', bins=50)
            plt.xlabel('Confidence')
            plt.ylabel('Conunt')
            plt.legend(loc='best')
            plt.title('True: {0}, False: {1}'.format(len(conf_TN+conf_TP),len(conf_FN+conf_FP)))
            # plt.savefig('image/'+args.pred_eval_name+'hist_tf.png')
            plt.savefig(args.pred_eval_name+'hist_tf.png')
            plt.close()
            

            # histogram TN TP FN FP
            plt.hist((conf_TN,conf_TP,conf_FN,conf_FP), label=('TN', 'TP','FN','FP'),histtype='bar', bins=30)
            plt.xlabel('Confidence')
            plt.ylabel('Conunt')
            plt.legend(loc='best')
            plt.title('TN: {0}, TP: {1}, FN: {2}, FP: {3}'.format(len(conf_TN),len(conf_TP),len(conf_FN),len(conf_FP)))
            # plt.savefig('image/'+args.pred_eval_name+'hist_4.png')
            plt.savefig(args.pred_eval_name+'hist_4.png')
            plt.close()
            
            # scatter graph
            if len(conf_TN):
                plt.scatter(conf_TN, itn, alpha=0.4, color='tab:blue', label='TN', s=20)
            if len(conf_TP):
                plt.scatter(conf_TP, itp, alpha=0.4, color='tab:orange', label='TP', s=20)
            if len(conf_FN):
                plt.scatter(conf_FN, ifn, alpha=0.4, color='tab:green', marker='x', label='FN', s=20)
            if len(conf_FP):
                plt.scatter(conf_FP, ifp, alpha=0.4, color='tab:red', marker='x', label='FT', s=20)
            plt.legend(loc='best')
            plt.xlabel('Confidence')
            plt.ylabel('Image Index')
            # plt.savefig('image/'+args.pred_eval_name+'scater.png')
            plt.savefig(args.pred_eval_name+'scater.png')
            plt.close()

            # histogram 
            plt.hist(((conf_TN+conf_TP+conf_FN+conf_FP)), histtype='bar', bins=50)
            plt.xlabel('Confidence')
            plt.ylabel('Conunt')
            # plt.savefig('image/'+args.pred_eval_name+'hist.png')
            plt.savefig(args.pred_eval_name+'hist.png')
            plt.close()

    ##################################### save evalutations #####################################

