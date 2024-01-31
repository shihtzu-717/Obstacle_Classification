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

    # # 텍스트 추가
    # text_top = f"predicted class: {label},  {re[2]:.2f}%"
    # text_bottom = f"target: {re[3]}"
    
    text_top = f"target: {re[3]}"
    text_bottom = f"predicted class: {label},  {re[2]:.2f}%"
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
            cv2.imwrite(os.path.join(true_data_path, 'crop_images', fn), crop_img)
        cv2.imwrite(os.path.join(true_data_path, 'inference', fn), background)
    else:
        false_data_path = str(Path(pred_save_path) / label / 'false_data')
        shutil.copy(image_path, os.path.join(false_data_path, 'images'))
        annot_path = image_path.replace('images', 'annotations')
        annot_path = annot_path.replace('.jpg', '.txt')
        annot_path = annot_path.replace('.png', '.txt')
        if not use_cropimg:
            shutil.copy(annot_path, os.path.join(false_data_path, 'annotations'))
            cv2.imwrite(os.path.join(false_data_path, 'crop_images', fn), crop_img)
        cv2.imwrite(os.path.join(false_data_path, 'inference', fn), background)

def softmax(x):
    exp_x = torch.exp(x - torch.max(x))
    softmax_x = exp_x / torch.sum(exp_x)
    return softmax_x

# def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
#                     data_loader: Iterable, optimizer: torch.optim.Optimizer,
#                     device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
#                     model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
#                     wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
#                     num_training_steps_per_epoch=None, update_freq=None, use_amp=False, use_softlabel=False):
def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
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
            # if use_softlabel:
            #     targets = torch.tensor([0 if i==2 or i==0 else 1 for i in targets]).to(device)
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
# def evaluate(data_loader, model, device, criterion=torch.nn.CrossEntropyLoss(), use_amp=False, use_softlabel=False):
def evaluate(data_loader, model, device, criterion=torch.nn.CrossEntropyLoss(), use_amp=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header): # 학습할 때는 data가 data_loader_val임 
        images = batch[0].to(device, non_blocking=True)
        target = batch[-1].to(device, non_blocking=True)
        # if use_softlabel:
        #     target = torch.tensor([0 if i==2 or i==0 else 1 for i in target]).to(device)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        
        acc1, acc5 = accuracy(output, target, topk=(1, 5)) # top2는 의미 없어 5로 변경

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

        for class_name, class_id in data_loader.dataset.class_to_idx.items():
            # if use_softlabel:
            #     class_id = 0 if class_id==2 or class_id==0 else 1 
            #     class_name = 'negative' if class_name == 'amb_neg' else class_name
            #     class_name = 'positive' if class_name == 'amb_pos' else class_name

            mask = (target == class_id)
            target_class = torch.masked_select(target, mask)
            data_size = target_class.shape[0]
            if data_size > 0:
                mask = mask.unsqueeze(1).expand_as(output)
                output_class = torch.masked_select(output, mask)
                # if use_softlabel:
                #     output_class = output_class.view(-1, 2)
                # else:
                #     output_class = output_class.view(-1, len(data_loader.dataset.class_to_idx))
                output_class = output_class.view(-1, len(data_loader.dataset.class_to_idx))
                acc1_class, acc5_class = accuracy(output_class, target_class, topk=(1, 5)) # top5는 의미 없어 2로 변경
                metric_logger.meters[f'acc1_{class_name}'].update(acc1_class.item(), n=data_size)
                metric_logger.meters[f'acc5_{class_name}'].update(acc5_class.item(), n=data_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@2 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
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

        # # File 이름에 label이 있는지 확인
        spltnm = str(data[1]).split('_')
        # target = int(spltnm[0][1]) if spltnm[0][0] == 't' else -1
        target = -1


        # # label이 따로 있는 경우 아래 4가지 label로 지정
        if target == -1:
            if spltnm[0] == 'etc':
                target = 0
            elif spltnm[0] == 'negative':
                target = 1
            elif spltnm[0] == 'paper':
                target = 2
            elif spltnm[0] == 'plastic':
                target = 3
            elif spltnm[0] == 'roadkill':
                target = 4
            elif spltnm[0] == 'stone':
                target = 5
            elif spltnm[0] == 'tire':
                target = 6
            elif spltnm[0] == 'vinyl':
                target = 7
            elif spltnm[0] == 'wood':
                target = 8
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
        probs = softmax(output_tensor) # softmax 통과

        probs_max = ((torch.max(probs)).detach().cpu().numpy())*100
        result.append((pred, probs_max, target, data[0] / data.image_path, data.label, data.bbox, str(idx)))
        idx += 1
        
    ##################################### save result image & anno #####################################

    if args.pred_save:
        import os
        for i in ['etc', 'paper', 'plastic', 'roadkill', 'stone', 'tire', 'vinyl', 'wood', 'negative']:
            os.makedirs(Path(args.pred_save_path) / i / 'images', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) / i / 'annotations', exist_ok=True)

        if float(args.conf) > 0:
            os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'images', exist_ok=True)
            os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'annotations', exist_ok=True)
        
        if args.pred_save_with_conf:
            for i in ['etc', 'paper', 'plastic', 'roadkill', 'stone', 'tire', 'vinyl', 'wood', 'negative']:
                os.makedirs(Path(args.pred_save_path) / i / 'inference', exist_ok=True)
                for j in ['true_data', 'false_data']:
                    os.makedirs(Path(args.pred_save_path) / i / j / 'images', exist_ok=True)
                    os.makedirs(Path(args.pred_save_path) / i / j / 'annotations', exist_ok=True)
                    os.makedirs(Path(args.pred_save_path) / i / j / 'inference', exist_ok=True)
            
            if float(args.conf) > 0:
                os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'inference', exist_ok=True)
                for i in ['true_data', 'false_data']:
                    os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / i / 'images', exist_ok=True)
                    os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / i / 'annotations', exist_ok=True)
                    os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / i / 'inference', exist_ok=True)

            if not args.use_cropimg:
                for i in ['etc', 'paper', 'plastic', 'roadkill', 'stone', 'tire', 'vinyl', 'wood', 'negative']:
                    os.makedirs(Path(args.pred_save_path) / i / 'true_data' / 'crop_images', exist_ok=True)
                    os.makedirs(Path(args.pred_save_path) / i / 'false_data' / 'crop_images', exist_ok=True)
                if float(args.conf) > 0:
                    os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'true_data' / 'crop_images', exist_ok=True)
                    os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'false_data' / 'crop_images', exist_ok=True)

        etc = []
        paper = []
        plastic = []
        roadkill = []
        stone = []
        tire = []
        vinyl = []
        wood = []
        negative = []
        pos_2_neg = []
        for x in result:
            if x[0] == 0:
                etc.append((x[3], 'etc', x[1], x[4], x[5], x[6]))
                if x[1] >= args.conf:
                    pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
            elif x[0] == 1:
                negative.append((x[3], 'negative', x[1], x[4], x[5], x[6]))
                if x[1] >= args.conf:
                    pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
            elif x[0] == 2:
                paper.append((x[3], 'paper', x[1], x[4], x[5], x[6]))
                if x[1] >= args.conf:
                    pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
            elif x[0] == 3:
                plastic.append((x[3], 'plastic', x[1], x[4], x[5], x[6]))
                if x[1] >= args.conf:
                    pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
            elif x[0] == 4:
                roadkill.append((x[3], 'roadkill', x[1], x[4], x[5], x[6]))
                if x[1] >= args.conf:
                    pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
            elif x[0] == 5:
                stone.append((x[3], 'stone', x[1], x[4], x[5], x[6]))
                if x[1] >= args.conf:
                    pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
            elif x[0] == 6:
                tire.append((x[3], 'tire', x[1], x[4], x[5], x[6]))
                if x[1] >= args.conf:
                    pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
            elif x[0] == 7:
                vinyl.append((x[3], 'vinyl', x[1], x[4], x[5], x[6]))
                if x[1] >= args.conf:
                    pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
            elif x[0] == 8:
                wood.append((x[3], 'wood', x[1], x[4], x[5], x[6]))
                if x[1] >= args.conf:
                    pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
        
        with open(Path(args.pred_save_path)/"conf_avg.txt", 'w') as f:
            f.write('')

        # ### save_crop ###
        # if not os.path.exists(Path(args.pred_save_path) / 'negative' / 'crop_images'):
        #     os.mkdir(os.path.join(args.pred_save_path, 'negative', 'crop_images'))
        # if not os.path.exists(Path(args.pred_save_path) / 'positive' / 'crop_images'):
        #     os.mkdir(os.path.join(args.pred_save_path, 'positive', 'crop_images'))
        # ### save_crop ###

        # 'etc', 'negative', 'paper', 'plastic', 'roadkill', 'stone', 'tire', 'vinyl', 'wood'
        cls_0_sum = 0
        cls_0_min = 100.0
        cls_0_max = 0
        for cls_0 in tqdm(etc, desc='Class_0 images copying... '):
            img_path = str(cls_0[0])
            cls_0_conf = float(cls_0[2])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(cls_0[0], Path(args.pred_save_path) / 'etc' / 'images')

            if not args.use_cropimg:
                shutil.copy(annot_path, Path(args.pred_save_path) / 'etc' / 'annotations')

            if args.pred_save_with_conf:
                create_images_with_conf(img_path, cls_0, 'etc', args.pred_save_path, args.use_cropimg, args)
                    
            cls_0_sum = cls_0_sum + cls_0_conf
            if cls_0_max < cls_0_conf:
                cls_0_max = cls_0_conf
            if cls_0_min > cls_0_conf:
                cls_0_min = cls_0_conf
        try:
            print(f"Class_0 AVG: {cls_0_sum / len(etc):.2f}%")
            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write(f"Class_0 CNT: {len(etc)}, ")
                f.write(f"Class_0 AVG: {cls_0_sum / len(etc):.2f}%, ")
                f.write(f"Class_0 MAX: {cls_0_max:.2f}%, ")
                f.write(f"Class_0 MIN: {cls_0_min:.2f}%\n")
        except ZeroDivisionError:
            print("No Class_0 Data")

        cls_1_sum = 0
        cls_1_min = 100.0
        cls_1_max = 0
        for cls_1 in tqdm(negative, desc='Class_1 images copying... '):
            img_path = str(cls_1[0])
            cls_1_conf = float(cls_1[2])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(cls_1[0], Path(args.pred_save_path) / 'negative' / 'images')

            if not args.use_cropimg:
                shutil.copy(annot_path, Path(args.pred_save_path) / 'negative' / 'annotations')

            if args.pred_save_with_conf:
                create_images_with_conf(img_path, cls_1, 'negative', args.pred_save_path, args.use_cropimg, args)
                    
            cls_1_sum = cls_1_sum + cls_1_conf
            if cls_1_max < cls_1_conf:
                cls_1_max = cls_1_conf
            if cls_1_min > cls_1_conf:
                cls_1_min = cls_1_conf
        try:
            print(f"Class_1 AVG: {cls_1_sum / len(negative):.2f}%")
            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write(f"Class_1 CNT: {len(negative)}, ")
                f.write(f"Class_1 AVG: {cls_1_sum / len(negative):.2f}%, ")
                f.write(f"Class_1 MAX: {cls_1_max:.2f}%, ")
                f.write(f"Class_1 MIN: {cls_1_min:.2f}%\n")
        except ZeroDivisionError:
            print("No Class_1 Data")
        
        cls_2_sum = 0
        cls_2_min = 100.0
        cls_2_max = 0
        for cls_2 in tqdm(paper, desc='Class_2 images copying... '):
            img_path = str(cls_2[0])
            cls_2_conf = float(cls_2[2])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(cls_2[0], Path(args.pred_save_path) / 'paper' / 'images')

            if not args.use_cropimg:
                shutil.copy(annot_path, Path(args.pred_save_path) / 'paper' / 'annotations')

            if args.pred_save_with_conf:
                create_images_with_conf(img_path, cls_2, 'paper', args.pred_save_path, args.use_cropimg, args)
                    
            cls_2_sum = cls_2_sum + cls_2_conf
            if cls_2_max < cls_2_conf:
                cls_2_max = cls_2_conf
            if cls_2_min > cls_2_conf:
                cls_2_min = cls_2_conf
        try:
            print(f"Class_2 AVG: {cls_2_sum / len(paper):.2f}%")
            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write(f"Class_2 CNT: {len(paper)}, ")
                f.write(f"Class_2 AVG: {cls_2_sum / len(paper):.2f}%, ")
                f.write(f"Class_2 MAX: {cls_2_max:.2f}%, ")
                f.write(f"Class_2 MIN: {cls_2_min:.2f}%\n")
        except ZeroDivisionError:
            print("No Class_2 Data")
        
        cls_3_sum = 0
        cls_3_min = 100.0
        cls_3_max = 0
        for cls_3 in tqdm(plastic, desc='Class_3 images copying... '):
            img_path = str(cls_3[0])
            cls_3_conf = float(cls_3[2])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(cls_3[0], Path(args.pred_save_path) / 'plastic' / 'images')

            if not args.use_cropimg:
                shutil.copy(annot_path, Path(args.pred_save_path) / 'plastic' / 'annotations')

            if args.pred_save_with_conf:
                create_images_with_conf(img_path, cls_3, 'plastic', args.pred_save_path, args.use_cropimg, args)
                    
            cls_3_sum = cls_3_sum + cls_3_conf
            if cls_3_max < cls_3_conf:
                cls_3_max = cls_3_conf
            if cls_3_min > cls_3_conf:
                cls_3_min = cls_3_conf
        try:
            print(f"Class_3 AVG: {cls_3_sum / len(plastic):.2f}%")
            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write(f"Class_3 CNT: {len(plastic)}, ")
                f.write(f"Class_3 AVG: {cls_3_sum / len(plastic):.2f}%, ")
                f.write(f"Class_3 MAX: {cls_3_max:.2f}%, ")
                f.write(f"Class_3 MIN: {cls_3_min:.2f}%\n")
        except ZeroDivisionError:
            print("No Class_3 Data")

        cls_4_sum = 0
        cls_4_min = 100.0
        cls_4_max = 0
        for cls_4 in tqdm(roadkill, desc='Class_4 images copying... '):
            img_path = str(cls_4[0])
            cls_4_conf = float(cls_4[2])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(cls_4[0], Path(args.pred_save_path) / 'roadkill' / 'images')

            if not args.use_cropimg:
                shutil.copy(annot_path, Path(args.pred_save_path) / 'roadkill' / 'annotations')

            if args.pred_save_with_conf:
                create_images_with_conf(img_path, cls_4, 'roadkill', args.pred_save_path, args.use_cropimg, args)
                    
            cls_4_sum = cls_4_sum + cls_4_conf
            if cls_4_max < cls_4_conf:
                cls_4_max = cls_4_conf
            if cls_4_min > cls_4_conf:
                cls_4_min = cls_4_conf
        try:
            print(f"Class_4 AVG: {cls_4_sum / len(roadkill):.2f}%")
            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write(f"Class_4 CNT: {len(roadkill)}, ")
                f.write(f"Class_4 AVG: {cls_4_sum / len(roadkill):.2f}%, ")
                f.write(f"Class_4 MAX: {cls_4_max:.2f}%, ")
                f.write(f"Class_4 MIN: {cls_4_min:.2f}%\n")
        except ZeroDivisionError:
            print("No Class_4 Data")
 
        cls_5_sum = 0
        cls_5_min = 100.0
        cls_5_max = 0
        for cls_5 in tqdm(stone, desc='Class_5 images copying... '):
            img_path = str(cls_5[0])
            cls_5_conf = float(cls_5[2])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(cls_5[0], Path(args.pred_save_path) / 'stone' / 'images')

            if not args.use_cropimg:
                shutil.copy(annot_path, Path(args.pred_save_path) / 'stone' / 'annotations')

            if args.pred_save_with_conf:
                create_images_with_conf(img_path, cls_5, 'stone', args.pred_save_path, args.use_cropimg, args)
                    
            cls_5_sum = cls_5_sum + cls_5_conf
            if cls_5_max < cls_5_conf:
                cls_5_max = cls_5_conf
            if cls_5_min > cls_5_conf:
                cls_5_min = cls_5_conf
        try:
            print(f"Class_5 AVG: {cls_5_sum / len(stone):.2f}%")
            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write(f"Class_5 CNT: {len(stone)}, ")
                f.write(f"Class_5 AVG: {cls_5_sum / len(stone):.2f}%, ")
                f.write(f"Class_5 MAX: {cls_5_max:.2f}%, ")
                f.write(f"Class_5 MIN: {cls_5_min:.2f}%\n")
        except ZeroDivisionError:
            print("No Class_5 Data")


        cls_6_sum = 0
        cls_6_min = 100.0
        cls_6_max = 0
        for cls_6 in tqdm(tire, desc='Class_6 images copying... '):
            img_path = str(cls_6[0])
            cls_6_conf = float(cls_6[2])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(cls_6[0], Path(args.pred_save_path) / 'tire' / 'images')

            if not args.use_cropimg:
                shutil.copy(annot_path, Path(args.pred_save_path) / 'tire' / 'annotations')

            if args.pred_save_with_conf:
                create_images_with_conf(img_path, cls_6, 'tire', args.pred_save_path, args.use_cropimg, args)
                    
            cls_6_sum = cls_6_sum + cls_6_conf
            if cls_6_max < cls_6_conf:
                cls_6_max = cls_6_conf
            if cls_6_min > cls_6_conf:
                cls_6_min = cls_6_conf
        try:
            print(f"Class_6 AVG: {cls_6_sum / len(tire):.2f}%")
            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write(f"Class_6 CNT: {len(tire)}, ")
                f.write(f"Class_6 AVG: {cls_6_sum / len(tire):.2f}%, ")
                f.write(f"Class_6 MAX: {cls_6_max:.2f}%, ")
                f.write(f"Class_6 MIN: {cls_6_min:.2f}%\n")
        except ZeroDivisionError:
            print("No Class_6 Data")

        
        cls_7_sum = 0
        cls_7_min = 100.0
        cls_7_max = 0
        for cls_7 in tqdm(vinyl, desc='Class_7 images copying... '):
            img_path = str(cls_7[0])
            cls_7_conf = float(cls_7[2])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(cls_7[0], Path(args.pred_save_path) / 'vinyl' / 'images')

            if not args.use_cropimg:
                shutil.copy(annot_path, Path(args.pred_save_path) / 'vinyl' / 'annotations')

            if args.pred_save_with_conf:
                create_images_with_conf(img_path, cls_7, 'vinyl', args.pred_save_path, args.use_cropimg, args)
                    
            cls_7_sum = cls_7_sum + cls_7_conf
            if cls_7_max < cls_7_conf:
                cls_7_max = cls_7_conf
            if cls_7_min > cls_7_conf:
                cls_7_min = cls_7_conf
        try:
            print(f"Class_7 AVG: {cls_7_sum / len(vinyl):.2f}%")
            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write(f"Class_7 CNT: {len(vinyl)}, ")
                f.write(f"Class_7 AVG: {cls_7_sum / len(vinyl):.2f}%, ")
                f.write(f"Class_7 MAX: {cls_7_max:.2f}%, ")
                f.write(f"Class_7 MIN: {cls_7_min:.2f}%\n")
        except ZeroDivisionError:
            print("No Class_7 Data")

        cls_8_sum = 0
        cls_8_min = 100.0
        cls_8_max = 0
        for cls_8 in tqdm(wood, desc='Class_8 images copying... '):
            img_path = str(cls_8[0])
            cls_8_conf = float(cls_8[2])
            annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
            shutil.copy(cls_8[0], Path(args.pred_save_path) / 'wood' / 'images')

            if not args.use_cropimg:
                shutil.copy(annot_path, Path(args.pred_save_path) / 'wood' / 'annotations')

            if args.pred_save_with_conf:
                create_images_with_conf(img_path, cls_8, 'wood', args.pred_save_path, args.use_cropimg, args)
                    
            cls_8_sum = cls_8_sum + cls_8_conf
            if cls_8_max < cls_8_conf:
                cls_8_max = cls_8_conf
            if cls_8_min > cls_8_conf:
                cls_8_min = cls_8_conf
        try:
            print(f"Class_8 AVG: {cls_8_sum / len(wood):.2f}%")
            with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                f.write(f"Class_8 CNT: {len(wood)}, ")
                f.write(f"Class_8 AVG: {cls_8_sum / len(wood):.2f}%, ")
                f.write(f"Class_8 MAX: {cls_8_max:.2f}%, ")
                f.write(f"Class_8 MIN: {cls_8_min:.2f}%\n")
        except ZeroDivisionError:
            print("No Class_8 Data")

        if len(pos_2_neg) == 0:
            for pn in tqdm(pos_2_neg, desc='Change to negative data due to low confidence level'):
                img_path = str(pn[0])
                pn_conf = float(pn[2])
                annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                shutil.copy(pn[0], Path(args.pred_save_path) / 'pos_2_neg' / 'images')
                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / 'pos_2_neg' / 'annotations')
                if args.pred_save_with_conf:
                    create_images_with_conf(img_path, pn, 'pos_2_neg', args.pred_save_path, args.use_cropimg, args)

        
    ##################################### save result image & anno #####################################

    ##################################### save evalutations #####################################
    if args.pred_eval:
        print(np.array(result)[...,2].shape)
        if np.sum(np.array(result)[...,2]) < 0:
            conf_TN = [x[1] for x in result if (x[0]==0)]
            conf_TP = [x[1] for x in result if (x[0]==1)]
            conf_FN = []
            conf_FP = []

            # index set    
            itn = [i for i in range(len(result)) if (result[i][0]==0)]
            itp = [i for i in range(len(result)) if (result[i][0]==1)]

            # histogram P-N 
            plt.hist((conf_TN, conf_TP), label=('etc', 'paper', 'plastic', 'roadkill', 'stone', 'tire', 'vinyl', 'wood', 'negative'), histtype='bar', bins=50)
            plt.xlabel('Confidence')
            plt.ylabel('Conunt')
            plt.legend(loc='upper left')
            plt.savefig(args.pred_eval_name+'hist_PN.png')
            plt.close()

        else:
            y_pred = [i[0] for i in result]
            y_target = [i[2] for i in result]
            # y_pred = []
            # y_target = []
            # for i in result:
            #     if args.conf >= i[1] and i[0] != 8: # conf가 특정 값 이하이고 neg가 아닌 경우
            #         y_pred.append(8) # negative
            #         y_target.append(i[2])
            #     else:
            #         y_pred.append(i[0])
            #         y_target.append(i[2])     
            neg_val = 1

            # precision recall 계산
            precision = precision_score(y_target, y_pred, average= "macro")
            recall = recall_score(y_target, y_pred, average= "macro")
            cm = confusion_matrix(y_target, y_pred)
            cm_display = ConfusionMatrixDisplay(cm).plot()
            try:
                cls_report = classification_report(y_target, y_pred, target_names=['etc', 'paper', 'plastic', 'roadkill', 'stone', 'tire', 'vinyl', 'wood', 'negative'])
            except:
                pass
                # cls_report = classification_report(y_target, y_pred, target_names=["Class_0"])

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
            conf_TN = [x[1] for p, t, x in zip(y_pred, y_target, result) if p==t and p==neg_val] 
            conf_TP = [x[1] for p, t, x in zip(y_pred, y_target, result) if p==t and p!=neg_val] 
            conf_FN = [x[1] for p, t, x in zip(y_pred, y_target, result) if p!=t and p==neg_val] 
            conf_FP = [x[1] for p, t, x in zip(y_pred, y_target, result) if p!=t and p!=neg_val] 
            
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
            itn = [i for i in range(len(result)) if (y_pred[i]==y_target[i] and y_pred[i]==neg_val)]
            itp = [i for i in range(len(result)) if (y_pred[i]==y_target[i] and y_pred[i]!=neg_val)]
            ifn = [i for i in range(len(result)) if (y_pred[i]!=y_target[i] and y_pred[i]==neg_val)]
            ifp = [i for i in range(len(result)) if (y_pred[i]!=y_target[i] and y_pred[i]!=neg_val)]
            
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

@torch.no_grad()
def prediction_several_models(args, device):
    import os
    import sys
    import random
    from preprocess_data import make_dataset_file
    from datasets import PotholeDataset, get_split_data
    from sklearn.metrics import precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay, classification_report

    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    totorch = transforms.ToTensor()

    output_dir_path = args.pred_save_path
    graph_save_dir = args.pred_eval_name
    
    load_model_list = []
    result_dict = {}
    
    
    for ckpt in args.resume:
        # nb_cls = 9
        print(f'Model File: {ckpt}')
        name = ckpt.split('/')[-2]+ '_'
        ops = Path(ckpt).parts[-2]
        opsdict = dict(zip([str(d) for i,d in enumerate(str(ops).split('_')) if i%2==0], 
                       [str(d) for i, d in enumerate(ops.split('_')) if i%2==1]))
        if not os.path.exists(output_dir_path):
            os.makedirs(Path(output_dir_path), exist_ok=True)

        if not os.path.exists(graph_save_dir):
            os.makedirs(Path(graph_save_dir), exist_ok=True)
        
        args.pred_eval_name = os.path.join(graph_save_dir, name)
        args.pred_save_path = os.path.join(output_dir_path, name)
        args.nb_classes = int(opsdict['nbclss'])
        args.padding = opsdict['pad']
        args.padding_size = float(opsdict['padsize'])
        args.use_bbox = True if opsdict['box'] == 'True' else False
        args.use_shift = True if opsdict['shift'] == 'True' else False

        # print(args.resume)
        # print(args.padding, args.padding_size, args.use_bbox)
        # print(args.pred_eval_name, args.pred_save_path)
        # print(args.nb_classes, args.use_shift)

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
        utils.auto_load_model_with_model(
            ckpt, args=args, model=model, model_without_ddp=model,
            optimizer=None, loss_scaler=None, model_ema=None)
        model.eval()

        load_model_list.append((ckpt, model))
        result_dict[ckpt] = []
    
    # Data laod     
    data_list = []
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

        # # File 이름에 label이 있는지 확인
        spltnm = str(data[1]).split('_')
        # target = int(spltnm[0][1]) if spltnm[0][0] == 't' else -1
        target = -1
        

        # # label이 따로 있는 경우 아래 4가지 label로 지정
        if target == -1:
            if spltnm[0] == '0_etc':
                target = 0
            elif spltnm[0] == '1_paper':
                target = 1
            elif spltnm[0] == '2_plastic+can+glass':
                target = 2
            elif spltnm[0] == '3_stone':
                target = 3
            elif spltnm[0] == '4_tire':
                target = 4
            elif spltnm[0] == '5_plastic_bag':
                target = 5
            elif spltnm[0] == '6_wood':
                target = 6
            elif spltnm[0] == '7_natual':
                target = 7
            elif spltnm[0] == '8_negative':
                target = 8
            else:
                target =-1

        crop_img = cv2.resize(crop_img, (args.input_size, args.input_size))
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        pil_image=Image.fromarray(crop_img)
        input_tensor = totorch(pil_image).to(device)
        input_tensor = input_tensor.unsqueeze(dim=0)
        input_tensor = tonorm(input_tensor)

        for model_name, model_weights in load_model_list:
            # model output 
            output_tensor = model_weights(input_tensor) 
            pred, conf = int(torch.argmax(output_tensor).detach().cpu().numpy()), float((torch.max(output_tensor)).detach().cpu().numpy())
            probs = softmax(output_tensor) # softmax 통과
            probs_max = ((torch.max(probs)).detach().cpu().numpy())*100

            result_dict[model_name].append((pred, probs_max, target, data[0] / data.image_path, data.label, data.bbox, str(idx)))
            idx += 1

    
    # ##################################### save result image & anno #####################################
    for result_k, result_v in result_dict.items():
        name = result_k.split('/')[-2]+ '_'
        ops = Path(result_k).parts[-2]
        opsdict = dict(zip([str(d) for i,d in enumerate(str(ops).split('_')) if i%2==0], 
                       [str(d) for i, d in enumerate(ops.split('_')) if i%2==1]))

        if not os.path.exists(output_dir_path):
            os.makedirs(Path(output_dir_path), exist_ok=True)

        if not os.path.exists(graph_save_dir):
            os.makedirs(Path(graph_save_dir), exist_ok=True)
        
        args.pred_eval_name = os.path.join(graph_save_dir, name)
        args.pred_save_path = os.path.join(output_dir_path, name)
        if args.pred_save:
            import os
            for i in ['0_etc', '1_paper', '2_plastic+can+glass', '3_stone', '4_tire', '5_plastic_bag', '6_wood', '7_natual', '8_negative']:
                os.makedirs(Path(args.pred_save_path) / i / 'images', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) / i / 'annotations', exist_ok=True)

            if float(args.conf) > 0:
                os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'images', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'annotations', exist_ok=True)
        
            if args.pred_save_with_conf:
                for i in ['0_etc', '1_paper', '2_plastic+can+glass', '3_stone', '4_tire', '5_plastic_bag', '6_wood', '7_natual', '8_negative']:
                    os.makedirs(Path(args.pred_save_path) / i / 'inference', exist_ok=True)
                    for j in ['true_data', 'false_data']:
                        os.makedirs(Path(args.pred_save_path) / i / j / 'images', exist_ok=True)
                        os.makedirs(Path(args.pred_save_path) / i / j / 'annotations', exist_ok=True)
                        os.makedirs(Path(args.pred_save_path) / i / j / 'inference', exist_ok=True)
            
                if float(args.conf) > 0:
                    os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'inference', exist_ok=True)
                    for i in ['true_data', 'false_data']:
                        os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / i / 'images', exist_ok=True)
                        os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / i / 'annotations', exist_ok=True)
                        os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / i / 'inference', exist_ok=True)

                if not args.use_cropimg:
                    for i in ['0_etc', '1_paper', '2_plastic+can+glass', '3_stone', '4_tire', '5_plastic_bag', '6_wood', '7_natual', '8_negative']:
                        os.makedirs(Path(args.pred_save_path) / i / 'true_data' / 'crop_images', exist_ok=True)
                        os.makedirs(Path(args.pred_save_path) / i / 'false_data' / 'crop_images', exist_ok=True)
                    if float(args.conf) > 0:
                        os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'true_data' / 'crop_images', exist_ok=True)
                        os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'false_data' / 'crop_images', exist_ok=True)

            etc = []
            negative = []
            paper = []
            plastic = []
            natual = []
            stone = []
            tire = []
            plastic_bag = []
            wood = []
            pos_2_neg = []
            for x in result_v:
                if x[0] == 0:
                    etc.append((x[3], '0_etc', x[1], x[4], x[5], x[6]))
                    if x[1] >= args.conf:
                        pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
                elif x[0] == 1:
                    paper.append((x[3], '1_paper', x[1], x[4], x[5], x[6]))
                    if x[1] >= args.conf:
                        pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
                elif x[0] == 2:
                    plastic.append((x[3], '2_plastic+can+glass', x[1], x[4], x[5], x[6]))
                    if x[1] >= args.conf:
                        pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
                elif x[0] == 3:
                    stone.append((x[3], '3_stone', x[1], x[4], x[5], x[6]))
                    if x[1] >= args.conf:
                        pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
                elif x[0] == 4:
                    tire.append((x[3], '4_tire', x[1], x[4], x[5], x[6]))
                    if x[1] >= args.conf:
                        pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
                elif x[0] == 5:
                    plastic_bag.append((x[3], '5_plastic_bag', x[1], x[4], x[5], x[6]))
                    if x[1] >= args.conf:
                        pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
                elif x[0] == 6:
                    wood.append((x[3], '6_wood', x[1], x[4], x[5], x[6]))
                    if x[1] >= args.conf:
                        pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
                elif x[0] == 7:
                    natual.append((x[3], '7_natual', x[1], x[4], x[5], x[6]))
                    if x[1] >= args.conf:
                        pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
                elif x[0] == 8:
                    negative.append((x[3], '8_negative', x[1], x[4], x[5], x[6]))
                    if x[1] >= args.conf:
                        pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))

            with open(Path(args.pred_save_path)/"conf_avg.txt", 'w') as f:
                f.write('')

            # ### save_crop ###
            # if not os.path.exists(Path(args.pred_save_path) / 'negative' / 'crop_images'):
            #     os.mkdir(os.path.join(args.pred_save_path, 'negative', 'crop_images'))
            # if not os.path.exists(Path(args.pred_save_path) / 'positive' / 'crop_images'):
            #     os.mkdir(os.path.join(args.pred_save_path, 'positive', 'crop_images'))
            # ### save_crop ###

            cls_0_sum = 0
            cls_0_min = 100.0
            cls_0_max = 0
            for cls_0 in tqdm(etc, desc='Class_0 images copying... '):
                img_path = str(cls_0[0])
                cls_0_conf = float(cls_0[2])
                annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                shutil.copy(cls_0[0], Path(args.pred_save_path) / '0_etc' / 'images')

                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / '0_etc' / 'annotations')

                if args.pred_save_with_conf:
                    create_images_with_conf(img_path, cls_0, '0_etc', args.pred_save_path, args.use_cropimg, args)
                    
                cls_0_sum = cls_0_sum + cls_0_conf
                if cls_0_max < cls_0_conf:
                    cls_0_max = cls_0_conf
                if cls_0_min > cls_0_conf:
                    cls_0_min = cls_0_conf
            try:
                print(f"Class_0 AVG: {cls_0_sum / len(etc):.2f}%")
                with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                    f.write(f"Class_0 CNT: {len(etc)}, ")
                    f.write(f"Class_0 AVG: {cls_0_sum / len(etc):.2f}%, ")
                    f.write(f"Class_0 MAX: {cls_0_max:.2f}%, ")
                    f.write(f"Class_0 MIN: {cls_0_min:.2f}%\n")
            except ZeroDivisionError:
                print("No Class_0 Data")

            cls_1_sum = 0
            cls_1_min = 100.0
            cls_1_max = 0
            for cls_1 in tqdm(paper, desc='Class_1 images copying... '):
                img_path = str(cls_1[0])
                cls_1_conf = float(cls_1[2])
                annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                shutil.copy(cls_1[0], Path(args.pred_save_path) / '1_paper' / 'images')

                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / '1_paper' / 'annotations')

                if args.pred_save_with_conf:
                    create_images_with_conf(img_path, cls_1, '1_paper', args.pred_save_path, args.use_cropimg, args)
                    
                cls_1_sum = cls_1_sum + cls_1_conf
                if cls_1_max < cls_1_conf:
                    cls_1_max = cls_1_conf
                if cls_1_min > cls_1_conf:
                    cls_1_min = cls_1_conf
            try:
                print(f"Class_1 AVG: {cls_1_sum / len(paper):.2f}%")
                with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                    f.write(f"Class_1 CNT: {len(paper)}, ")
                    f.write(f"Class_1 AVG: {cls_1_sum / len(paper):.2f}%, ")
                    f.write(f"Class_1 MAX: {cls_1_max:.2f}%, ")
                    f.write(f"Class_1 MIN: {cls_1_min:.2f}%\n")
            except ZeroDivisionError:
                print("No Class_1 Data")


            cls_2_sum = 0
            cls_2_min = 100.0
            cls_2_max = 0
            for cls_2 in tqdm(plastic, desc='Class_2 images copying... '):
                img_path = str(cls_2[0])
                cls_2_conf = float(cls_2[2])
                annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                shutil.copy(cls_2[0], Path(args.pred_save_path) / '2_plastic+can+glass' / 'images')
                
                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / '2_plastic+can+glass' / 'annotations')

                if args.pred_save_with_conf:
                    create_images_with_conf(img_path, cls_2, '2_plastic+can+glass', args.pred_save_path, args.use_cropimg, args)
                    
                cls_2_sum = cls_2_sum + cls_2_conf
                if cls_2_max < cls_2_conf:
                    cls_2_max = cls_2_conf
                if cls_2_min > cls_2_conf:
                    cls_2_min = cls_2_conf
            try:
                print(f"Class_2 AVG: {cls_2_sum / len(plastic):.2f}%")
                with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                    f.write(f"Class_2 CNT: {len(plastic)}, ")
                    f.write(f"Class_2 AVG: {cls_2_sum / len(plastic):.2f}%, ")
                    f.write(f"Class_2 MAX: {cls_2_max:.2f}%, ")
                    f.write(f"Class_2 MIN: {cls_2_min:.2f}%\n")
            except ZeroDivisionError:
                print("No Class_2 Data")

            cls_3_sum = 0
            cls_3_min = 100.0
            cls_3_max = 0
            for cls_3 in tqdm(stone, desc='Class_3 images copying... '):
                img_path = str(cls_3[0])
                cls_3_conf = float(cls_3[2])
                annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                shutil.copy(cls_3[0], Path(args.pred_save_path) / '3_stone' / 'images')

                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / '3_stone' / 'annotations')

                if args.pred_save_with_conf:
                    create_images_with_conf(img_path, cls_3, '3_stone', args.pred_save_path, args.use_cropimg, args)
                    
                cls_3_sum = cls_3_sum + cls_3_conf
                if cls_3_max < cls_3_conf:
                    cls_3_max = cls_3_conf
                if cls_3_min > cls_3_conf:
                    cls_3_min = cls_3_conf
            try:
                print(f"Class_3 AVG: {cls_3_sum / len(stone):.2f}%")
                with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                    f.write(f"Class_3 CNT: {len(stone)}, ")
                    f.write(f"Class_3 AVG: {cls_3_sum / len(stone):.2f}%, ")
                    f.write(f"Class_3 MAX: {cls_3_max:.2f}%, ")
                    f.write(f"Class_3 MIN: {cls_3_min:.2f}%\n")
            except ZeroDivisionError:
                print("No Class_3 Data")
            
            cls_4_sum = 0
            cls_4_min = 100.0
            cls_4_max = 0
            for cls_4 in tqdm(tire, desc='Class_4 images copying... '):
                img_path = str(cls_4[0])
                cls_4_conf = float(cls_4[2])
                annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                shutil.copy(cls_4[0], Path(args.pred_save_path) / '4_tire' / 'images')

                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / '4_tire' / 'annotations')

                if args.pred_save_with_conf:
                    create_images_with_conf(img_path, cls_4, '4_tire', args.pred_save_path, args.use_cropimg, args)
                    
                cls_4_sum = cls_4_sum + cls_4_conf
                if cls_4_max < cls_4_conf:
                    cls_4_max = cls_4_conf
                if cls_4_min > cls_4_conf:
                    cls_4_min = cls_4_conf
            try:
                print(f"Class_4 AVG: {cls_4_sum / len(tire):.2f}%")
                with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                    f.write(f"Class_4 CNT: {len(tire)}, ")
                    f.write(f"Class_4 AVG: {cls_4_sum / len(tire):.2f}%, ")
                    f.write(f"Class_4 MAX: {cls_4_max:.2f}%, ")
                    f.write(f"Class_4 MIN: {cls_4_min:.2f}%\n")
            except ZeroDivisionError:
                print("No Class_4 Data")

            cls_5_sum = 0
            cls_5_min = 100.0
            cls_5_max = 0
            for cls_5 in tqdm(plastic_bag, desc='Class_5 images copying... '):
                img_path = str(cls_5[0])
                cls_5_conf = float(cls_5[2])
                annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                shutil.copy(cls_5[0], Path(args.pred_save_path) / '5_plastic_bag' / 'images')

                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / '5_plastic_bag' / 'annotations')

                if args.pred_save_with_conf:
                    create_images_with_conf(img_path, cls_5, '5_plastic_bag', args.pred_save_path, args.use_cropimg, args)
                    
                cls_5_sum = cls_5_sum + cls_5_conf
                if cls_5_max < cls_5_conf:
                    cls_5_max = cls_5_conf
                if cls_5_min > cls_5_conf:
                    cls_5_min = cls_5_conf
            try:
                print(f"Class_5 AVG: {cls_5_sum / len(plastic_bag):.2f}%")
                with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                    f.write(f"Class_5 CNT: {len(plastic_bag)}, ")
                    f.write(f"Class_5 AVG: {cls_5_sum / len(plastic_bag):.2f}%, ")
                    f.write(f"Class_5 MAX: {cls_5_max:.2f}%, ")
                    f.write(f"Class_5 MIN: {cls_5_min:.2f}%\n")
            except ZeroDivisionError:
                print("No Class_5 Data")

            cls_6_sum = 0
            cls_6_min = 100.0
            cls_6_max = 0
            for cls_6 in tqdm(wood, desc='Class_6 images copying... '):
                img_path = str(cls_6[0])
                cls_6_conf = float(cls_6[2])
                annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                shutil.copy(cls_6[0], Path(args.pred_save_path) / '6_wood' / 'images')

                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / '6_wood' / 'annotations')

                if args.pred_save_with_conf:
                    create_images_with_conf(img_path, cls_6, '6_wood', args.pred_save_path, args.use_cropimg, args)
                    
                cls_6_sum = cls_6_sum + cls_6_conf
                if cls_6_max < cls_6_conf:
                    cls_6_max = cls_6_conf
                if cls_6_min > cls_6_conf:
                    cls_6_min = cls_6_conf
            try:
                print(f"Class_6 AVG: {cls_6_sum / len(wood):.2f}%")
                with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                    f.write(f"Class_6 CNT: {len(wood)}, ")
                    f.write(f"Class_6 AVG: {cls_6_sum / len(wood):.2f}%, ")
                    f.write(f"Class_6 MAX: {cls_6_max:.2f}%, ")
                    f.write(f"Class_6 MIN: {cls_6_min:.2f}%\n")
            except ZeroDivisionError:
                print("No Class_6 Data")

        
            cls_7_sum = 0
            cls_7_min = 100.0
            cls_7_max = 0
            for cls_7 in tqdm(natual, desc='Class_7 images copying... '):
                img_path = str(cls_7[0])
                cls_7_conf = float(cls_7[2])
                annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                shutil.copy(cls_7[0], Path(args.pred_save_path) / '7_natual' / 'images')

                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / '7_natual' / 'annotations')

                if args.pred_save_with_conf:
                    create_images_with_conf(img_path, cls_7, '7_natual', args.pred_save_path, args.use_cropimg, args)
                    
                cls_7_sum = cls_7_sum + cls_7_conf
                if cls_7_max < cls_7_conf:
                    cls_7_max = cls_7_conf
                if cls_7_min > cls_7_conf:
                    cls_7_min = cls_7_conf
            try:
                print(f"Class_7 AVG: {cls_7_sum / len(natual):.2f}%")
                with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                    f.write(f"Class_7 CNT: {len(natual)}, ")
                    f.write(f"Class_7 AVG: {cls_7_sum / len(natual):.2f}%, ")
                    f.write(f"Class_7 MAX: {cls_7_max:.2f}%, ")
                    f.write(f"Class_7 MIN: {cls_7_min:.2f}%\n")
            except ZeroDivisionError:
                print("No Class_7 Data")

            cls_8_sum = 0
            cls_8_min = 100.0
            cls_8_max = 0
            for cls_8 in tqdm(negative, desc='Class_8 images copying... '):
                img_path = str(cls_8[0])
                cls_8_conf = float(cls_8[2])
                annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                shutil.copy(cls_8[0], Path(args.pred_save_path) / '8_negative' / 'images')

                if not args.use_cropimg:
                    shutil.copy(annot_path, Path(args.pred_save_path) / '8_negative' / 'annotations')

                if args.pred_save_with_conf:
                    create_images_with_conf(img_path, cls_8, '8_negative', args.pred_save_path, args.use_cropimg, args)
                    
                cls_8_sum = cls_8_sum + cls_8_conf
                if cls_8_max < cls_8_conf:
                    cls_8_max = cls_8_conf
                if cls_8_min > cls_8_conf:
                    cls_8_min = cls_8_conf
            try:
                print(f"Class_8 AVG: {cls_8_sum / len(negative):.2f}%")
                with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                    f.write(f"Class_8 CNT: {len(negative)}, ")
                    f.write(f"Class_8 AVG: {cls_8_sum / len(negative):.2f}%, ")
                    f.write(f"Class_8 MAX: {cls_8_max:.2f}%, ")
                    f.write(f"Class_8 MIN: {cls_8_min:.2f}%\n")
            except ZeroDivisionError:
                print("No Class_8 Data")
            
            if len(pos_2_neg) == 0:
                for pn in tqdm(pos_2_neg, desc='Change to negative data due to low confidence level'):
                    img_path = str(pn[0])
                    pn_conf = float(pn[2])
                    annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                    shutil.copy(pn[0], Path(args.pred_save_path) / 'pos_2_neg' / 'images')
                    if not args.use_cropimg:
                        shutil.copy(annot_path, Path(args.pred_save_path) / 'pos_2_neg' / 'annotations')
                    if args.pred_save_with_conf:
                        create_images_with_conf(img_path, pn, 'pos_2_neg', args.pred_save_path, args.use_cropimg, args)


        
        # ##################################### save result image & anno #####################################

        # ##################################### save evalutations #####################################
        # import warnings
        # warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

        if args.pred_eval:
            if np.sum(np.array(result_v)[...,2]) < 0:
                conf_TN = [x[1] for x in result_v if (x[0]==0)]
                conf_TP = [x[1] for x in result_v if (x[0]==1)]
                conf_FN = []
                conf_FP = []

                # index set    
                itn = [i for i in range(len(result_v)) if (result_v[i][0]==0)]
                itp = [i for i in range(len(result_v)) if (result_v[i][0]==1)]

                # histogram P-N 
                plt.hist((conf_TN, conf_TP), label=('0_etc', '1_paper', '2_plastic+can+glass', '3_stone', '4_tire', '5_plastic_bag', '6_wood', '7_natual', '8_negative'),histtype='bar', bins=50)
                plt.xlabel('Confidence')
                plt.ylabel('Conunt')
                plt.legend(loc='upper left')
                # plt.savefig('image/'+args.pred_eval_name+'hist_PN.png')
                plt.savefig(args.pred_eval_name+'hist_PN.png')
                plt.close()

            else:
                y_pred = [i[0] for i in result_v]
                y_target = [i[2] for i in result_v]
                # y_pred = []
                # y_target = []
                # for i in result_v:
                #     if args.conf >= i[1] and i[0] != 8: # conf가 특정 값 이하이고 neg가 아닌 경우
                #         y_pred.append(0) # negative
                #         y_target.append(i[2])
                #     else:
                #         y_pred.append(i[0])
                #         y_target.append(i[2])     
                neg_val = 8

                # precision recall 계산
                precision = precision_score(y_target, y_pred, average= "macro")
                recall = recall_score(y_target, y_pred, average= "macro")
                cm = confusion_matrix(y_target, y_pred)
                cm_display = ConfusionMatrixDisplay(cm).plot()
                try:
                    cls_report = classification_report(y_target, y_pred, target_names=['0_etc', '1_paper', '2_plastic+can+glass', '3_stone', '4_tire', '5_plastic_bag', '6_wood', '7_natual', '8_negative'])
                except:
                    pass
                    # cls_report = classification_report(y_target, y_pred, target_names=["Class_0"])
            
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
                    for i in result_v:
                        if i[2] != 2:
                            y_pred = [i[0] for i in result_v]
                            y_target = [i[2] for i in result_v]
                            not_include_neg_list.append(i)
                    result_v = not_include_neg_list

                # collect data 
                conf_TN = [x[1] for p, t, x in zip(y_pred, y_target, result_v) if p==t and p==neg_val] 
                conf_TP = [x[1] for p, t, x in zip(y_pred, y_target, result_v) if p==t and p!=neg_val] 
                conf_FN = [x[1] for p, t, x in zip(y_pred, y_target, result_v) if p!=t and p==neg_val] 
                conf_FP = [x[1] for p, t, x in zip(y_pred, y_target, result_v) if p!=t and p!=neg_val] 
            
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
                itn = [i for i in range(len(result_v)) if (y_pred[i]==y_target[i] and y_pred[i]==neg_val)]
                itp = [i for i in range(len(result_v)) if (y_pred[i]==y_target[i] and y_pred[i]!=neg_val)]
                ifn = [i for i in range(len(result_v)) if (y_pred[i]!=y_target[i] and y_pred[i]==neg_val)]
                ifp = [i for i in range(len(result_v)) if (y_pred[i]!=y_target[i] and y_pred[i]!=neg_val)]
                
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

    # ##################################### save evalutations #####################################

@torch.no_grad()
def prediction_several_models_cls_name(args, device):
    import os
    import sys
    import random
    from collections import OrderedDict
    from preprocess_data import make_dataset_file
    from datasets import PotholeDataset, get_split_data
    from sklearn.metrics import precision_score , recall_score , confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    totorch = transforms.ToTensor()

    output_dir_path = args.pred_save_path
    graph_save_dir = args.pred_eval_name
    
    load_model_list = []
    result_dict = {}
    
    for idx, ckpt in enumerate(args.resume):
        # nb_cls = 9
        print(f'Model File: {ckpt}')
        name = ckpt.split('/')[-2]+ '_'
        ops = Path(ckpt).parts[-2]
        opsdict = dict(zip([str(d) for i,d in enumerate(str(ops).split('_')) if i%2==0], 
                       [str(d) for i, d in enumerate(ops.split('_')) if i%2==1]))
        if not os.path.exists(output_dir_path):
            os.makedirs(Path(output_dir_path), exist_ok=True)

        if not os.path.exists(graph_save_dir):
            os.makedirs(Path(graph_save_dir), exist_ok=True)
        
        args.pred_eval_name = os.path.join(graph_save_dir, name)
        args.pred_save_path = os.path.join(output_dir_path, name)
        args.nb_classes = int(opsdict['nbclss'])
        args.padding = opsdict['pad']
        args.padding_size = float(opsdict['padsize'])
        args.use_bbox = True if opsdict['box'] == 'True' else False
        args.use_shift = True if opsdict['shift'] == 'True' else False
        # print(args.resume)
        # print(args.padding, args.padding_size, args.use_bbox)
        # print(args.pred_eval_name, args.pred_save_path)
        # print(args.nb_classes, args.use_shift)

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
        utils.auto_load_model_with_model(
            ckpt, args=args, model=model, model_without_ddp=model,
            optimizer=None, loss_scaler=None, model_ema=None)
        model.eval()

        load_model_list.append((ckpt, model, idx))
        result_dict[ckpt] = []
    
    # Data laod     
    data_list = []
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

        # # File 이름에 label이 있는지 확인
        # spltnm = str(data[1]).split('_')
        spltnm = str(data[1])
        # target = int(spltnm[0][1]) if spltnm[0][0] == 't' else -1
        target = -1
        crop_img = cv2.resize(crop_img, (args.input_size, args.input_size))
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        pil_image=Image.fromarray(crop_img)
        input_tensor = totorch(pil_image).to(device)
        input_tensor = input_tensor.unsqueeze(dim=0)
        input_tensor = tonorm(input_tensor)

        for model_name, model_weights, i in load_model_list:
            with open(args.cls_name_txt[i], 'r') as f:
                class_name_list = [i.strip() for i in f.readlines()]

            if target == -1:
                for i in class_name_list:
                    if spltnm == i:
                        target = class_name_list.index(spltnm)
                # if target == -1:
                #     print(spltnm)
                #     print(data[0] / data.image_path)
            # model output 
            output_tensor = model_weights(input_tensor) 
            pred, conf = int(torch.argmax(output_tensor).detach().cpu().numpy()), float((torch.max(output_tensor)).detach().cpu().numpy())
            probs = softmax(output_tensor) # softmax 통과
            probs_max = ((torch.max(probs)).detach().cpu().numpy())*100

            result_dict[model_name].append((pred, probs_max, target, data[0] / data.image_path, data.label, data.bbox, str(idx)))
            idx += 1

    
    # ##################################### save result image & anno #####################################
    model_idx = 0
    for result_k, result_v in result_dict.items():
        class_name_list = []
        print(f'model name: {result_k}')
        with open(args.cls_name_txt[model_idx], 'r') as f:
                for i in f.readlines():
                    class_name_list.append(i.strip())
        # class_name_list == ['0_etc', '1_paper', '2_plastic+can+glass', '3_stone', '4_tire', '5_plastic_bag', '6_wood', '7_natual', '8_negative']
        name = result_k.split('/')[-2]+ '_'
        ops = Path(result_k).parts[-2]
        opsdict = dict(zip([str(d) for i,d in enumerate(str(ops).split('_')) if i%2==0], 
                       [str(d) for i, d in enumerate(ops.split('_')) if i%2==1]))

        if not os.path.exists(output_dir_path):
            os.makedirs(Path(output_dir_path), exist_ok=True)

        if not os.path.exists(graph_save_dir):
            os.makedirs(Path(graph_save_dir), exist_ok=True)
        
        args.pred_eval_name = os.path.join(graph_save_dir, name)
        args.pred_save_path = os.path.join(output_dir_path, name)
        if args.pred_save:
            import os
            for i in class_name_list:
                os.makedirs(Path(args.pred_save_path) / i / 'images', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) / i / 'annotations', exist_ok=True)

            if float(args.conf) > 0:
                os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'images', exist_ok=True)
                os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'annotations', exist_ok=True)
        
            if args.pred_save_with_conf:
                for i in class_name_list:
                    os.makedirs(Path(args.pred_save_path) / i / 'inference', exist_ok=True)
                    for j in ['true_data', 'false_data']:
                        os.makedirs(Path(args.pred_save_path) / i / j / 'images', exist_ok=True)
                        os.makedirs(Path(args.pred_save_path) / i / j / 'annotations', exist_ok=True)
                        os.makedirs(Path(args.pred_save_path) / i / j / 'inference', exist_ok=True)
            
                if float(args.conf) > 0:
                    os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'inference', exist_ok=True)
                    for i in ['true_data', 'false_data']:
                        os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / i / 'images', exist_ok=True)
                        os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / i / 'annotations', exist_ok=True)
                        os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / i / 'inference', exist_ok=True)

                if not args.use_cropimg:
                    for i in class_name_list:
                        os.makedirs(Path(args.pred_save_path) / i / 'true_data' / 'crop_images', exist_ok=True)
                        os.makedirs(Path(args.pred_save_path) / i / 'false_data' / 'crop_images', exist_ok=True)
                    if float(args.conf) > 0:
                        os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'true_data' / 'crop_images', exist_ok=True)
                        os.makedirs(Path(args.pred_save_path) /'pos_2_neg' / 'false_data' / 'crop_images', exist_ok=True)
            
            # etc = []
            # negative = []
            # paper = []
            # plastic = []
            # natual = []
            # stone = []
            # tire = []
            # plastic_bag = []
            # wood = []
            # pos_2_neg = []

            pred_dataset = OrderedDict()
            for i in class_name_list:
                pred_dataset[i] = []
            pos_2_neg = []

            idx = 0
            for k, v in pred_dataset.items():
                for x in result_v:
                    if x[0] == idx:
                        v.append((x[3], k, x[1], x[4], x[5], x[6]))
                        if x[1] >= args.conf:
                            pos_2_neg.append((x[3], 'pos_2_neg', x[1], x[4], x[5], x[6]))
                idx += 1


            with open(Path(args.pred_save_path)/"conf_avg.txt", 'w') as f:
                f.write('')

            # ### save_crop ###
            # if not os.path.exists(Path(args.pred_save_path) / 'negative' / 'crop_images'):
            #     os.mkdir(os.path.join(args.pred_save_path, 'negative', 'crop_images'))
            # if not os.path.exists(Path(args.pred_save_path) / 'positive' / 'crop_images'):
            #     os.mkdir(os.path.join(args.pred_save_path, 'positive', 'crop_images'))
            # ### save_crop ###

            
            for cls_name, data_list in pred_dataset.items():
                cls_sum = 0
                cls_min = 100.0
                cls_max = 0
                for cls_num in tqdm(data_list, desc=f'{cls_name} images copying... '):
                    img_path = str(cls_num[0])
                    cls_conf = float(cls_num[2])
                    annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                    shutil.copy(cls_num[0], Path(args.pred_save_path) / cls_name / 'images')

                    if not args.use_cropimg:
                        shutil.copy(annot_path, Path(args.pred_save_path) / cls_name / 'annotations')
                    
                    if args.pred_save_with_conf:
                        create_images_with_conf(img_path, cls_num, cls_name, args.pred_save_path, args.use_cropimg, args)

                    cls_sum = cls_sum + cls_conf
                    if cls_max < cls_conf:
                        cls_max = cls_conf
                    if cls_min > cls_conf:
                        cls_min = cls_conf
                
                try:
                    print(f"{cls_name} AVG: {cls_sum / len(data_list):.2f}%")
                    with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                        f.write(f"{cls_name} CNT: {len(data_list)}, ")
                        f.write(f"{cls_name} AVG: {cls_sum / len(data_list):.2f}%, ")
                        f.write(f"{cls_name} MAX: {cls_max:.2f}%, ")
                        f.write(f"{cls_name} MIN: {cls_min:.2f}%\n")
                except ZeroDivisionError:
                    print(f"No {cls_name} Data")
            
            if len(pos_2_neg) == 0:
                for pn in tqdm(pos_2_neg, desc='Change to negative data due to low confidence level'):
                    img_path = str(pn[0])
                    pn_conf = float(pn[2])
                    annot_path = (img_path[:-3]+'txt').replace('images', 'annotations')
                    shutil.copy(pn[0], Path(args.pred_save_path) / 'pos_2_neg' / 'images')
                    if not args.use_cropimg:
                        shutil.copy(annot_path, Path(args.pred_save_path) / 'pos_2_neg' / 'annotations')
                    if args.pred_save_with_conf:
                        create_images_with_conf(img_path, pn, 'pos_2_neg', args.pred_save_path, args.use_cropimg, args)


        
        # ##################################### save result image & anno #####################################

        # ##################################### save evalutations #####################################
        # import warnings
        # warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

        if args.pred_eval:
            # target class 번호의 합이 0 이상? (데이터 클래스 분류가 X이면 합이 음수임)  
            if np.sum(np.array(result_v)[..., 2]) < 0: 
                conf_TN = [x[1] for x in result_v if (x[0]==0)]
                conf_TP = [x[1] for x in result_v if (x[0]==1)]
                conf_FN = []
                conf_FP = []

                # index set    
                itn = [i for i in range(len(result_v)) if (result_v[i][0]==0)]
                itp = [i for i in range(len(result_v)) if (result_v[i][0]==1)]

                # histogram P-N 
                plt.hist((conf_TN, conf_TP), label=class_name_list, histtype='bar', bins=50)
                plt.xlabel('Confidence')
                plt.ylabel('Conunt')
                plt.legend(loc='upper left')
                # plt.savefig('image/'+args.pred_eval_name+'hist_PN.png')
                plt.savefig(args.pred_eval_name+'hist_PN.png')
                plt.close()

            else:
                y_pred = [i[0] for i in result_v]
                y_target = [i[2] for i in result_v]
                # y_pred = []
                # y_target = []
                # for i in result_v:
                #     if args.conf >= i[1] and i[0] != 8: # conf가 특정 값 이하이고 neg가 아닌 경우
                #         y_pred.append(0) # negative
                #         y_target.append(i[2])
                #     else:
                #         y_pred.append(i[0])
                #         y_target.append(i[2])    
                neg_val = -1
                for i, v in enumerate(class_name_list):
                    if 'neg' in v:
                        neg_val = i

                # precision recall 계산
                precision = precision_score(y_target, y_pred, average= "macro")
                recall = recall_score(y_target, y_pred, average= "macro")
                cm = confusion_matrix(y_target, y_pred)
                cm_display = ConfusionMatrixDisplay(cm).plot()
                accuracy = accuracy_score(y_target, y_pred)
                cls_report = ''
                try:
                    cls_report = classification_report(y_target, y_pred, target_names=class_name_list)
                except:
                    pass
                    # cls_report = classification_report(y_target, y_pred, target_names=["Class_0"])
            
                plt.title('Precision: {0:.4f}, Recall: {1:.4f}'.format(precision, recall))
                # plt.savefig('image/'+args.pred_eval_name+'cm.png')
                plt.savefig(args.pred_eval_name+'cm.png')
                plt.close()

                acc_class_dict = OrderedDict()
                for i in range(len(class_name_list)):
                    acc_class_dict[i] = [0, 0, 0] # true, false, total

                print(y_target.count(-1))

                for i in range(len(y_pred)):
                    t = y_target[i]
                    p = y_pred[i]
                    acc_class_dict[t][2] += 1
                    if t == p:
                        # print(acc_class_dict[t])
                        acc_class_dict[t][0] += 1
                    else:
                        # print(acc_class_dict[t])
                        acc_class_dict[t][1] += 1

                print()
                acc_string = ""
                for k, v in acc_class_dict.items():
                    # print(f'class: {class_name_list[k]:<20} ->  true: {v[0]:<4} false: {v[1]:<4} total:{v[2]:<4} Accuracy: {v[0] / v[2]:.4f}')
                    try:
                        acc_string += f'class: {class_name_list[k]:<50} ->  true: {v[0]:<4} false: {v[1]:<4} total:{v[2]:<4} Accuracy: {v[0] / v[2]:.4f}\n'
                    except:
                        acc_string += f'class: {class_name_list[k]:<50} ->  true: {v[0]:<4} false: {v[1]:<4} total:{v[2]:<4} Accuracy: 0\n'
                print(acc_string)

                print()
                print(cm)
                print("\n정확도(Accuracy): {0:.4f}".format(accuracy))
                print('정밀도(Precision): {0:.4f}, 재현율(Recall): {1:.4f}\n'.format(precision, recall))
                with open(Path(args.pred_save_path)/"conf_avg.txt", "a") as f:
                    f.write(f'\nModel File: {ckpt}\n')
                    f.write(f'\n{acc_string}\n')
                    f.write("정확도(Accuracy): {0:.4f}\n".format(accuracy))
                    f.write('정밀도(Precision): {0:.4f}, 재현율(Recall): {1:.4f}\n'.format(precision, recall))
                    f.write(cls_report)
                    f.write('F1-score : {0:.4f}\n'.format(2 * (precision * recall) / (precision + recall)))
                    f.write(str(cm))
                print(cls_report)
                print('F1-score : {0:.4f}\n'.format(2 * (precision * recall) / (precision + recall)))

                if args.eval_not_include_neg:
                    not_include_neg_list = []
                    for i in result_v:
                        if i[2] != 2:
                            y_pred = [i[0] for i in result_v]
                            y_target = [i[2] for i in result_v]
                            not_include_neg_list.append(i)
                    result_v = not_include_neg_list

                # collect data 
                conf_TN = [x[1] for p, t, x in zip(y_pred, y_target, result_v) if p==t and p==neg_val] 
                conf_TP = [x[1] for p, t, x in zip(y_pred, y_target, result_v) if p==t and p!=neg_val] 
                conf_FN = [x[1] for p, t, x in zip(y_pred, y_target, result_v) if p!=t and p==neg_val] 
                conf_FP = [x[1] for p, t, x in zip(y_pred, y_target, result_v) if p!=t and p!=neg_val] 
            
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
                itn = [i for i in range(len(result_v)) if (y_pred[i]==y_target[i] and y_pred[i]==neg_val)]
                itp = [i for i in range(len(result_v)) if (y_pred[i]==y_target[i] and y_pred[i]!=neg_val)]
                ifn = [i for i in range(len(result_v)) if (y_pred[i]!=y_target[i] and y_pred[i]==neg_val)]
                ifp = [i for i in range(len(result_v)) if (y_pred[i]!=y_target[i] and y_pred[i]!=neg_val)]
                
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

                # sample prediction images 
                if args.sample_images:
                    rows = 3
                    cols = 3
                    img_num = 3 
                    img_cnt = rows * cols
                    total_cnt = img_num*img_cnt

                    if total_cnt > len(result_v):
                        total_cnt = len(result_v)

                    sample_data = list(random.sample(result_v, total_cnt))
                    sample_images = [Image.open(i[3]) for i in sample_data]

                    for n in range(img_num):
                        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20,20))
                        fig.subplots_adjust(wspace=0.7, hspace=0.7)  # Adjust the horizontal and vertical space
   
                        img_count = 0
                        for i in range(rows):
                            for j in range(cols):
                                try:
                                    axes[i, j].imshow(sample_images[img_count + (img_cnt)*n])
                                    axes[i, j].axis('off')
                                    # t = f'target: {sample_data[img_count + (img_cnt)*n][2]}, pred: {sample_data[img_count + (img_cnt)*n][0]}, {sample_data[img_count + (img_cnt)*n][1]:.2f}%'
                                    l = '_'.join((sample_data[img_count + (img_cnt)*n][4]).split('_')[1:])
                                    # p = class_name_list.index(int((sample_data[img_count + (img_cnt)*n][4]).split('_')[0]))
                                    # 클래스 번호가 중간에 뚫리면 바꿔야함
                                    p = '_'.join((class_name_list[int((sample_data[img_count + (img_cnt)*n][4]).split('_')[0])]).split('_')[1:])
                            
                                    # t = f'target: {sample_data[img_count + (img_cnt)*n][4]}, pred: {sample_data[img_count + (img_cnt)*n][0]}, {sample_data[img_count + (img_cnt)*n][1]:.2f}%'
                                    # t = f'label: {l}, pred: {p}, {sample_data[img_count + (img_cnt)*n][1]:.2f}%'
                                    # t = f'label: {l}, pred: {p}'
                                    t = f'{l} -> {p}'
                                    axes[i, j].set_title(t, fontsize=25)
                                    img_count+=1

                                except:
                                    continue

                        plt.savefig(f'{args.pred_eval_name}pred_sample_{n}.png', dpi=400)
                        plt.close()
                        print(f"Save Sample Prediction Images {n+1}/{img_num}")

        model_idx += 1
        print(f"Finish Prediction {ckpt}")
    # ##################################### save evalutations #####################################
