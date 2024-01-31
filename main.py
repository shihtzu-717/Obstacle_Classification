# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import sys
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import os

from pathlib import Path

from timm.data.mixup import Mixup
from timm.models import create_model

from timm.utils import ModelEma
from optim_factory import create_optimizer, LayerDecayValueAssigner

from datasets import build_dataset, PotholeDataset, get_split_data
from preprocess_data import make_dataset_file, make_crop_dataset_file
# from engine import train_one_epoch, evaluate, prediction, prediction_several_models, prediction_several_models_cls_name

from engine_feature import train_one_epoch, evaluate, prediction, prediction_several_models_cls_name

from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import models.convnext
import models.convnext_isotropic

def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script for image classification', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Per GPU batch size')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--update_freq', default=1, type=int,
                        help='gradient accumulation steps')

    # Model parameters
    parser.add_argument('--model', default='convnext_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.2, metavar='PCT',
                        help='Drop path rate (default: 0.2)')
    parser.add_argument('--input_size', default=224, type=int,
                        help='image input size')
    parser.add_argument('--layer_scale_init_value', default=1e-6, type=float,
                        help="Layer scale initial values")

    # EMA related parameters
    parser.add_argument('--model_ema', type=str2bool, default=False)
    parser.add_argument('--model_ema_decay', type=float, default=0.9999, help='')
    parser.add_argument('--model_ema_force_cpu', type=str2bool, default=False, help='')
    parser.add_argument('--model_ema_eval', type=str2bool, default=False, help='Using ema to eval during training.')

    # Optimization parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--cosine_scheduler', type=str2bool, default=True,
                        help='learn with cosine scheduler')


    parser.add_argument('--lr', type=float, default=4e-4, metavar='LR',
                        help='learning rate (default: 4e-3), with total batch size 4096')
    parser.add_argument('--layer_decay', type=float, default=1.0)
    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-6)')
    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='num of steps to warmup LR, will overload warmup_epochs if set > 0')
    parser.add_argument('--lossfn', type=str, default='BCE', help='loss function'),

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.0,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    # Evaluation parameters
    parser.add_argument('--crop_pct', type=float, default=None)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', type=str2bool, default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    # parser.add_argument('--mixup', type=float, default=0.0,
    #                     help='mixup alpha, mixup enabled if > 0.')
    # parser.add_argument('--cutmix', type=float, default=0.0,
    #                     help='cutmix alpha, cutmix enabled if > 0.')
    # parser.add_argument('--use_softlabel', type=str2bool, default=False,
    #                     help='Softlabel using 2classes output for 4classes input.')

    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--head_init_scale', default=1.0, type=float,
                        help='classifier head initial scale, typically adjusted in fine-tuning')
    parser.add_argument('--model_key', default='model|module', type=str,
                        help='which key to load from saved state dict, usually model or model_ema')
    parser.add_argument('--model_prefix', default='', type=str)

    # Dataset parameters
    parser.add_argument('--data_path', default=[], nargs='+', type=str, help='dataset path')
    parser.add_argument('--eval_data_path', type=str, nargs='+', help='dataset path for evaluation')
    parser.add_argument('--nb_classes', default=4, type=int,
                        help='number of the classification types')
    parser.add_argument('--imagenet_default_mean_and_std', type=str2bool, default=True)
    parser.add_argument('--data_set', default='image_folder', choices=['CIFAR', 'IMNET', 'image_folder'],
                        type=str, help='ImageNet dataset path')
    parser.add_argument('--output_dir', default='results',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='log',
                        help='path where to tensorboard log')
    parser.add_argument('--log_name', default=None,
                        help='name where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--resume', default='', nargs='+', help='resume from checkpoint')
    # parser.add_argument('--resume', default='', help='resume from checkpoint')

    parser.add_argument('--auto_resume', type=str2bool, default=False)
    parser.add_argument('--save_ckpt', type=str2bool, default=True)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--save_ckpt_num', default=1, type=int)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', type=str2bool, default=False,
                        help='Perform evaluation only')

    parser.add_argument('--dist_eval', type=str2bool, default=True,
                        help='Enabling distributed evaluation')
    parser.add_argument('--disable_eval', type=str2bool, default=False,
                        help='Disabling evaluation during training')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', type=str2bool, default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', type=str2bool, default=False)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    parser.add_argument('--use_amp', type=str2bool, default=False,
                        help="Use PyTorch's AMP (Automatic Mixed Precision) or not")

    # Weights and Biases arguments
    parser.add_argument('--enable_wandb', type=str2bool, default=False,
                        help="enable logging to Weights and Biases")
    parser.add_argument('--project', default='convnext', type=str,
                        help="The name of the W&B project where you're sending the new run.")
    parser.add_argument('--wandb_run_nm', default='convnext', type=str,
                        help="The name of the W&B run name where you're sending the new run.")
    parser.add_argument('--wandb_ckpt', type=str2bool, default=False,
                        help="Save model checkpoints as W&B Artifacts.")

    # Image Crop & Padding
    parser.add_argument('--use_cropimg', type=str2bool, default=False,
                        help='Use oringinal code for data input')
    parser.add_argument('--split_file_write', type=str2bool, default=True,
                        help='split_file_write')
    parser.add_argument('--upsample', default=[1, 20, 1, 25], nargs='+', type=int)
    parser.add_argument('--padding', type=str, default='FIX2',
                        help='select padding mode')
    parser.add_argument('--padding_size', type=float, default=0.0,
                        help='set padding size')
    parser.add_argument('--use_shift', type=str2bool, default=False,
                        help="enable use_shift mode in image cropping")
    parser.add_argument('--use_bbox', type=str2bool, default=False,
                        help="enable use_bbox mode in image cropping")
    parser.add_argument('--imsave', type=str2bool, default=False,
                        help="enable imsave mode in image cropping")
    parser.add_argument('--test_val_ratio', default=[0.0, 0.2], nargs='+', type=float)
    parser.add_argument('--use_class', default=[0,1,2,3,4,5,6], nargs='+', type=int)
    parser.add_argument('--label_list', default=None, nargs='+', type=str) # ['positive', 'negative']

    # predictions and evaluations
    parser.add_argument('--pred', type=str2bool, default=False, help='Perform prediction only')
    parser.add_argument('--pred_save', type=str2bool, default=False, help='Save prediction result')
    # parser.add_argument('--pred_save_path', type=str, default='/home/daree/nas/set1', help='set root path for save result images')
    parser.add_argument('--pred_save_path', type=str, help='set root path for save result images')
    parser.add_argument('--pred_eval', type=str2bool, default=True, help='Save prediction evaluation')
    parser.add_argument('--pred_eval_name', type=str, default='', help='name for saving graph')

    # parser.add_argument('--soft_label_ratio', type=float, default=0.7, help='soft lebel ratio')
    # parser.add_argument('--soft_type', type=int, default=1, help='soft lebel type')
    # parser.add_argument('--label_ratio', type=float, default=0.95, help='name for saving graph')

    parser.add_argument('--pred_save_with_conf', type=str2bool, default=False, help='Save prediction result with confidence')
    parser.add_argument('--eval_not_include_neg', type=str2bool, default=False, help='evaluation not include negative data')

    parser.add_argument('--path_type', type=str2bool, required=True, default=False, help='Data input type is path')
    parser.add_argument('--txt_type', type=str2bool, required=True, default=False, help='Data input type is txt file')
    parser.add_argument('--train_txt_path', type=str, default="", help='Train Data Input Path')
    parser.add_argument('--valid_txt_path', type=str, default="", help='Validation Data Input Path')
    parser.add_argument('--conf', type=float, default="0.0", help='Confidence Level')
    parser.add_argument('--several_models', type=str2bool, default=False, help='Use Several Models')
    parser.add_argument('--cls_name_txt', required=True, default="", nargs='+', help='Class Name Text File Path')
    parser.add_argument('--sample_images', type=str2bool, default=False, help='Save Prediction Sample Images')
    parser.add_argument('--early_stop', type=str2bool, default=True, help='Use Early Stoping')
    parser.add_argument('--early_stop_epoch', type=int, default=40, help='Use Early Stoping Epoch')
    parser.add_argument('--use_type', type=str2bool, default=False, help='Inference Based on Data Type')
    
    return parser

def main(args):
    utils.init_distributed_mode(args)

    print(args)
    with open(Path(args.output_dir)/'args.txt', 'w') as f:
        f.write(str(args))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # Prediction 실행 시
    if args.pred:
        if args.several_models:
            # prediction_several_models(args, device)
            print(args.cls_name_txt)
            prediction_several_models_cls_name(args, device)
        else:
            prediction(args, device)
        return

    # Evlaluation 실행 시
    elif args.eval:
        if args.path_type:
            sets = get_split_data(data_root=Path(args.eval_data_path),
                                  test_r=args.test_val_ratio[0],
                                  val_r=args.test_val_ratio[1],
                                  file_write=args.split_file_write,
                                  label_list = args.label_list)
            dataset_val = PotholeDataset(
                data_set=sets['test'],
                data_path=Path(args.eval_data_path),
                args=args,
                is_train=False)
            dataset_train = dataset_val
        elif args.txt_type:
            if args.valid_txt_path == "":
                print("Please Check the valid_txt_path")
                sys.exit(1)

            valid_data = make_dataset_file(args.valid_txt_path)
            dataset_val = PotholeDataset(
                data_set=valid_data,
                data_path=Path(args.eval_data_path),
                args=args,
                is_train=False)
            dataset_train = dataset_val

    # Train 실행 시
    else:
        if args.path_type:
            tr=[]
            vr=[]
            for path in args.data_path:
                settmp = get_split_data(data_root=Path(path),
                                        test_r=args.test_val_ratio[0],
                                        val_r=args.test_val_ratio[1],
                                        file_write=args.split_file_write,
                                        label_list = args.label_list,
                                        use_cropimg = args.use_cropimg)
                tr = tr + settmp['train']
                vr = vr + settmp['val']
            sets = dict(train=tr, val=vr)
            dataset_train = PotholeDataset(
                data_set=sets['train'],
                args=args)
            dataset_val = PotholeDataset(
                data_set=sets['val'],
                args=args,
                is_train=False)

            train_set = set()
            for i in sets['train']:
                train_path = os.path.join(i.data_set, i.image_path)
                if not os.path.exists(train_path):
                    print(train_path, "is not exists")
                else:
                    train_set.add(train_path+'\n')

            val_set = set()
            for i in sets['val']:
                val_path = os.path.join(i.data_set, i.image_path)
                if not os.path.exists(val_path):
                    print(val_path, "is not exists")
                else:
                    val_set.add(val_path+'\n')

            with open(Path(args.output_dir)/'train.txt', 'w') as f:
                f.writelines(train_set)
            with open(Path(args.output_dir)/'valid.txt', 'w') as f:
                f.writelines(val_set)

        elif args.txt_type:
            if args.train_txt_path == "":
                print("Please Check the train_txt_path")
                sys.exit(1)
            if args.valid_txt_path == "":
                print("Please Check the valid_txt_path")
                sys.exit(1)

            if args.use_cropimg:
                train_data = make_crop_dataset_file(args.train_txt_path)
                valid_data = make_crop_dataset_file(args.valid_txt_path)
            else:
                train_data = make_dataset_file(args.train_txt_path)
                valid_data = make_dataset_file(args.valid_txt_path)
            # print(len(train_data))
            # print(len(valid_data))

            dataset_train = PotholeDataset(
                data_set=train_data,  # 그냥 RawData 리스트를 주면 됨.
                args=args)
            dataset_val = PotholeDataset(
                data_set=valid_data,
                args=args,
                is_train=False)

        # Data 수량 확인
        s=num_cl_tr=num_cl_vl=''
        for cl in dataset_train.classes:
            s = s + '   ' + cl
            num_cl_tr = num_cl_tr + '       ' + str(len([i for i in dataset_train.input_set[3] if i==cl]))
            num_cl_vl = num_cl_vl + '       ' + str(len([i for i in dataset_val.input_set[3] if i==cl]))
        print("----------------------------------------------\ndata count after upsampling")
        print(s)
        print(num_cl_tr)
        print(num_cl_vl)

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed,
    )
    print("Sampler_train = %s" % str(sampler_train))

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    else:
        sampler_val = torch.utils.data.RandomSampler(dataset_val, shuffle=True)

    # Set log folder name
    if global_rank == 0 and args.log_dir is not None:
        KST = datetime.timezone(datetime.timedelta(hours=9))
        d = datetime.datetime.now(tz=KST)
        if args.log_name is not None:
            args.log_dir = args.log_dir + '/' + args.log_name
        else:
            # 이름이 지정되지 않으면 날짜-시간으로 설정
            args.log_dir = args.log_dir + f'/time_{d.strftime("%y%m%d")}_{d.strftime("%H%M%S")}'

        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    if global_rank == 0 and args.enable_wandb:
        wandb_logger = utils.WandbLogger(args)
    else:
        wandb_logger = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # 모델 생성
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        layer_scale_init_value=args.layer_scale_init_value,
        head_init_scale=args.head_init_scale,
        )

    # Pretrain model load
    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')
        print("Using EMA with decay = %.8f" % args.model_ema_decay)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Update frequent = %d" % args.update_freq)
    print("Number of training examples = %d" % len(dataset_train))
    print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

    if args.layer_decay < 1.0 or args.layer_decay > 1.0:
        num_layers = 12 # convnext layers divided into 12 parts, each with a different decayed lr value.
        assert args.model in ['convnext_small', 'convnext_base', 'convnext_large', 'convnext_xlarge'], \
             "Layer Decay impl only supports convnext_small/base/large/xlarge"
        assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
    else:
        assigner = None

    if assigner is not None:
        print("Assigned values = %s" % str(assigner.values))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp, skip_list=None,
        get_num_layer=assigner.get_layer_id if assigner is not None else None,
        get_layer_scale=assigner.get_scale if assigner is not None else None)

    loss_scaler = NativeScaler() # if args.use_amp is False, this won't be used

    if args.cosine_scheduler:
        print("Use Cosine LR scheduler")
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
    else:
        lr_schedule_values = None

    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    if len(wd_schedule_values):
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    else:
        print("wd_schedule_values is empty")

    from softLabelLoss import softLabelLoss
    criterion = softLabelLoss(args, mixup_fn)

    print("criterion = %s" % str(criterion))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)

    if args.eval:
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, criterion=criterion, use_amp=args.use_amp, use_softlabel=args.use_softlabel)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
        with open('results/eval.txt', 'a', encoding='utf-8') as af:
            af.write(f'{args.resume}\t{args.eval_data_path}')
            for class_name in dataset_val.classes:
                acc_name = f'acc1_{class_name}'
                if acc_name in list(test_stats.keys()):
                    acc_value = test_stats[acc_name]
                    af.write(f'\t{acc_name}\t{acc_value}')
            af.write('\n')
        print(test_stats)
        return

    max_accuracy = 0.0
    val_loss = 1.0
    min_loss = 1.0
    if args.model_ema and args.model_ema_eval:
        max_accuracy_ema = 0.0

    ## 학습 시작
    print("Start training for %d epochs" % args.epochs)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
        if wandb_logger:
            wandb_logger.set_steps()
        # train_stats = train_one_epoch(
        #     model, criterion, data_loader_train, optimizer,
        #     device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
        #     log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
        #     lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
        #     num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,
        #     use_amp=args.use_amp, use_softlabel=args.use_softlabel
        # )
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer,
            device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
            log_writer=log_writer, wandb_logger=wandb_logger, start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
            num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,use_amp=args.use_amp
        )
        if args.output_dir and args.save_ckpt:
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                utils.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)

        if train_stats["loss"] < min_loss:
            min_loss = train_stats["loss"]

        if data_loader_val is not None:
            # test_stats = evaluate(data_loader_val, model, device, criterion=criterion, use_amp=args.use_amp, use_softlabel=args.use_softlabel)
            test_stats = evaluate(data_loader_val, model, device, criterion=criterion, use_amp=args.use_amp)
            print(f"Accuracy of the model on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            if max_accuracy < test_stats["acc1"]:
                max_accuracy = test_stats["acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best", model_ema=model_ema)
                    torch.save(model.state_dict(), Path(args.output_dir) / 'checkpoint-best_weights.pth')

            if test_stats["loss"] < val_loss:
                val_loss = test_stats["loss"]

            print(f'Max accuracy: {max_accuracy:.2f}%\n\n')

            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            # repeat testing routines for EMA, if ema eval is turned on
            if args.model_ema and args.model_ema_eval:
                test_stats_ema = evaluate(data_loader_val, model_ema.ema, device, use_amp=args.use_amp, use_softlabel=args.use_softlabel)
                print(f"Accuracy of the model EMA on {len(dataset_val)} test images: {test_stats_ema['acc1']:.1f}%")
                if max_accuracy_ema < test_stats_ema["acc1"]:
                    max_accuracy_ema = test_stats_ema["acc1"]
                    if args.output_dir and args.save_ckpt:
                        utils.save_model(
                            args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch="best-ema", model_ema=model_ema)
                    print(f'Max EMA accuracy: {max_accuracy_ema:.2f}%')
                if log_writer is not None:
                    log_writer.update(test_acc1_ema=test_stats_ema['acc1'], head="perf", step=epoch)
                log_stats.update({**{f'test_{k}_ema': v for k, v in test_stats_ema.items()}})
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        if wandb_logger:
            wandb_logger.log_epoch_metrics(log_stats)
        
        # Early Stopping
        if args.early_stop:
            if epoch == args.early_stop_epoch:
                break

    if wandb_logger and args.wandb_ckpt and args.save_ckpt and args.output_dir:
        wandb_logger.log_checkpoints()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ConvNeXt training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
