# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)  # 创建一个参数解析器,用于DETR检测器,不显示帮助信息
    parser.add_argument('--lr', default=1e-4, type=float)  # 设置基础学习率,默认为0.0001
    parser.add_argument('--lr_backbone', default=1e-5, type=float)  # 设置backbone网络的学习率,默认为0.00001
    parser.add_argument('--batch_size', default=2, type=int)  # 设置训练的批次大小,默认为2
    parser.add_argument('--weight_decay', default=1e-4, type=float)  # 设置权重衰减,用于防止过拟合,默认为0.0001
    parser.add_argument('--epochs', default=300, type=int)  # 设置训练的总轮数,默认为300轮
    parser.add_argument('--lr_drop', default=200, type=int)  # 设置学习率下降的轮数,默认在第200轮降低学习率
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')  # 设置梯度裁剪的最大范数,防止梯度爆炸,默认为0.1

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")  # 预训练模型的路径,如果设置则只训练mask head

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")  # 选择使用的backbone网络,默认为resnet50
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")  # 是否使用空洞卷积,如果为True则在最后一个卷积块中使用
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")  # 位置编码的类型,可选sine或learned,默认使用sine

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")  # Transformer编码器的层数,默认为6层
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")  # Transformer解码器的层数,默认为6层
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")  # Transformer前馈网络的中间维度,默认为2048
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")  # Transformer的嵌入维度,默认为256。这是模型的核心参数,决定了特征表示的丰富程度。维度越大模型容量越大,但计算开销也越大。256是在效果和效率之间的折中选择
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")  # Transformer中使用的dropout比率,默认为0.1
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")  # Transformer中注意力头的数量,默认为8。注意力头的数量影响模型捕捉不同特征的能力 - 更多的头可以并行关注不同的特征模式,但也会增加计算量。头的数量通常设置为隐藏维度的1/32到1/64,8个头是常用的默认值
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")  # 查询的数量,即预测的目标框数量,默认为100
    parser.add_argument('--pre_norm', action='store_true')  # 是否在Transformer中使用预归一化

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")  # 是否训练分割头,如果设置则进行实例分割任务

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")  # 是否禁用辅助解码损失

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")  # 匹配代价中类别项的系数,默认为1
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")  # 匹配代价中边界框L1距离项的系数,默认为5
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")  # 匹配代价中GIoU项的系数,默认为2

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)  # mask损失的权重系数,默认为1
    parser.add_argument('--dice_loss_coef', default=1, type=float)  # dice损失的权重系数,默认为1
    parser.add_argument('--bbox_loss_coef', default=5, type=float)  # 边界框损失的权重系数,默认为5
    parser.add_argument('--giou_loss_coef', default=2, type=float)  # GIoU损失的权重系数,默认为2
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")  # 无目标类别的相对分类权重,默认为0.1

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')  # 使用的数据集,默认为COCO数据集
    parser.add_argument('--coco_path', type=str)  # COCO数据集的路径
    parser.add_argument('--coco_panoptic_path', type=str)  # COCO全景分割数据集的路径
    parser.add_argument('--remove_difficult', action='store_true')  # 是否移除困难样本

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')  # 输出目录的路径,为空则不保存
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')  # 使用的设备,默认使用CUDA
    parser.add_argument('--seed', default=42, type=int)  # 随机种子,默认为42
    parser.add_argument('--resume', default='', help='resume from checkpoint')  # 从检查点恢复训练的路径
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')  # 开始训练的轮数,默认为0
    parser.add_argument('--eval', action='store_true')  # 是否只进行评估
    parser.add_argument('--num_workers', default=2, type=int)  # 数据加载的工作进程数,默认为2

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')  # 分布式训练的进程数,默认为1
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')  # 设置分布式训练的URL
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
