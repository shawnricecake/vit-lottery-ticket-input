# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for adjusting keep rate -- Youwei Liang
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

from helpers import adjust_keep_rate
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    writer=None,
                    set_training_mode=True,
                    args=None,
                    model_pretrained=None,
                    all_index_record=None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 200
    log_interval = 100
    it = epoch * len(data_loader)
    ITERS_PER_EPOCH = len(data_loader)

    base_rate = args.base_keep_rate

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        #####################################################
        if args.adjust_keep_rate:
            keep_rate = adjust_keep_rate(it, epoch, warmup_epochs=args.shrink_start_epoch,
                                             total_epochs=args.shrink_start_epoch + args.shrink_epochs,
                                             ITERS_PER_EPOCH=ITERS_PER_EPOCH, base_keep_rate=base_rate)
        else:
            keep_rate = base_rate
        #####################################################

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():

            #####################################################
            if args.lottery and model_pretrained is not None and keep_rate < 1:
                outputs = model_pretrained(samples, keep_rate=keep_rate)
                all_index_record = []
                for e in model_pretrained.blocks:
                    all_index_record.append(e.idx_record)
                model.module.all_idx_record = all_index_record
            elif args.random and keep_rate < 1:
                if "small" in args.model:
                    N = 197
                    repeat = 384
                elif "tiny" in args.model:
                    N = 197
                    repeat = 192
                elif "base" in args.model:
                    N = 197
                    repeat = 768
                    if "dim576" in args.model:
                        repeat = 576
                else:
                    assert "Error: have not support this kind model: {}".format(args.model)
                batch_size_here = samples.shape[0]
                all_index_record = []

                #######################################################################################################
                left_tokens_pre = N - 1
                left_tokens = math.ceil(keep_rate * (N - 1))
                mask1 = torch.zeros(batch_size_here, left_tokens, 1, requires_grad=False).to(device, non_blocking=True)
                for b in range(batch_size_here):
                    mask_tt = np.random.choice(left_tokens_pre, left_tokens, replace=False)
                    mask_tt = np.array([mask_tt])
                    mask_tt = np.transpose(mask_tt)
                    mask_tt = torch.from_numpy(mask_tt)
                    mask_tt = mask_tt.to(device, non_blocking=True)
                    mask1[b, :, :] = mask_tt
                mask1 = mask1.type(torch.int64)
                mask1 = mask1.repeat(1, 1, repeat)
                all_index_record.append(mask1)

                left_tokens_pre = left_tokens
                left_tokens = math.ceil(keep_rate * left_tokens)
                mask2 = torch.zeros(batch_size_here, left_tokens, 1, requires_grad=False).to(device, non_blocking=True)
                for b in range(batch_size_here):
                    mask_tt = np.random.choice(left_tokens_pre, left_tokens, replace=False)
                    mask_tt = np.array([mask_tt])
                    mask_tt = np.transpose(mask_tt)
                    mask_tt = torch.from_numpy(mask_tt)
                    mask_tt = mask_tt.to(device, non_blocking=True)
                    mask2[b, :, :] = mask_tt
                mask2 = mask2.type(torch.int64)
                mask2 = mask2.repeat(1, 1, repeat)
                all_index_record.append(mask2)

                left_tokens_pre = left_tokens
                left_tokens = math.ceil(keep_rate * left_tokens)
                mask3 = torch.zeros(batch_size_here, left_tokens, 1, requires_grad=False).to(device, non_blocking=True)
                for b in range(batch_size_here):
                    mask_tt = np.random.choice(left_tokens_pre, left_tokens, replace=False)
                    mask_tt = np.array([mask_tt])
                    mask_tt = np.transpose(mask_tt)
                    mask_tt = torch.from_numpy(mask_tt)
                    mask_tt = mask_tt.to(device, non_blocking=True)
                    mask3[b, :, :] = mask_tt
                mask3 = mask3.type(torch.int64)
                mask3 = mask3.repeat(1, 1, repeat)
                all_index_record.append(mask3)
                #######################################################################################################

                model.module.all_idx_record = all_index_record
            elif args.random_fixed:
                model.module.all_idx_record = all_index_record
            #####################################################

            outputs = model(samples, keep_rate=keep_rate if not (args.lottery or args.random) else None,
                            qkv_change=args.qkv_change)

            #####################################################
            # return to None for the test part
            model.module.all_idx_record = None
            #####################################################

            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if torch.distributed.get_rank() == 0 and it % log_interval == 0:
            writer.add_scalar('loss', loss_value, it)
            writer.add_scalar('lr', optimizer.param_groups[0]["lr"], it)
            writer.add_scalar('keep_rate', keep_rate, it)
        it += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, keep_rate


@torch.no_grad()
def evaluate(data_loader, model, device, keep_rate=None, args=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, keep_rate=keep_rate, qkv_change=args.qkv_change)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_sparse(data_loader, model, device, keep_rate=None, args=None,
                    model_pretrained=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():

            #####################################################
            if args.sparse_eval and model_pretrained is not None:
                outputs = model_pretrained(images, keep_rate=keep_rate)
                all_index_record = []
                for e in model_pretrained.blocks:
                    all_index_record.append(e.idx_record)
                model.module.all_idx_record = all_index_record
            elif args.sparse_eval and args.random and keep_rate < 1:
                if "small" in args.model:
                    N = 197
                    repeat = 384
                elif "tiny" in args.model:
                    N = 197
                    repeat = 192
                elif "base" in args.model:
                    N = 197
                    repeat = 768
                else:
                    assert "Error: have not support this kind model: {}".format(args.model)
                batch_size_here = images.shape[0]
                all_index_record = []

                #######################################################################################################
                left_tokens_pre = N - 1
                left_tokens = math.ceil(keep_rate * (N - 1))
                mask1 = torch.zeros(batch_size_here, left_tokens, 1, requires_grad=False).to(device, non_blocking=True)
                for b in range(batch_size_here):
                    mask_tt = np.random.choice(left_tokens_pre, left_tokens, replace=False)
                    mask_tt = np.array([mask_tt])
                    mask_tt = np.transpose(mask_tt)
                    mask_tt = torch.from_numpy(mask_tt)
                    mask_tt = mask_tt.to(device, non_blocking=True)
                    mask1[b, :, :] = mask_tt
                mask1 = mask1.type(torch.int64)
                mask1 = mask1.repeat(1, 1, repeat)
                all_index_record.append(mask1)

                left_tokens_pre = left_tokens
                left_tokens = math.ceil(keep_rate * left_tokens)
                mask2 = torch.zeros(batch_size_here, left_tokens, 1, requires_grad=False).to(device, non_blocking=True)
                for b in range(batch_size_here):
                    mask_tt = np.random.choice(left_tokens_pre, left_tokens, replace=False)
                    mask_tt = np.array([mask_tt])
                    mask_tt = np.transpose(mask_tt)
                    mask_tt = torch.from_numpy(mask_tt)
                    mask_tt = mask_tt.to(device, non_blocking=True)
                    mask2[b, :, :] = mask_tt
                mask2 = mask2.type(torch.int64)
                mask2 = mask2.repeat(1, 1, repeat)
                all_index_record.append(mask2)

                left_tokens_pre = left_tokens
                left_tokens = math.ceil(keep_rate * left_tokens)
                mask3 = torch.zeros(batch_size_here, left_tokens, 1, requires_grad=False).to(device, non_blocking=True)
                for b in range(batch_size_here):
                    mask_tt = np.random.choice(left_tokens_pre, left_tokens, replace=False)
                    mask_tt = np.array([mask_tt])
                    mask_tt = np.transpose(mask_tt)
                    mask_tt = torch.from_numpy(mask_tt)
                    mask_tt = mask_tt.to(device, non_blocking=True)
                    mask3[b, :, :] = mask_tt
                mask3 = mask3.type(torch.int64)
                mask3 = mask3.repeat(1, 1, repeat)
                all_index_record.append(mask3)
                #######################################################################################################

                model.module.all_idx_record = all_index_record
            #####################################################

            output = model(images, keep_rate=keep_rate, qkv_change=args.qkv_change)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def get_acc(data_loader, model, device, keep_rate=None, tokens=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, keep_rate, tokens)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return metric_logger.acc1.global_avg
