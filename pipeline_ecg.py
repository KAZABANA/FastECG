# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:25:30 2023

@author: COCHE User
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from datacollection import ECGcodedataset_loading,ECGdataset_prepare_finetuning_sepe_semi
from torchmetrics.classification import MultilabelAveragePrecision
from model_src_ecg.model_code_default import NN_default, Cutmix, tar_augmentation,DistributionBalancedLoss
import os
from tqdm import tqdm
from pytorchtools import EarlyStopping
from evaluation import print_result, find_thresholds
from scipy.stats import rankdata
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
def setup():
    dist.init_process_group('nccl')
def cleanup():
    dist.destroy_process_group()
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

## model validation on single GPU
def validate(model, valloader, device, iftest=False, threshold=0.5 * np.ones(5), iftrain=False, args=None):
    model.eval()
    losses, probs, lbls, logit = [], [], [], []
    for step, (inp_windows_t, lbl_t) in enumerate(valloader):
        inp_windows_t, lbl_t = inp_windows_t.float().to(device), lbl_t.int().to(device)
        with torch.no_grad():
            out = model(inp_windows_t)
            loss = F.binary_cross_entropy_with_logits(out, lbl_t.float())
            prob = out.sigmoid().data.cpu().numpy()
            losses.append(loss.item())
            probs.append(prob)
            lbls.append(lbl_t.data.cpu().numpy())
            logit.append(out.data.cpu().numpy())
    lbls = np.concatenate(lbls)
    probs = np.concatenate(probs)
    if iftest:
        valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'test', threshold)
    elif iftrain:
        threshold = find_thresholds(lbls.copy(), probs.copy())
        valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'train', threshold)
    else:
        threshold = find_thresholds(lbls, probs)
        valid_result = print_result(np.mean(losses), lbls, probs, 'valid', threshold)
    neg_ratio = (len(probs) - np.sum(probs, axis=0)) / np.sum(probs, axis=0)
    valid_result.update({'neg_ratio': neg_ratio})
    valid_result.update({'threshold': threshold})
    return valid_result

## pretraining backbone on single GPU
def Large_model_pretraining(args, device='cuda:0'):
    batch_size = 128 * 8
    dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_Ningbo', 'WFDB_ChapmanShaoxing']
    model_config = args.model_config

    if model_config == 'large':
        num_layers, complexity = 47, 768
    elif model_config == 'light':
        num_layers, complexity = 35, 128
    else:
        num_layers, complexity = 35, 256

    setup_seed(args.seed)
    dataset_train, dataset_valid = ECGcodedataset_loading(args=args, device=device)
    loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True)
    loader_valid = DataLoader(dataset=dataset_valid, batch_size=batch_size, shuffle=True)
    early_stopping = EarlyStopping(10, verbose=True, dataset_name=args.pretrain_dataset + model_config, delta=0,
                                   args=args)  # 15
    Epoch = args.pretrain_epoch
    num_leads = 12
    num_class = args.num_class
    net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=num_leads, num_layers=num_layers,
                     rank_list=0).to(device)
    print(count_parameters(net))

    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    net.train()
    for epoch in range(Epoch):
        # validate(net, loader_train, device,iftrain=True)
        valid_result = validate(net, loader_valid, device)
        early_stopping(1 / valid_result['Map_value'], net)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        running_loss = 0.0
        net.train()
        for i, (images, labels) in tqdm(enumerate(loader_train), total=len(loader_train)):
            images = images.float().to(device)
            labels = labels.float().to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            loss.backward()
            # allocated_memory = torch.cuda.memory_allocated(device)#max_
            optimizer.step()
            running_loss += loss.item()
            # print(f"GPU Memory Allocated: {allocated_memory / 1024 / 1024:.2f} MB")
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
    return
def count_parameters(model):
    # for n,p in model.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n and 'bias' not in n and n !='classifier.1.weight':
            # print(n)
            p.requires_grad = False
    return

# def mark_only_lora_as_trainable(model: nn.Module) -> None:
#     for n, p in model.named_parameters():
#         if 'lora_' not in n and 'bias' not in n:
#             # print(n)
#             p.requires_grad = False
#     return

def compute_information(loader_train, net, device):
    # device = str(net.device)
    net.train()
    grad_mat = []
    # net.freeze_A_grad()
    for i, (images, labels) in tqdm(enumerate(loader_train), total=len(loader_train)):
        net.zero_grad()
        images = images.float().to(device)
        labels = labels.float().to(device)
        outputs = net(images)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        loss.backward()
        # print(net.classifier[0].lora_B.grad)
        grad_mat.append(net.compute_grad())
    grad_mat = np.vstack(grad_mat)
    information = np.mean(grad_mat, axis=0)
    return information
def load_pretrained_model(net, path, args, device='cuda:0'):
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k.find('classifier.1') < 0}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net
def model_config_initialization(args, loader_train, device='cuda:0'):
    path = args.root + '/pretrained_checkpoint/'
    model_config = args.model_config
    if model_config == 'base':
        file_name_pretrain = args.pretrain_dataset + model_config + 'bias_checkpoint.pkl'
    else:
        file_name_pretrain = args.pretrain_dataset + model_config + 'bias_full_checkpoint.pkl'
    print(file_name_pretrain)
    setup_seed(args.seed)
    r = args.r#
    if model_config == 'base':
        num_layers, complexity = 35, 256
    elif model_config == 'medium':
        num_layers, complexity = 47, 512
    elif model_config == 'large':
        num_layers, complexity = 47, 768
    compress_ratio = args.compress_ratio
    num_leads = 12
    num_class = args.num_class
    print('current method double check:', args.ranklist)
    print('current rank double check:', r)
    if args.ranklist == 'lora_FastECG': ## One-shot Rank Allocation based on the importance of each low-rank matrix.
        args.r_low = r // 4 * 2
        net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=num_leads, num_layers=num_layers,
                         rank_list=r, information=args.information).to(device)
        if compress_ratio > 0:
            net = load_pretrained_model(net, path + file_name_pretrain, args, device=device)
            fisher_information = compute_information(loader_train, net, device)
            sorted_ranks = rankdata(fisher_information)
            lora_ranks = np.zeros_like(sorted_ranks)
            lora_ranks[sorted_ranks > np.floor(num_layers * compress_ratio)] = r
            lora_ranks[sorted_ranks <= np.floor(num_layers * compress_ratio)] = args.r_low
            del net
            torch.cuda.empty_cache()
            net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=num_leads, num_layers=num_layers,
                         rank_list=lora_ranks.astype(int), dropout_coef=args.dropout_coef)
            print('rank distribution:', lora_ranks)
        mark_only_lora_as_trainable(net)
    elif args.ranklist == 'lora_ave': ## equals LoRA
        net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=num_leads, num_layers=num_layers,
                         rank_list=r)
        mark_only_lora_as_trainable(net)
    else: ## equals Full fine-tuning
        net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=num_leads, num_layers=num_layers,rank_list=0)
    net.to(device)
    net = load_pretrained_model(net, path + file_name_pretrain, args, device=device)
    print(path + file_name_pretrain)
    if 'lora' in args.ranklist:
        params_to_update = []
        for name, param in net.named_parameters():
            if name.find('lora') > -1:
                params_to_update.append(param)
            elif name.find('bias') > -1:
                params_to_update.append(param)
            elif name.find('classifier.1.weight') > -1:
                params_to_update.append(param)
        optimizer = optim.AdamW(params_to_update, lr=args.learning_rate)
    else:
        optimizer = optim.AdamW(net.parameters(), lr=0.001)
    return net, optimizer
def get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)
def rank_distribution(net,method,r):
    conv_rank = []
    for layer in net.encoder_layers:
        sum_r = 0
        sum_r += layer.conv1.r
        sum_r += layer.conv2.r
        sum_r += layer.conv3.r
        ave_r = sum_r/3
        conv_rank.append(ave_r)
    transformers_rank = []
    for layer in net.transformer_layers:
        sum_r = 0
        sum_r += layer.fc1.r
        sum_r += layer.fc2.r
        sum_r += layer.self_attention.c_attn.r * 3
        ave_r = sum_r/5
        transformers_rank.append(ave_r)
    classifier_rank = []
    for layer in net.classifier:
        if layer.r > 0 :
            classifier_rank.append(layer.r)
    all_rank = [conv_rank,transformers_rank,classifier_rank]
    print(all_rank)
    return all_rank

def loading_lora_checkpoint(net, path,device='cuda:0'):
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = net.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    # net.load_state_dict(torch.load(path))
    return net

def pipeline_start_default_semi(args): ## pipeline for model fine-tuning on downstream datasets
    print('semiconfig:', args.semi_config)
    device = args.device
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    if args.finetune_label_ratio < 0.02:
        batch_size = 16
    else:
        batch_size = 64
    setup_seed(args.seed)
    dataset_train, dataset_ULtrain, dataset_valid, dataset_test, positive_weight, negative_weight = ECGdataset_prepare_finetuning_sepe_semi(args=args)
    unlabel_size = args.unlabel_amount_coeff * batch_size
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    loader_ULtrain = DataLoader(dataset_ULtrain, batch_size=unlabel_size, shuffle=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    label_iter = iter(loader_train)
    unlabel_iter = iter(loader_ULtrain)
    iteration = len(loader_train) * args.finetune_epoch
    print('max_iteration:', iteration)
    path = args.root + '/pretrained_checkpoint/'
    model_config = args.model_config
    save_name = 'ECG' + args.finetune_dataset + args.semi_config + model_config + args.ranklist + 'ratio' + str(
        args.finetune_label_ratio) + 'seed' + str(args.seed)
    print('save_name:',save_name)
    start_time = time.time()
    net, optimizer = model_config_initialization(args, loader_train, device=device)
    early_stopping = EarlyStopping(10, verbose=True,dataset_name=save_name,delta=0, args=args)  # 15
    step = 0
    net.train()
    setup_seed(args.seed)
    my_lr_scheduler = get_linear_schedule_with_warmup(optimizer,int(iteration*0.01), iteration, last_epoch=-1)
    if args.ranklist == 'lora_FastECG' and args.enable_dropout:
        net.network_rank_state_reset() ## enable random-deactivation
    running_loss = 0.0
    #count_parameters(net)
    for current in range(iteration):
        if current % args.interval == 0:
            print('training_loss:', running_loss/args.interval)
            running_loss = 0.0
            valid_result = validate(net, loader_valid, device)
            early_stopping(1 / valid_result['Map_value'], net)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        ## mini-batch sampling
        try:
            images, labels = next(label_iter)
        except Exception as err:
            label_iter = iter(loader_train)
            images, labels = next(label_iter)
        try:
            unlabel_images, _ = next(unlabel_iter)
        except Exception as err:
            unlabel_iter = iter(loader_ULtrain)
            unlabel_images, _ = next(unlabel_iter)
        if len(unlabel_images) != unlabel_size or len(labels) != batch_size:
            continue
        images = images.float().to(device)
        labels = labels.float().to(device)
        unlabel_images = unlabel_images.to(device)
        with torch.no_grad():
            images, labels = Cutmix(images, labels, device)
            aug_target_data_weak = tar_augmentation(unlabel_images, 'Weak', device)
        net.train()
        optimizer.zero_grad()
        if args.semi_config == 'nosemi':
            inputs = images
            outputs = net(inputs)
        else:
            inputs = torch.cat((images, aug_target_data_weak)) ## lightweight semi-supervised learning
            outputs = net(inputs, semi_flag=True)
        loss = F.binary_cross_entropy_with_logits(outputs, labels)
        #print(net.classifier[1].weight.shape)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        step += 1
        my_lr_scheduler.step()
    end_time = time.time()
    running_time = (end_time - start_time) / (current + 1)
    print(f"running time {running_time:.2f} 秒")
    allocated_memory = torch.cuda.max_memory_allocated(device)  # max_
    print(f"GPU Memory Allocated: {allocated_memory / 1024 / 1024:.2f} MB")
    print('load_name:', save_name + '_checkpoint.pkl')
    if args.ranklist == 'FT':
        net.load_state_dict(torch.load(path + save_name + '_checkpoint.pkl', map_location=device))
    else:
        net = loading_lora_checkpoint(net, path + save_name + '_checkpoint.pkl', device=args.device)
    trainable_num = count_parameters(net)
    all_rank = rank_distribution(net,args.ranklist,args.r)
    print('trainable_num:', trainable_num)
    net.eval()
    if args.ranklist == 'lora_FastECG':
        print('merging...')
        net.merge_net()
    with torch.no_grad():
        valid_result = validate(net, loader_valid, device)
        test_result = validate(net, loader_test, device, iftest=True, threshold=valid_result['threshold'])
        test_result.update({'trainable_num': trainable_num})
        test_result.update({'memory': allocated_memory})
        test_result.update({'time': running_time})
        test_result.update({'all_rank': all_rank})
    return test_result

def validate_ddp(model, valloader, device): ## model validation on multi-GPU cards
    model.eval()
    probs, lbls = [], []
    for step, (inp_windows_t, lbl_t) in enumerate(valloader):
        inp_windows_t, lbl_t = inp_windows_t.float().to(device), lbl_t.to(device)
        with torch.no_grad():
            out = model(inp_windows_t)
            loss = F.binary_cross_entropy_with_logits(out, lbl_t.float())
            prob = out.sigmoid().data
            probs.append(prob)
            lbls.append(lbl_t.data)
    lbls = torch.cat(lbls, dim=0)
    probs = torch.cat(probs, dim=0)
    metric = MultilabelAveragePrecision(num_labels=probs.shape[1], average="macro", thresholds=None)
    Map_value = metric(probs, lbls.int())
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    Map_value = reduce_value(Map_value)
    return Map_value

def Large_model_pretraining_ddp(args): ## pretraining the backbone on multi-GPU cards
    ## default training seed = 21
    print(args.ranklist)
    batch_size = 128 * 4
    setup()
    model_config = args.model_config
    rank = dist.get_rank()
    world_size = int(os.environ['WORLD_SIZE'])
    pid = os.getpid()
    print(f'current pid: {pid}')
    print(f'Current rank {rank}')
    device_id = rank % torch.cuda.device_count()
    device_id = torch.device(device_id)
    print(f'device_id {device_id}')
    if model_config == 'base':
        num_layers, complexity = 35, 256
        batch_size = 128 * 8
    elif model_config == 'tinylight':
        num_layers, complexity = 14, 64
        batch_size = 128 * 16
    elif model_config == 'tiny':
        num_layers, complexity = 23, 128
        batch_size = 128 * 16
    elif model_config == 'medium':
        num_layers, complexity = 47, 512
        batch_size = 128 * 8
    elif model_config == 'mediumv2':
        num_layers, complexity = 35, 512
        batch_size = 128 * 8
    elif model_config == 'large':
        num_layers, complexity = 47, 768
        batch_size = 128 * 4
    setup_seed(args.seed)
    dataset_train, dataset_valid = ECGcodedataset_loading()
    sampler = DistributedSampler(dataset_train)
    sampler_valid = DistributedSampler(dataset_valid)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, sampler=sampler, pin_memory=True)
    loader_valid = DataLoader(dataset_valid, batch_size=batch_size, sampler=sampler_valid, pin_memory=True)
    early_stopping = EarlyStopping(10, verbose=True, dataset_name=args.pretrain_dataset + model_config + 'bias_full',
                                   delta=0, args=args)  # 15
    Epoch = args.pretrain_epoch
    #iteration = Epoch * len(loader_train)
    num_leads = 12
    num_class = args.num_class
    net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=num_leads, num_layers=num_layers,
                     rank_list=0).to(device_id)
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(device_id)
    net = DistributedDataParallel(net, device_ids=[device_id])
    print(count_parameters(net))
    learning_rate = 0.001 * world_size * (batch_size / 1024)
    print('model_config:',model_config)
    print('learning_rate:', learning_rate)
    print('batch_size:', batch_size)
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.01)
    #my_lr_scheduler = get_linear_schedule_with_warmup(optimizer, int(iteration * 0.01), iteration, last_epoch=-1)
    net.train()
    for epoch in range(Epoch):
        sampler.set_epoch(epoch)
        running_loss = 0.0
        net.train()
        #for i, (images, labels) in tqdm(enumerate(loader_train), total=len(loader_train)):
        for i, (images, labels) in enumerate(loader_train):
            images = images.float().to(device_id)
            labels = labels.float().to(device_id)
            images, labels = Cutmix(images, labels, device_id)
            optimizer.zero_grad()
            outputs = net(images)
            loss = F.binary_cross_entropy_with_logits(outputs, labels)
            loss.backward()
            # allocated_memory = torch.cuda.memory_allocated(device)#max_
            optimizer.step()
            #my_lr_scheduler.step()
            running_loss += loss.item()
            # print(f"GPU Memory Allocated: {allocated_memory / 1024 / 1024:.2f} MB")
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
        if device_id != torch.device("cpu"):
            torch.cuda.synchronize(device_id)
        valid_result = validate_ddp(net, loader_valid, device_id)
        if rank == 0:
            print(f'epoch {epoch}, mAP {valid_result}')
            early_stopping(1 / valid_result, net.module)
        dist.barrier()
    return

def reduce_value(value, average=True):
    world_size = int(os.environ['WORLD_SIZE'])
    if world_size < 2:  # 单GPU的情况
        return value

    with torch.no_grad():
        dist.all_reduce(value)  # 对不同设备之间的value求和
        if average:  # 如果需要求平均，获得多块GPU计算loss的均值
            value /= world_size
    return value
