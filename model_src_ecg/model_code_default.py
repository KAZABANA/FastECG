# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:57:38 2023

@author: COCHE User
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from copy import deepcopy
import numpy as np
from typing import Optional
from torch.optim.optimizer import Optimizer
from .Lora_layer_default import *
import math
from torch.nn.functional import binary_cross_entropy_with_logits
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
def Cutmix(x, y, device, alpha=0.75):
    lam = np.random.beta(alpha, alpha)
    rand_index = torch.randperm(x.size()[0]).to(device)
    x = x.permute(0,2,1,3)
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    #print(bbx1, bby1, bbx2, bby2)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_mix = lam * y_a + (1 - lam) * y_b
    return  x.permute(0,2,1,3), y_mix

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim, max_len=200):
        super(PositionalEncoding, self).__init__()

        positional_encoding = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        x = x + self.positional_encoding[:x.size(1), :]
        return x

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads,rank_list, dropout_coef, seq_length=96,information='fisher'):
        super(TransformerLayer, self).__init__()
        config = {'n_head':num_heads, 'r':rank_list[0], 'lora_attn_alpha':1,'lora_dropout':0.0} 
        self.self_attention = Attention(hidden_dim, seq_length, config, scale=True,information=information, dropout_coef = dropout_coef)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
#        self.dropout = nn.Dropout(dropout)
        self.fc1 = Linear(hidden_dim, 4 * hidden_dim,r=rank_list[1],information=information, dropout_coef = dropout_coef)
        self.fc2 = Linear(4 * hidden_dim, hidden_dim,r=rank_list[2],information=information, dropout_coef = dropout_coef)
    
    def feed_forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def compute_grad_layer(self):
        grad_list=[self.self_attention.c_attn.estimate_grad(),self.fc1.estimate_grad(),self.fc2.estimate_grad()]
        return grad_list
    
    def freeze_A_grad_layer(self):
        self.self_attention.c_attn.lora_A.requires_grad = False
        self.fc1.lora_A.requires_grad = False
        self.fc2.lora_A.requires_grad = False
        return 
    
    def forward(self, x):
        x = x + self.self_attention(self.norm1(x))[0]
        x = x + self.feed_forward(self.norm2(x))
        return x

    def reset_rank_state(self):
        self.self_attention.c_attn.enable_deactivation = not self.self_attention.c_attn.enable_deactivation
        self.fc1.enable_deactivation = not self.fc1.enable_deactivation
        self.fc2.enable_deactivation = not self.fc2.enable_deactivation
        return

    def merge_layer(self):
        self.self_attention.c_attn.merge()
        self.fc1.merge()
        self.fc2.merge()
        return

class MyResidualBlock(nn.Module):
    def __init__(self,input_complexity,output_complexity,stride,downsample,rank_list,dropout_coef,information='fisher'):
        super(MyResidualBlock,self).__init__()
        self.downsample = downsample
        self.stride = stride if self.downsample else 1
        K = 9
        P = (K-1)//2
        self.conv1 = Conv2d(in_channels=input_complexity,
                               out_channels=output_complexity,
                               kernel_size=(1,K),
                               stride=(1,self.stride),
                               padding=(0,P),
                               bias=True,r=rank_list[0],information=information,dropout_coef=dropout_coef) #False
        self.bn1 = nn.BatchNorm2d(output_complexity)

        self.conv2 = Conv2d(in_channels=output_complexity,
                               out_channels=output_complexity,
                               kernel_size=(1,K),
                               padding=(0,P),
                               bias=True,r=rank_list[1],information=information,dropout_coef=dropout_coef) #False
        self.bn2 = nn.BatchNorm2d(output_complexity)

        if self.downsample:
            self.idfunc_0 = nn.AvgPool2d(kernel_size=(1,self.stride),stride=(1,self.stride))
            self.conv3 = Conv2d(in_channels=input_complexity,
                                      out_channels=output_complexity,
                                      kernel_size=(1,1),
                                      bias=True,r=rank_list[2],information=information,dropout_coef=dropout_coef) #False

    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))       
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.conv3(identity)
        x = x+identity
        return x
    
    def compute_grad_layer(self):
        grad_list=[self.conv1.estimate_grad(),self.conv2.estimate_grad(),self.conv3.estimate_grad()]
        return grad_list
    
    def freeze_A_grad_layer(self):
        self.conv1.lora_A.requires_grad = False
        self.conv2.lora_A.requires_grad = False
        self.conv3.lora_A.requires_grad = False
        return 

    def reset_rank_state(self):
        self.conv1.enable_deactivation = not self.conv1.enable_deactivation
        self.conv2.enable_deactivation = not self.conv2.enable_deactivation
        self.conv3.enable_deactivation = not self.conv3.enable_deactivation
        return

    def merge_layer(self):
        self.conv1.merge()
        self.conv2.merge()
        self.conv3.merge()
        return
class NN_default(nn.Module): ## backbone model definition
    def __init__(self,nOUT,complexity,inputchannel,num_layers=35,rank_list=32,information='fisher',num_encoder_layers=3,dropout_coef=0.2):
        super(NN_default,self).__init__()
        
        stride = 4
        #assert num_layers!=None
        self.num_layers = num_layers
        self.num_encoder_layers=num_encoder_layers
        self.num_classifier_layers=2
        assert (num_layers-3*self.num_encoder_layers-self.num_classifier_layers) % 3 == 0
        if isinstance(rank_list, int):
            self.rank_list = (np.zeros(num_layers)+rank_list).astype(int)
        else:
            self.rank_list = rank_list
            
        self.encoder_layers = nn.ModuleList(
                [MyResidualBlock(inputchannel, complexity, stride, downsample=True, rank_list=self.rank_list[0:3],information=information,dropout_coef=dropout_coef)])
        self.encoder_layers += nn.ModuleList(
                [MyResidualBlock(complexity, complexity, stride, downsample=True, rank_list=self.rank_list[3*(i+1):3*(i+2)],information=information,dropout_coef=dropout_coef) for i in range(self.num_encoder_layers-1)])
        
        self.classifier = nn.ModuleList(
            [Linear(complexity,complexity, r=self.rank_list[num_layers-2-i],merge_weights=False,information=information,dropout_coef=dropout_coef) for i in range(self.num_classifier_layers-1)]
        )
        self.classifier += nn.ModuleList([Linear(complexity,nOUT, r=0,merge_weights=False,information=information,dropout_coef=dropout_coef)])
        
        self.num_transformer_layers=self.num_layers-(3*self.num_encoder_layers+self.num_classifier_layers)
        complexity=complexity
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(complexity, num_heads=16,seq_length=96,rank_list=self.rank_list[3*(i+self.num_encoder_layers):3*(i+1+self.num_encoder_layers)],information=information,dropout_coef=dropout_coef) for i in range(self.num_transformer_layers // 3)]
        )
        
        self.position_encoding = PositionalEncoding(complexity)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.Pretrained = False
        
    def compute_grad(self):
        grad_list=[]
        for layer in self.encoder_layers:
            grad_list.extend(layer.compute_grad_layer())
            
        for layer in self.transformer_layers:
            grad_list.extend(layer.compute_grad_layer())
            
        for layer in self.classifier:
            if layer.r > 0:
                grad_list.extend([layer.estimate_grad()])
        
        return grad_list

    def network_rank_state_reset(self):
        for layer in self.encoder_layers:
            layer.reset_rank_state()

        for layer in self.transformer_layers:
            layer.reset_rank_state()

    def merge_net(self):
        for layer in self.encoder_layers:
            layer.merge_layer()

        for layer in self.transformer_layers:
            layer.merge_layer()

        for layer in self.classifier:
            if layer.r > 0:
                layer.merge()
        return

    def freeze_A_grad(self):
        for layer in self.encoder_layers:
            layer.freeze_A_grad_layer()
            
        for layer in self.transformer_layers:
            layer.freeze_A_grad_layer()
            
        for layer in self.classifier:
            layer.lora_A.requires_grad = False
        return 
    
    def feature_extraction(self, x, semi_flag = False):
        for layer in self.encoder_layers:
            x = layer(x)
        if semi_flag:
            #print(x.shape)
            x = x[0:len(x)//2,:,:,:]
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.position_encoding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(2)
        return x

    def forward(self, x, semi_flag = False):
        x = self.feature_extraction(x, semi_flag)

        for layer in self.classifier:
            x = layer(x)
        return x
def gaussian_noise(input, std,device):
    input_shape =input.size()
    noise = torch.normal(mean=0, std=std, size =input_shape)
    noise = noise.to(device)
    return input + noise

def filp_time(input):
    input=torch.flip(input,[3])
    return input

def filp_channel(input):
    rand_index=torch.randperm(input.shape[1])
    input=input[:,rand_index,:,:]
    return input

def dropout_burst(input):
    for i in range(input.shape[1]):
        length=np.random.randint(input.shape[3]/2)
        discard_start=np.random.randint(length,input.shape[3]-length)
        input[:,i,:,discard_start-length:discard_start+length]=0
    return input

def tar_augmentation(input, type,device):
    if type=='Weak':
        aug_type=np.random.randint(4)
        if aug_type == 0:
            input = filp_time(input)
        if aug_type == 1:
            input = dropout_burst(input)
        if aug_type == 2:
            input = gaussian_noise(input, 0.05, device)
        if aug_type == 3:
            input = filp_channel(input)
    elif type=='Strong':
        aug_list = [0,1,2,3]
        std = 0.5
        aug_que=np.unique(np.random.choice(aug_list, 4))
        np.random.shuffle(aug_que)
        for aug_type in aug_que:
            if aug_type == 0:
                input = filp_time(input)
            if aug_type == 1:
                input = dropout_burst(input)
            if aug_type == 2:
                input = gaussian_noise(input, std, device)
            if aug_type == 3:
                input = filp_channel(input)
    return input


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6
    def forward(self, inputs, targets):
        inputs = inputs.sigmoid()
        loss = -self.alpha * (1 - inputs) ** self.gamma * targets * torch.log(inputs+self.eps) \
               - (1 - self.alpha) * inputs ** self.gamma * (1 - targets) * torch.log(1 - inputs+self.eps)

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class DistributionBalancedLoss(nn.Module):
#https://github.com/tensorsense/faceflow/blob/d6059a28265d28392bde28c31f24ac9ec77901f0/lib/losses/db.py#L3
    def __init__(
        self,
        reduction="mean",
        pos_counts=None,
        neg_counts=None,
    ):
        super().__init__()

        self.reduction = reduction
        self.cls_criterion = binary_cross_entropy_with_logits

        # focal loss params
        self.gamma = 2.0
        self.balance_param = 2.0

        # mapping function params
        self.map_alpha = 0.1
        self.map_beta = 10.0
        self.map_gamma = 0.2

        self.pos_count = torch.from_numpy(pos_counts).float()
        self.neg_count = torch.from_numpy(neg_counts).float()
        self.num_classes = self.pos_count.shape[0]
        self.train_num = self.pos_count[0] + self.neg_count[0]

        # regularization params
        self.neg_scale = 2.0
        init_bias = 0.05

        self.init_bias = (
            -torch.log(self.train_num / self.pos_count - 1) * init_bias / self.neg_scale
        )
        self.freq_inv = torch.ones(self.pos_count.shape) / self.pos_count

    def forward(self, cls_score, label):
        cls_score = cls_score.clone()
        weight = self.rebalance_weight(label.float())
        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        # focal
        logpt = -self.cls_criterion(
            cls_score.clone(), label, weight=None, reduction="none"
        )
        # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
        pt = torch.exp(logpt)
        loss = self.cls_criterion(
            cls_score, label.float(), weight=weight, reduction="none"
        )
        loss = ((1 - pt) ** self.gamma) * loss
        loss = self.balance_param * loss

        # Check reduction option and return loss accordingly
        if self.reduction == "none":
            pass
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            self.reduction = loss.sum()

        return loss

    def logit_reg_functions(self, labels, logits, weight=None):
        self.init_bias = self.init_bias.to(logits)
        logits += self.init_bias
        logits = logits * (1 - labels) * self.neg_scale + logits * labels
        weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        self.freq_inv = self.freq_inv.to(gt_labels)
        repeat_rate = torch.sum(gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = (
            torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma))
            + self.map_alpha
        )
        return weight

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.sigmoid()

        intersection = (inputs * targets).sum(dim=1)
        union = inputs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1. - dice_score.mean()

        return dice_loss