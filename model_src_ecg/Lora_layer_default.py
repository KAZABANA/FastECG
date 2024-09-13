# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:31:52 2023

@author: COCHE User
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import List
import random
def get_drop_state(dropout_value=0.2):
    r = random.random()  # 生成一个0到1之间的随机数
    if r < dropout_value:
        return 0
    else:
        return 1
def mean_squared_norm(matrix):
    mean_squared_sum = torch.mean(matrix**2)
    return mean_squared_sum

class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        information: str = 'fisher',
        dropout_coef: float = 0.2,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        self.dropout_coef = dropout_coef
        self.fan_in_fan_out = fan_in_fan_out
        self.in_features = in_features
        self.out_features = out_features
        self.para_nums = in_features * out_features
        self.information = information
        #self.weight.requires_grad = True
        self.lora_alpha = self.lora_alpha * r
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r,in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            self.enable_deactivation = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_B)

    def merge(self):
        if self.enable_deactivation:
            self.weight.data = self.weight.data +  (1- self.dropout_coef) * self.lora_B @ self.lora_A * self.scaling
        else:
            self.weight.data = self.weight.data + self.lora_B @ self.lora_A * self.scaling
            #print('no_l')
        self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r == 0 or self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        elif self.enable_deactivation and get_drop_state(self.dropout_coef) == 0:
            result = F.linear(x, T(self.weight), bias=self.bias)
            return result
        else:
            result = F.linear(x, T(self.weight) + self.lora_B @ self.lora_A * self.scaling, bias=self.bias)
            return result
    def estimate_grad(self):
        if self.r == 0:
            if self.information == 'fisher':
                estimated_w_grad = mean_squared_norm(self.weight.grad)
            else:
                estimated_w_grad = mean_squared_norm(torch.mul(self.weight,self.weight.grad))
        else:
            with torch.no_grad():
                #estimated_w_grad = mean_squared_norm((self.lora_B @ self.lora_A.grad) + (self.lora_B.grad @ self.lora_A))
                if self.information == 'fisher':
                    estimated_w_grad = mean_squared_norm(self.lora_B.grad @ self.lora_A)
                else:
                    estimated_w_grad = mean_squared_norm(torch.mul(self.weight,self.lora_B.grad @ self.lora_A))
        return estimated_w_grad.detach().cpu().numpy().item()
class ConvLoRA_split(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0., merge_weights=True, information='fisher',dropout_coef=0.2,**kwargs):
        super(ConvLoRA_split, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        self.dropout_coef = dropout_coef
        self.lora_alpha = self.lora_alpha * r
        # Actual trainable parameters
        if r > 0:
            if isinstance(kernel_size, int):
                self.in_features = in_channels * kernel_size
                self.out_features= out_channels//self.conv.groups*kernel_size
                self.lora_A = nn.Parameter(self.conv.weight.new_zeros((r, self.in_features)))
                self.lora_B = nn.Parameter(self.conv.weight.new_zeros((self.out_features, r)))
            else:
                self.in_features = in_channels * kernel_size[1]
                self.out_features= out_channels//self.conv.groups*kernel_size[0]
                self.lora_A = nn.Parameter(self.conv.weight.new_zeros((r, self.in_features)))
                self.lora_B = nn.Parameter(self.conv.weight.new_zeros((self.out_features, r)))
            self.scaling = self.lora_alpha / self.r
            self.conv.weight.requires_grad = False
            self.enable_deactivation = False
        self.reset_parameters()
        self.merged = False
        self.information = information

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A,B the same way as the default for nn.Linear 
            # and E (singular values) for zero 
            nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
            nn.init.zeros_(self.lora_B)

    def merge(self):
        if self.enable_deactivation:
            self.conv.weight.data = self.conv.weight.data + (1-self.dropout_coef) * (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
        else:
            self.conv.weight.data = self.conv.weight.data + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
        self.merged = True

    def forward(self, x):
        if self.r == 0 or self.merged:
            return self.conv(x)
        elif self.enable_deactivation and get_drop_state(self.dropout_coef) == 0:
            return self.conv(x)
        else:
            return self.conv._conv_forward(x,
                    self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                    self.conv.bias)

    def estimate_grad(self):
        if self.r == 0:
            if self.information == 'fisher':
                estimated_w_grad = mean_squared_norm(self.conv.weight.grad)
            else:
                estimated_w_grad =  mean_squared_norm(torch.mul(self.conv.weight,self.conv.weight.grad))
        else:
            with torch.no_grad():
                if self.information == 'fisher':
                    estimated_w_grad = mean_squared_norm(self.lora_B.grad @ self.lora_A)
                else:
                    estimated_w_grad =  mean_squared_norm(torch.mul(self.conv.weight,(self.lora_B.grad @ self.lora_A).view(self.conv.weight.shape)))
        return estimated_w_grad.detach().cpu().numpy().item()

class MergedLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
        enable_lora: List[bool] = [False],
        fan_in_fan_out: bool = False,
        merge_weights: bool = True,
        information : str = 'fisher',
        dropout_coef: float = 0.2,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        self.r = r
        self.out_features = out_features
        self.information = information
        self.dropout_coef = dropout_coef
        self.lora_alpha = self.lora_alpha * r
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            #self.lora_A.requires_grad = False
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            ) # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            #self.scaling = 1 / 32
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features, ), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
            self.enable_deactivation = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result
    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0), 
            self.lora_B.unsqueeze(-1), 
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))
    def estimate_grad(self):
        if self.r == 0:
            if self.information == 'fisher':
                estimated_w_grad = mean_squared_norm(self.weight.grad)
            else:
                estimated_w_grad = mean_squared_norm(torch.mul(self.weight,self.weight.grad))
        else:
            with torch.no_grad():
                if self.information == 'fisher':
                    estimated_w_grad = 0
                    hidden_dim = self.out_features // 3
                    for i in range(3):
                        if self.enable_lora[i]:
                            estimated_w_grad += mean_squared_norm(self.lora_B.grad[i*hidden_dim:(i+1) * hidden_dim,:]
                                                                  @ self.lora_A[i*self.r: (i+1) * self.r,:])
                    estimated_w_grad = estimated_w_grad/sum(self.enable_lora)
                else:
                    estimated_w_grad = 0
                    hidden_dim = self.out_features//3
                    flag = 0
                    for i in range(3):
                        if self.enable_lora[i]:
                            estimated_w_grad += mean_squared_norm(torch.mul(self.weight[:,i*hidden_dim:(i+1) * hidden_dim].T,
                                                    self.lora_B.grad[flag*hidden_dim:(flag+1) * hidden_dim,:]
                                                    @ self.lora_A[flag*self.r: (flag+1) * self.r,:]))
                            flag += 1
                    estimated_w_grad = estimated_w_grad/sum(self.enable_lora)
        return estimated_w_grad.detach().cpu().numpy().item()

    def merge(self):
        if self.enable_deactivation:
            self.weight.data = self.weight.data + (1-self.dropout_coef) * self.merge_AB() * self.scaling
            #print('d')
        else:
            self.weight.data = self.weight.data + self.merge_AB() * self.scaling
            #print('no_m')
        self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w
        if self.r == 0 or self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if not self.enable_deactivation or get_drop_state(self.dropout_coef) == 1:
                #print('sss')
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result
        
class Attention(nn.Module): # copy from https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/model.py
    def __init__(self, nx, n_ctx, config, dropout_coef, scale=False,information='fisher'):
        super(Attention, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        # n_ctx: seqence_length
        assert n_state % config['n_head'] == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config['n_head']
        self.split_size = n_state
        self.scale = scale
        self.c_attn = MergedLinear(
            nx, n_state * 3,
            r=config['r'],
            lora_alpha=config['lora_attn_alpha'],
            lora_dropout=config['lora_dropout'],
            enable_lora=[True, True, True],
            fan_in_fan_out=True,
            merge_weights=False,
            information=information,
            dropout_coef=dropout_coef
        )
        self.c_proj = Conv1D(n_state, nx)

        self.config = config
    
    def _attn(self, q, k, v, len_kv=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        #print(nd,ns)
        b = self.bias[:, :, ns-nd:ns, :ns]
        #print(w[0,:])
        w = w * b - 1e10 * (1 - b) # 前面的元素不能注意到后面的向量，所以强制其注意力分数为0
        #print(w[0,:])
        # q : (batch, head, q_seq_length, head_features)
        # k : (batch, head, head_features, kv_seq_length)
        # w : (batch, head, q_seq_length, kv_seq_length)
        # v : (batch, head, kv_seq_length, head_features)
        if len_kv is not None:
            # _len = torch.arange(96)， _len[None, :].unsqueeze(1).unsqueeze(2).shape
            _len = torch.arange(k.size(-1), device=k.device)
            _input_msk =  _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10) 

        w = nn.Softmax(dim=-1)(w)
        #print(w[0,0,:,:])
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1).contiguous()  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3).contiguous()  # (batch, head, seq_length, head_features)

    def forward(self, x, history=None, layer_past=None, len_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        #_input_msk = None

        len_kv = None

        if layer_past is not None:
            # key : (batch, head, head_features, seq_length)
            # value : (batch, head, seq_length, head_features)
            # layer_past, key : (batch, head, seq_length, head_features)
            if len_past is None:
                past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
                key = torch.cat((past_key, key), dim=-1)
                value = torch.cat((past_value, value), dim=-2)
            else:
                key_seq = key.shape[-1]
                assert key_seq == 1

                _batch = torch.arange(0, key.shape[0], dtype=torch.long, device=key.device)

                past_key, past_value = layer_past[0], layer_past[1]

                past_key[_batch,:,len_past,:] = key.squeeze(-1)
                past_value[_batch,:,len_past,:] = value.squeeze(-2)

                key = past_key.transpose(-2, -1)
                value = past_value

                len_kv = len_past + 1

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value, len_kv = len_kv)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)  # size_out = x.size()[:-1] + (nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight) # x = torch.addmm(bias, x.view(-1, x.size(-1)).shape, weight).shape
        x = x.view(*size_out) # x.view(*size_out).shape
        return x
    
class Conv2d(ConvLoRA_split):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)