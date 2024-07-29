# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 14:57:38 2023

@author: COCHE User
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
from sklearn.preprocessing import StandardScaler
import scipy.signal as sig

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
    x = x.permute(0, 2, 1, 3)
    y_a = y
    y_b = y[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    # print(bbx1, bby1, bbx2, bby2)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    y_mix = lam * y_a + (1 - lam) * y_b
    return x.permute(0, 2, 1, 3), y_mix

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = Parameter(w)
        self.bias = Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)  
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight) 
        x = x.view(*size_out) 
        return x

class Attention(nn.Module):  # implementation from https://github.com/microsoft/LoRA/blob/main/examples/NLG/src/model.py
    def __init__(self, nx, n_ctx, config, scale=False):
        super(Attention, self).__init__()
        n_state = nx
        assert n_state % config['n_head'] == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = config['n_head']
        self.split_size = n_state
        self.scale = scale
        self.c_attn = nn.Linear(nx, n_state * 3)
        self.c_proj = Conv1D(n_state, nx)

        self.config = config

    def _attn(self, q, k, v, len_kv=None):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns - nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)  # 前面的元素不能注意到后面的向量，所以强制其注意力分数为0
        if len_kv is not None:
            # _len = torch.arange(96)， _len[None, :].unsqueeze(1).unsqueeze(2).shape
            _len = torch.arange(k.size(-1), device=k.device)
            _input_msk = _len[None, :] >= (len_kv)[:, None]
            w = w.masked_fill(_input_msk.unsqueeze(1).unsqueeze(2), -1.0e10)

        w = nn.Softmax(dim=-1)(w)
        # print(w[0,0,:,:])
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

        # _input_msk = None

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

                past_key[_batch, :, len_past, :] = key.squeeze(-1)
                past_value[_batch, :, len_past, :] = value.squeeze(-2)

                key = past_key.transpose(-2, -1)
                value = past_value

                len_kv = len_past + 1

        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value, len_kv=len_kv)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        return a, present

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
    def __init__(self, hidden_dim, num_heads, seq_length=96):
        super(TransformerLayer, self).__init__()
        config = {'n_head': num_heads}
        self.self_attention = Attention(hidden_dim, seq_length, config, scale=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        #        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.fc2 = nn.Linear(4 * hidden_dim, hidden_dim)

    def feed_forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = x + self.self_attention(self.norm1(x))[0]
        x = x + self.feed_forward(self.norm2(x))
        return x



class MyResidualBlock(nn.Module): ## implementation from  Nejedly, P. et al. Classification of ECG using ensemble of residual CNNs with attention mechanism
    def __init__(self, input_complexity, output_complexity, stride, downsample):
        super(MyResidualBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride if self.downsample else 1
        K = 9
        P = (K - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=input_complexity,
                            out_channels=output_complexity,
                            kernel_size=(1, K),
                            stride=(1, self.stride),
                            padding=(0, P),
                            bias=True)  # False
        self.bn1 = nn.BatchNorm2d(output_complexity)

        self.conv2 = nn.Conv2d(in_channels=output_complexity,
                            out_channels=output_complexity,
                            kernel_size=(1, K),
                            padding=(0, P),
                            bias=True)  # False
        self.bn2 = nn.BatchNorm2d(output_complexity)

        if self.downsample:
            self.idfunc_0 = nn.AvgPool2d(kernel_size=(1, self.stride), stride=(1, self.stride))
            self.conv3 = nn.Conv2d(in_channels=input_complexity,
                                out_channels=output_complexity,
                                kernel_size=(1, 1),
                                bias=True)  # False

    def forward(self, x):
        identity = x
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if self.downsample:
            identity = self.idfunc_0(identity)
            identity = self.conv3(identity)
        x = x + identity
        return x



class NN_default(nn.Module):  ## backbone model definition
    def __init__(self, nOUT, complexity, inputchannel, num_layers=35,num_encoder_layers=3):
        super(NN_default, self).__init__()

        stride = 4
        # assert num_layers!=None
        self.num_layers = num_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_classifier_layers = 2
        assert (num_layers - 3 * self.num_encoder_layers - self.num_classifier_layers) % 3 == 0

        self.encoder_layers = nn.ModuleList(
            [MyResidualBlock(inputchannel, complexity, stride, downsample=True)])
        self.encoder_layers += nn.ModuleList(
            [MyResidualBlock(complexity, complexity, stride, downsample=True) for i in range(self.num_encoder_layers - 1)])

        self.classifier = nn.ModuleList(
            [nn.Linear(complexity, complexity) for i in range(self.num_classifier_layers - 1)]
        )
        self.classifier += nn.ModuleList(
            [nn.Linear(complexity, nOUT)])

        self.num_transformer_layers = self.num_layers - (3 * self.num_encoder_layers + self.num_classifier_layers)
        complexity = complexity
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(complexity, num_heads=16, seq_length=96) for i in
             range(self.num_transformer_layers // 3)]
        )

        self.position_encoding = PositionalEncoding(complexity)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.Pretrained = False

    def feature_extraction(self, x, semi_flag=False):
        for layer in self.encoder_layers:
            x = layer(x)
        if semi_flag:
            # print(x.shape)
            x = x[0:len(x) // 2, :, :, :]
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.position_encoding(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(2)
        return x

    def forward(self, x, semi_flag=False):
        x = self.feature_extraction(x, semi_flag)

        for layer in self.classifier:
            x = layer(x)
        return x

def scale_signal(x):
    """ scale ecg signal """
    for i in range(len(x)):
        scaler = StandardScaler()
        scaler.fit(np.expand_dims(x[i,:], 1))
        x[i,:] = scaler.transform(np.expand_dims(x[i,:], 1)).squeeze()
    return x

def preprocess_signal(x, sample_rate,max_length=9000):
    """ resample, filter, scale, ecg signal """
    x = filter_signal(x, sample_rate)
    denoised_signal = scale_signal(x)
    return denoised_signal


def filter_signal(x, sample_rate):
    """ filter ecg signal """
    nyq = sample_rate * 0.5
    filter_bandpass = [1.0,47.0]
    for i in range(len(x)):
        x[i,:] = sig.filtfilt(*sig.butter(3, [filter_bandpass[0] / nyq, filter_bandpass[1] / nyq], btype='bandpass'), x[i,:])
    return x

def load_pretrained_model(net, path, device='cpu',discard_last = True):
    pretrained_dict = torch.load(path, map_location=device)
    model_dict = net.state_dict()
    if discard_last: ## random initialize the last output layer
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k.find('classifier.1') < 0}
        key_list = np.copy(list(pretrained_dict.keys()))
        for k in key_list:
            if 'c_attn.weight' in k:
                pretrained_dict[k].data = pretrained_dict[k].data.transpose(0,1)
            if '.conv.bias' in k :
                new_keyname = k[0:-9]+'bias'
                pretrained_dict[new_keyname] = pretrained_dict.pop(k)
            if '.conv.weight' in k :
                new_keyname = k[0:-11]+'weight'
                pretrained_dict[new_keyname] = pretrained_dict.pop(k)
    else:## using the pretrained last output layer, num_of_class = 6
        key_list = np.copy(list(pretrained_dict.keys()))
        for k in key_list:
            if 'c_attn.weight' in k:
                pretrained_dict[k].data = pretrained_dict[k].data.transpose(0,1)
            if '.conv.bias' in k :
                new_keyname = k[0:-9]+'bias'
                pretrained_dict[new_keyname] = pretrained_dict.pop(k)
            if '.conv.weight' in k :
                new_keyname = k[0:-11]+'weight'
                pretrained_dict[new_keyname] = pretrained_dict.pop(k)
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)
    return net

def model_inference_pipeline(device_id='cpu'): ## run this function 
    num_layers, complexity,num_leads = 35, 256, 12
    num_class = 6
    net = NN_default(nOUT=num_class, complexity=complexity, inputchannel=num_leads, num_layers=num_layers).to(device_id)
    net = load_pretrained_model(net, path='D:\\FastECG_inference\\base_checkpoint.pkl',device=device_id,discard_last=False)
    random_signal = np.random.rand(12,6144).astype('float32')
    random_signal = preprocess_signal(random_signal, sample_rate=500)## sampling rate 400 or 500
    random_input = torch.from_numpy(random_signal[np.newaxis,:,np.newaxis,:]) ## (batch,leads,1,length), length = 6144,4096,....,should be divided by 16
    output = net(random_input)
    pred = output.sigmoid()
    return pred
    
def gaussian_noise(input, std, device):
    input_shape = input.size()
    noise = torch.normal(mean=0, std=std, size=input_shape)
    noise = noise.to(device)
    return input + noise


def filp_time(input):
    input = torch.flip(input, [3])
    return input


def filp_channel(input):
    rand_index = torch.randperm(input.shape[1])
    input = input[:, rand_index, :, :]
    return input


def dropout_burst(input):
    for i in range(input.shape[1]):
        length = np.random.randint(input.shape[3] / 2)
        discard_start = np.random.randint(length, input.shape[3] - length)
        input[:, i, :, discard_start - length:discard_start + length] = 0
    return input


def tar_augmentation(input, type, device):
    if type == 'Weak':
        aug_type = np.random.randint(4)
        if aug_type == 0:
            input = filp_time(input)
        if aug_type == 1:
            input = dropout_burst(input)
        if aug_type == 2:
            input = gaussian_noise(input, 0.05, device)
        if aug_type == 3:
            input = filp_channel(input)
    elif type == 'Strong':
        aug_list = [0, 1, 2, 3]
        std = 0.5
        aug_que = np.unique(np.random.choice(aug_list, 4))
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


