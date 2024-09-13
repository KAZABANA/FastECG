import numpy as np
import torch
import os

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self,patience=7, verbose=False,dataset_name='Ga',delta=0,args=None,path = '/pretrained_checkpoint'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            上次验证集损失值改善后等待几个epoch
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            如果是True，为每个验证集损失值改善打印一条信息
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            监测数量的最小变化，以符合改进的要求
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset_name=dataset_name
        self.args=args
        self.path = self.args.root + path
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''
        Saves model when validation loss decrease.
        验证损失减少时保存模型。
        '''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.chdir(self.path)
        if self.args.ranklist == 'FT':
            torch.save(model.state_dict(), self.dataset_name + '_checkpoint.pkl')
        else:
            saving_lora_checkpoint(model, self.dataset_name  + '_checkpoint.pkl')
        self.val_loss_min = val_loss

def saving_lora_checkpoint(net,path):
    net_state_dict = net.state_dict()
    saved_state_dict = {}
    for name, param in net_state_dict.items():
        if name.find('lora') > -1 or name.find('bias') > -1:
            saved_state_dict[name] = param
        elif name.find('classifier.1.weight') > -1:
            saved_state_dict[name] = param
        elif name.find('bn') > -1 or name.find('norm') > -1:
            saved_state_dict[name] = param
    torch.save(saved_state_dict, path)
    return 0