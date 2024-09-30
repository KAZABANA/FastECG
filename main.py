from pipeline_ecg import Large_model_pretraining_ddp,pipeline_start_default_semi
import os
import argparse
import numpy as np
import warnings
def ECG_config(seed,root):
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_config', type=str, default='base') ## Determine the backbone size (base, mediun, large)
    parser.add_argument('--semi_config', type=str, default='default') ## Whether or not to enable the lightweight semi-supervised learning module.
    parser.add_argument('--enable_dropout', type=bool, default=True) ## Whether or not to enable the random deactivation module.
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--ranklist', type=str, default='lora_FastECG') ## Set ranklist == lora_FastECG will activate the one-shot rank allocation module
    parser.add_argument('--root', type=str, default=root)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--r_low', type=int, default=2)
    parser.add_argument('--pretrain_epoch', type=int, default=50)
    parser.add_argument('--finetune_epoch', type=int, default=200)  # 150
    parser.add_argument('--compress_ratio', type=float, default=0.9) ## A hyper-parameter to control the number of important low-rank matrices during model training.
    parser.add_argument('--information', type=str, default='taylor')  # using taylor's formula to estimate the importance of differetn low-rank matrices.
    parser.add_argument('--num_class', type=int, default=25)
    parser.add_argument('--finetune_label_ratio', type=float, default=0.05)#0.05
    parser.add_argument('--mode', type=str, default='finetune')
    parser.add_argument('--pretrain_dataset', type=str, default='CODE_')
    parser.add_argument('--finetune_dataset', type=str, default='WFDB_ChapmanShaoxing')
    # dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_Ningbo', 'WFDB_ChapmanShaoxing']
    parser.add_argument('--r', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--unlabel_amount_coeff', type=float, default=1)
    parser.add_argument('--dropout_coef', type=float, default=0.2) ## Random deactivation probability.
    args = parser.parse_args()
    return args

def exp_main(args):
    dataset_list = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_Ningbo', 'WFDB_ChapmanShaoxing']
    num_class_list = [18, 19, 23, 16]
    method = 'lora_FastECG'
    print('current method:', method)
    args.ranklist = method
    save_file_name = ('result_FastECG' + '_ratio' +str(int(100 * args.finetune_label_ratio))
                      + '_rank' + str(args.r) + '_seed' + str(args.seed)
                      + '_' + args.model_config  + '.npy')
    print(save_file_name)
    if args.r == 16 :
        compress_ratio = [0.6,0.9,0.9,0.6]
    else:
        compress_ratio = [0.2,0.8,0.7,0.1]
    result_dataset = []
    for i in range(4):
        args.ranklist = method
        args.finetune_dataset = dataset_list[i]
        args.num_class = num_class_list[i]
        args.compress_ratio = compress_ratio[i]
        args.learning_rate = 1e-3
        print(args.finetune_dataset)
        print('compress ratio:', args.compress_ratio)
        print('labeled_ratio:', args.finetune_label_ratio)
        print('learning_rate:', args.learning_rate)
        result_dataset.append(pipeline_start_default_semi(args=args))
    os.chdir(args.root + '/result')
    np.save(save_file_name, result_dataset)

def Task_ECG(seed, root):
    args = ECG_config(seed, root)
    #dataset_organize(args) ## If you want to preprocess the downstream datasets, please uncomment this function
    print('seed:', args.seed)
    print('device:', args.device)
    print('model_config:', args.model_config)
    args.finetune_dataset = args.finetune_dataset
    if args.mode == 'pretrain': ## Usually, you don't need to pre-train the backbone by yourself.
        args.ranklist = 'FT'
        args.num_class = 6
        Large_model_pretraining_ddp(args)
    else:
        r_list = [16,4]
        print(r_list)
        for r in r_list:
            args.r = r
            args.model_config = 'base'
            #args.semi_config = 'nosemi'
            args.semi_config = 'default'
            args.dropout_coef = 0.2
            #args.enable_dropout = False
            exp_main(args)

if __name__ == '__main__':
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,2'
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    warnings.filterwarnings("ignore", message="divide by zero encountered in divide")
    warnings.filterwarnings("ignore", category=FutureWarning)
    Task = 'ECG'
    root = os.getcwd()
    Task_ECG(18, root)
