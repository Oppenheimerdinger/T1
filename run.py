import argparse
import os
import torch
import torch.backends
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='T1: Time Series Imputation')

    # basic config
    parser.add_argument('--task_name', type=str, default='imputation',
                        help='task name, options:[imputation, long_term_forecast, short_term_forecast, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='1: training, 0: testing')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, default='T1', help='model name')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s, t, h, d, b, w, m]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--data_source', type=str, default='tslib', choices=['tslib', 'benchpots', 'csdi'],
                        help='data source: tslib (CSV), benchpots (PhysioNet/PEMS), csdi (PM25)')

    # task parameters
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # T1 model specific
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size (number of variates)')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--n_heads', type=int, default=128, help='number of heads / embedding dimension')
    parser.add_argument('--patch_size', type=int, default=2, help='patch size for stem convolution')
    parser.add_argument('--patch_stride', type=int, default=1, help='patch stride for stem convolution')
    parser.add_argument('--n_blocks', type=int, nargs='+', default=[2, 2], help='number of blocks per stage')
    parser.add_argument('--kernel_size_large', type=int, nargs='+', default=[71, 31],
                        help='large kernel size per stage')
    parser.add_argument('--kernel_size_small', type=int, default=5, help='small kernel size')
    parser.add_argument('--ffn_ratio', type=float, default=1.0, help='FFN hidden ratio')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample stride between stages')
    parser.add_argument('--positional_encoding', action='store_true', default=True, help='use positional encoding')
    parser.add_argument('--no_positional_encoding', action='store_false', dest='positional_encoding')
    parser.add_argument('--qkv_bias', action='store_true', default=True, help='QKV projection bias')
    parser.add_argument('--cosine_attention', action='store_true', default=False, help='use cosine attention')
    parser.add_argument('--use_head_reconstruction', action='store_true', default=True, help='use ReconHead')
    parser.add_argument('--no_head_reconstruction', action='store_false', dest='use_head_reconstruction')
    parser.add_argument('--head_params_shared', action='store_true', default=True, help='share head parameters')
    parser.add_argument('--imputation_use_mask_embedding', action='store_true', default=False,
                        help='use mask as additional input channel for imputation')

    # dropout
    parser.add_argument('--drop_attn', type=float, default=0.0, help='attention dropout')
    parser.add_argument('--drop_ffn', type=float, default=0.0, help='FFN dropout')
    parser.add_argument('--drop_proj', type=float, default=0.0, help='projection dropout')
    parser.add_argument('--drop_path', type=float, default=0.0, help='drop path rate')
    parser.add_argument('--drop_head', type=float, default=0.0, help='head dropout')
    parser.add_argument('--dropout', type=float, default=0.1, help='general dropout (TSLib compat)')

    # SACA
    parser.add_argument('--SACA', action='store_true', default=False, help='enable SACA module')
    parser.add_argument('--SACA_first_only', action='store_true', default=False)
    parser.add_argument('--saca_hidden_dim_ratio', type=int, default=8)
    parser.add_argument('--drop_saca', type=float, default=0.0)
    parser.add_argument('--sig_q_mode', type=int, default=1)
    parser.add_argument('--sig_k_mode', type=int, default=1)
    parser.add_argument('--sig_v_mode', type=int, default=1)
    parser.add_argument('--mu_q_mode', type=int, default=1)
    parser.add_argument('--mu_k_mode', type=int, default=1)
    parser.add_argument('--mu_v_mode', type=int, default=1)

    # normalization (for TSLib compatibility)
    parser.add_argument('--use_SAN', action='store_true', default=False)
    parser.add_argument('--use_FAN', action='store_true', default=False)

    # TSLib compatibility args (used by exp_basic / other models)
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--use_norm', type=int, default=1, help='use normalization')

    # optimization
    parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=300, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--ORT_weight', type=float, default=0.0,
                        help='ORT (observed reconstruction task) loss weight')
    parser.add_argument('--MIT_weight', type=float, default=1.0,
                        help='MIT (masked imputation task) loss weight')
    parser.add_argument('--base_loss', type=str, default='MSE', choices=['MSE', 'MAE'],
                        help='base loss function for imputation (MSE or MAE)')
    parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision', default=False)
    parser.add_argument('--precision', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16'],
                        help='training precision (default: bf16)')

    # GPU
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu')
    parser.add_argument('--no_use_gpu', action='store_false', dest='use_gpu', help='disable gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type: cuda or mps')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    args = parser.parse_args()

    # Device setup
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU: cuda:{}'.format(args.gpu))
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
        else:
            args.device = torch.device("cpu")
        print('Using device: {}'.format(args.device))

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    from utils.print_args import print_args
    print_args(args)

    # Import experiment class
    from exp.exp_imputation import Exp_Imputation
    Exp = Exp_Imputation

    if args.is_training:
        for ii in range(args.itr):
            # Setting record of experiments
            setting = '{}_{}_{}_sl{}_pl{}_nh{}_ps{}_mr{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.seq_len,
                args.pred_len,
                args.n_heads,
                args.patch_size,
                args.mask_rate,
                ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp = Exp(args)
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            if args.use_gpu:
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_sl{}_pl{}_nh{}_ps{}_mr{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.seq_len,
            args.pred_len,
            args.n_heads,
            args.patch_size,
            args.mask_rate,
            ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp = Exp(args)
        exp.test(setting, test=1)

        if args.use_gpu:
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
