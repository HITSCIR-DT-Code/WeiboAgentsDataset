import argparse
import torch


def resolve_device(device='auto'):
    if device == 'auto':
        if torch.cuda.is_available():
            return 'cuda:0'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError(f'当前环境不支持 CUDA，但收到 device={device}')
        return device

    if device.startswith('mps'):
        if not torch.backends.mps.is_available():
            raise ValueError(f'当前环境不支持 MPS，但收到 device={device}')
        return 'mps'

    if device == 'cpu':
        return device

    raise ValueError(f'不支持的 device: {device}')


def parse_train_args(parser):
    parser.add_argument('--dataset_name',             type=str,   default='Weibo')
    parser.add_argument('--seed',                     type=int,   default=42)
    parser.add_argument('--device',                   type=str,   default='auto',
                        help='训练设备，支持 auto/cpu/cuda/cuda:0/mps')
    parser.add_argument('--interval',                 type=str,   default='month',
                        choices=['year', 'month', 'three_months', 'six_months',
                                 '9_months', '15_months', '18_months', '21_months', '24_months'])
    parser.add_argument('--early_stop',               action='store_true')
    parser.add_argument('--patience',                 type=int,   default=10)
    parser.add_argument('--coefficient',              type=float, default=1.1)
    parser.add_argument('--temporal_head_config',     type=int,   default=4)
    parser.add_argument('--structural_head_config',   type=int,   default=4)
    parser.add_argument('--batch_size',               type=int,   default=64)
    parser.add_argument('--hidden_dim',               type=int,   default=128)
    parser.add_argument('--temporal_drop',            type=float, default=0.5)
    parser.add_argument('--structural_drop',          type=float, default=0.0)
    parser.add_argument('--structural_learning_rate', type=float, default=1e-4)
    parser.add_argument('--temporal_learning_rate',   type=float, default=1e-5)
    parser.add_argument('--weight_decay',             type=float, default=1e-2)
    parser.add_argument('--epoch',                    type=int,   default=20)
    parser.add_argument('--window_size',              type=int,   default=-1)
    parser.add_argument('--temporal_module_type',     type=str,   default='attention',
                        choices=['attention', 'gru', 'lstm'])
    parser.add_argument('--weighted_loss',             action='store_true',
                        help='启用后使用训练集类别频率的倒数对 CrossEntropyLoss 加权，以缓解类别不平衡')
    return parser


def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    return args
