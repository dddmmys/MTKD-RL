#!/usr/bin/env python

import os
import torch
import torch.nn as nn
import datetime
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset.cifar100 import get_cifar100_dataloaders
from models import model_dict
from train_loops import train_avg, test
from setting import  teacher_model_path_dict
from utils import set_logger


# add some args
def get_parsers():
    parser = argparse.ArgumentParser(description='PyTorch Feature Map Extract From Teacher(s) ')
    parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet', help='path to dataset (default: imagenet)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='mini-batch size (default: 256), this is the total ' 'batch size of all GPUs on the current node when ' 'using Data Parallel or Distributed Data Parallel')  # 32*2
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18_imagenet')
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100', 'imagenet', 'tinyimagenet', 'dogs', 'cub_200_2011', 'mit67'], help='dataset')
    parser.add_argument('--trial', type=str, default='1', help='trial id')
    parser.add_argument('--teacher-name-list', default=['resnet32x4', 'wrn_28_4'], type=str, nargs='+', help='teacher models')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--checkpoint-dir', default='./checkpoint', type=str, help='checkpoint directory')
    parser.add_argument('--n-cls', default=100, type=int, help='number of class.')
    parser.add_argument('--multiprocessing-distributed', default=False, type=bool, help='Use multi-processing distributed training to launch ' 'N processes per node, which has N GPUs. This is the ' 'fastest way to use PyTorch for either single node or ' 'multi node data parallel training')

    return parser

# load teacher(s)
def load_teacher(model_path, n_cls, model_t, opt=None):
    model = model_dict[model_t](num_classes=n_cls).cuda()
    map_location = None if opt.gpu is None else {'cuda:0': 'cuda:%d' % (opt.gpu if opt.multiprocessing_distributed else 0)}
    model.load_state_dict(torch.load(model_path, map_location=map_location)['model'])
    model.eval()
    for t_n, t_p in model.named_parameters():
        t_p.requires_grad = False
    return model
def load_teacher_list(opt):
    print('==> loading teacher model list')
    teacher_model_list = [load_teacher(teacher_model_path_dict[model_name], opt.n_cls, model_name, opt) for model_name in opt.teacher_name_list]
    print('==> done')
    return teacher_model_list

# get outputs of teacher layer
def get_single_layer_output(train_loader, device, teacher_models):
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)

        with torch.no_grad():
            for t_model in teacher_models:
                t_features, t_logits = t_model(inputs, is_feat=True)
                print("shape of features and logits: \n", t_features[-2].shape, t_logits.shape)
                d1, d2, d3, d4 = t_features[-2].shape
                features_2_3d = t_features[-2].view(d1, d3 * d4, d2)
                logits_3d = t_logits.flatten().view(5, 5, 256)
                print("shape of features_2_3d and logits_3d: \n", features_2_3d.shape, logits_3d.shape)
        break

def convert_intermediate_layer(features):
    batch_size, channels, height, width = features.shape
    # print("features.shape is ", features.shape)
    features_3d = features.permute(0, 2, 3, 1).contiguous()  # [64, 8, 8, 384]
    features_3d = features_3d.view(batch_size, height * width, channels)  # [64, 64, 384]
    return features_3d

def get_single_layer_output_from_inputs(inputs, device, teacher_models):
    inputs = inputs.to(device, non_blocking=True)
    with torch.no_grad():
        for t_model in teacher_models:
            t_features, t_logits = t_model(inputs, is_feat=True)
            batch_size = t_logits.shape[0]
            features_3_3d = convert_intermediate_layer(t_features[-2])
            logits_3d = t_logits.view(batch_size, 1, -1)
    return features_3_3d, logits_3d


def main():
    parser = get_parsers()
    args = parser.parse_args()
    args.teacher_name_str = "_".join(args.teacher_name_list)
    args.teacher_num = len(args.teacher_name_list)
    args.n_cls = 100
    args.multiprocessing_distributed = False
    args.model_name = args.arch + '_'+ args.dataset+ '_'+ 'rl'+'_'+ args.trial+'_'+str(args.teacher_num)+'_'+args.teacher_name_str
    info_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    info = args.model_name + info_time
    if args.rank == 0 :
        args.log_txt = os.path.join(args.checkpoint_dir, info + '.txt')
        log_dir = os.path.dirname(args.log_txt)
        os.makedirs(log_dir, exist_ok=True)
        args.logger = set_logger(args.log_txt)
        args.logger.info("==========\nArgs:{}\n==========".format(args))
    if torch.cuda.is_available():
        if args.gpu:
            device = torch.device('cuda:{}'.format(args.gpu))
        else:
            device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    # load dataset
    train_loader, val_loader = get_cifar100_dataloaders(data_folder=args.data, batch_size=args.batch_size, num_workers=args.workers, shuffle_train=False, use_augmentation=False)

    criterion_ce = nn.CrossEntropyLoss().to(device)
    teacher_models = load_teacher_list(args)
    t_results = []
    for t_model in teacher_models:
        acc = test(0, t_model, device, val_loader, criterion_ce, args, verbose=False)
        t_results.append(round(acc, 2))
    args.logger.info('Teacher accruacy: '+ str(t_results))
    get_single_layer_output(train_loader, device, teacher_models)

if __name__ == '__main__':
    main()
