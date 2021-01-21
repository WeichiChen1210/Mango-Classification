import os
import argparse

import torch

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu-no', type=int,
                    help='cpu: -1, gpu: 0 ~ n ', default=0)

    parser.add_argument('--network', type=str,
                    help='the type of network', default='resnet18')

    # network type and pretrained or not
    parser.add_argument('--pretrain', dest='pretrained', action='store_true', help='use pretrained models')
    parser.add_argument('--scratch', dest='pretrained', action='store_false', help='train from scratch')
    parser.set_defaults(pretrained=False)

    parser.add_argument('--parallel', dest='dataparallel', action='store_true', help='use data parallel on multiple gpus')
    parser.add_argument('--single', dest='dataparallel', action='store_false', help='not use data parallel')
    parser.set_defaults(dataparallel=False)

    # hyper-parameters
    parser.add_argument('--start-epoch', type=int,
                    help='start epoch for training network', default=0)

    parser.add_argument('--epoch', type=int,
                    help='number of epoch for training network', default=150)
    
    parser.add_argument('--batchsize', type=int,
                    help='batch size', default=256)

    parser.add_argument('--num-workers', type=int,
                    help='number of workers for data loader', default=8)

    parser.add_argument('--lr', type=float,
                    help='initial learning rate', default=1e-2)

    parser.add_argument('--lr-milestone', type=list,
                    help='list of epoch for adjust learning rate', default=[83, 166, 240])

    parser.add_argument('--lr-gamma', type=float,
                    help='factor for decay learning rate', default=0.1)

    parser.add_argument('--momentum', type=float,
                    help='momentum for optimizer', default=0.9)

    parser.add_argument('--weight-decay', type=float,
                    help='factor for weight decay in optimizer', default=1e-4)

    parser.add_argument('--loss-coef', type=float,
                    help='loss-coef', default=0.005)

    parser.add_argument('--feature-loss-coef', type=float,
                    help='feature-loss-coef', default=0.01)
    
    parser.add_argument('--tri-coef', type=float,
                    help='coef for triplet loss', default=0.5)
    
    parser.add_argument('--ord-coef', type=float,
                    help='coef for ordinal loss', default=0.5)
    
    parser.add_argument('--temperature', type=float,
                    help='distillation temperature', default=3.0)

    parser.add_argument('--tri-margin', type=float,
                    help='triplet margin', default=2.0)
    
    parser.add_argument('--ord-margin', type=float,
                    help='ordinal margin', default=2.0)

    parser.add_argument('--ce-weight', type=list,
                    help='weights for CrossEntropy loss', default=None)

    parser.add_argument('--focal-gamma', type=float,
                    help='gamma for focal loss', default=0.5)
    
    parser.add_argument('--focal-alpha', type=list,
                    help='alpha for focal loss', default=[1.0, 1.0, 1.0])

    # paths
    parser.add_argument('--data-path', type=str,
                    help='path to dataset', default='./dataset/crop/')

    parser.add_argument('--load-path', type=str,
                    help='trained model load path', default=None)

    parser.add_argument('--save-path',type=str,
                    help='model save path', default='./trained_models/')

    return parser

def get_parameter():
    parser = build_parser()
    args = parser.parse_args()
    
    if args.dataparallel:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_no)
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
        print("Make dir: ",args.save_path)

    return args
