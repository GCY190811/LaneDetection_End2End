"""
for test usage
"""

import numpy as ny
import torch
import torch.optim
import torch.nn as nn

import os
import glob
import time
import sys
import shutil
import json
from tqdm import tqdm
from Dataloader.Load_Data_new import get_loader, get_homography, load_valid_set_file_all, write_lsq_results
from eval_lane import LaneEval
from Loss_crit import define_loss_crit, polynomial
from Networks.LSQ_layer import Net
from Networks.utils import define_args, save_weightmap, first_run, mkdir_if_missing, Logger, define_init_weights, define_scheduler, define_optim, AverageMeter


def main():
    global args
    global mse_policy
    parser = define_args()
    args = parser.parse_args()
    mse_policy = args.loss_policy == 'homography_mse'
    if args.clas:
        assert args.nclasses == 4

    print("==>>check args.no_cuda".format(args.no_cuda))
    if not args.no_cuda and not torch.cuda.is_available():
        raise Exception("No gpu available for usage")
    # 在运行前进行测试，选择最好的卷积计算方法，提升训练速率，适用于网络及输入尺寸不变的情况。
    torch.backends.cudnn.benchmark = args.cudnn

    # 保持图片比例，高固定为256.
    M_inv = get_homography(args.resize)  # ?

    # Dataloader for test set
    # ToDo: get_loader
    test_loader = get_loader(args.json_file,
                             args.image_dir,
                             args.batch_size,
                             num_workers=args.nworkers,
                             end_to_end=args.end_to_end,
                             resize=args.resize)

    # Define network
    model = Net(args)
    define_init_weights(model, args.weight_init)

    if not args.no_cuda:
        model = model.cuda()

    


if __name__ == "__main__":
    main()
