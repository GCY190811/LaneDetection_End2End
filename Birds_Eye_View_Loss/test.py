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


def test(loader, model, M_inv):
    model.eval()
    with torch.no_grad():
        # Start validation loop
        for i, (input, idx, index) in tqdm(enumerate(loader)):
            if not args.no_cuda:
                input = input.cuda(non_blocking=True)
                input = input.float()

            # Evaluate model
            try:
                beta0, beta1, beta2, beta3, weightmap_zeros, M, \
                output_net, outputs_line, outputs_horizon = model(input, True)  # end_to_end
            except RuntimeError as e:
                print("Batch with idx {} skipped due to singular matrix".format(idx.numpy()))
                print(e)
                continue

            # Write predictions to json file
            num_el = input.size(0)
            params_batch = torch.cat((beta0, beta1, beta2, beta3), 2).transpose(1, 2).data.tolist()
            
            _, line_pred = torch.max(outputs_line, 1)
            line_type = line_pred.data.tolist()

            horizon_pred = torch.round(nn.Sigmoid()(outputs_horizon))
            horizon_pred = horizon_pred.data.tolist()

            with open(test_set_path, 'w') as jsonFile:
                for j in range(num_el):
                    im_id = index[j]
                    json_line = valid_set_labels[im_id]
                    line_id = line_type[j]
                    horizon_est = horizon_pred[j]
                    params = params_batch[j]
                    json_line["params"] = params
                    json_line["line_id"] = line_id
                    json_line["horizon_est"] = horizon_est
                    json.dump(json_line, jsonFile)
                    jsonFile.write('\n')


            # # Plot weightmap and curves
            # if (i + 1) % 25 == 0:
            #     save_weightmap('valid', M, M_inv,
            #                    weightmap_zeros, beta0, beta1, beta2, beta3,
            #                    gt0, gt1, gt2, gt3, line_pred, gt, 0, i, input,
            #                    args.no_ortho, args.resize, args.save_path)
        return
    

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
    test_loader = get_testloaderV1(args.json_file,
                                   trainsform=args.resize,
                                   batch_size=args.batch_size,
                                   num_workers=args.nworkers,
                                   path=args.image_dir)

    # network
    model = Net(args)
    if not args.no_cuda:
        model = model.cuda()

    global test_set_path
    test_set_path = os.path.join(args.save_path, 'validation_set_dst.json')

    best_file_name = glob.glob(os.path.join(args.save_path, 'model_best*'))[0]
    if os.path.isfile(best_file_name):
        sys.stdout = Logger(os.path.join(args.save_path, 'Evaluate.txt'))
        print("=> loading checkpoint '{}'".format(best_file_name))
        checkpoint = torch.load(best_file_name)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(best_file_name))
    test(test_loader, model, M_inv)
    return


if __name__ == "__main__":
    main()
