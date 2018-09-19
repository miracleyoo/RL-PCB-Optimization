# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

from utils import *
from train import *
from config import Config
from tensorboardX import SummaryWriter
from models import policy_easy_net, policy_conv_net
import torch
import argparse
import warnings
import os

warnings.filterwarnings("ignore")


def main():
    folder_init(opt)
    pre_epoch = 0
    best_score = 1e-8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize model chosen
    try:
        if opt.MODEL == 'PolicyConvNet':
            policy = policy_conv_net.PolicyConvNet(opt)
        elif opt.MODEL == 'PolicyEasyNet':
            policy = policy_easy_net.PolicyEasyNet(opt)
    except KeyError('==> Your model is not found.'):
        exit(0)
    else:
        print("==> Model initialized successfully.")

    if opt.LOAD_SAVED_MOD:
        policy, pre_epoch, best_score = policy.load(map_location=device.type)
    policy.best_score = best_score
    policy.to_multi(device=device)

    # Instantiation of tensorboard and add net graph to it
    writer = SummaryWriter(opt.SUMMARY_PATH)
    dummy_input = wrap_np(gen_rand_data(opt), device)
    writer.add_graph(policy, dummy_input)

    # Start training or testing
    if not opt.MASS_TESTING:
        policy = training(opt, writer, policy, device, pre_epoch=pre_epoch)
        testing(opt, policy, device)
    else:
        steps = []
        for i in range(opt.TEST_EPOCH):
            with Timer(name='testing'):
                steps.append(testing(opt, policy, device))
        print("Average step: %d" % (sum(steps) // opt.TEST_EPOCH))


def str2bool(b):
    if b.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif b.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # Options
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('-lsm', '--LOAD_SAVED_MOD', type=str2bool,
                        help='If you want to load saved model')
    parser.add_argument('-und', '--USE_NEW_DATA', type=str2bool,
                        help='If you want to use new data')
    parser.add_argument('-gi', '--GPU_INDEX', type=str,
                        help='Index of GPUs you want to use')

    args = parser.parse_args()
    print(args)
    opt = Config()
    for k, v in vars(args).items():
        if v is not None and hasattr(opt, k):
            setattr(opt, k, v)
            print(k, v, getattr(opt, k))
    if args.GPU_INDEX:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_INDEX
    main()
