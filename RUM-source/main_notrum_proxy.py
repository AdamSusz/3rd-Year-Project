import copy
import os
from collections import OrderedDict

import arg_parser
import evaluation
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, ConcatDataset, Subset
import unlearn
import utils
import numpy as np

# import pruner
from trainer import validate

from unlearn.impl import wandb_init, wandb_finish
from surgical_plugins.cluster import get_features, get_distance, get_fs, get_fs_dist_only
from surgical_plugins.overlap import calculate_FC, compute_diagonal_fisher_information

import subprocess

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def main():
    args = arg_parser.parse_args()

    # Common parameters
    dataset = args.dataset
    arch = args.arch
    data = args.data
    epochs = args.epochs
    lr = args.lr
    decreasing_lr = args.decreasing_lr
    batch_size = args.batch_size
    class_to_replace = -1
    proxy = args.mem_proxy

    seed = args.seed
    unlearn = args.unlearn
    group_index = None
    num_indexes_to_replace = 1000
    unlearn_step = args.unlearn_step
    beta = 0.9
    gamma = 0.9
    mstep = 2
    if unlearn == "SCRUB" and dataset == "cifar100" and arch == "vgg16_bn_lth":
        beta = 0.9
        gamma = 0.9
        mstep = 7
    if unlearn == "SCRUB" and dataset == "cifar100" and arch == "resnet18":
        beta = 0.9
        gamma = 0.9
        mstep = 5
    elif unlearn == "SCRUB" and dataset == "cifar10" and arch == "vgg16_bn_lth":
        beta = 0.1
        gamma = 0.0005
        mstep = 7
    elif unlearn == "SCRUB" and dataset == "cifar10" and arch == "resnet18":
        beta = 0.9
        gamma = 0.9
        mstep = 4

    if unlearn_step==1:
        original_model_path = f"assets/checkpoints/0{dataset}_original_{arch}_bs{batch_size}_lr{lr}_seed{seed}_epochs{epochs}.pth.tar"
    elif unlearn_step==2:
        if arch == 'resnet18':
            original_model_path = f"assets/unlearn/seq_mix/seq_mix_None_{dataset}_{arch}_[-1]_num3000_groupidNone_proxy{proxy}_mix_step1_seed{seed}.pth.tar"
        elif arch == 'vgg16_bn_lth' and unlearn == 'NG':
            original_model_path = f"assets/unlearn/seq_mix/seq_mix_None_{dataset}_{arch}_[-1]_num3000_groupidNone_proxy{proxy}_mix_step1_seed{seed}.pth.tar"
        elif arch == 'vgg16_bn_lth' and unlearn == 'SCRUB':
            original_model_path = f"assets/unlearn/seq_mix/seq_mix_None_{dataset}_{arch}_[-1]_num3000_groupidNone_proxy{proxy}_mix_step1_seed{seed}.pth.tar"
    else:
        original_model_path = f"assets/unlearn/seq_mix/seq_mix_None_{dataset}_{arch}_[-1]_num3000_groupidNone_proxy{proxy}_mix_step{unlearn_step-1}_seed{seed}.pth.tar"
    low_step_model_path = f"assets/unlearn/{unlearn}/{unlearn}_{unlearn}{unlearn}{unlearn}_{dataset}_{arch}_[-1]_num{num_indexes_to_replace}_groupidNone_proxy{proxy}_low_seqTrue_step{unlearn_step}_seed{seed}.pth.tar"
    mid_step_model_path = f"assets/unlearn/{unlearn}/{unlearn}_{unlearn}{unlearn}{unlearn}_{dataset}_{arch}_[-1]_num{num_indexes_to_replace}_groupidNone_proxy{proxy}_mid_seqTrue_step{unlearn_step}_seed{seed}.pth.tar"
    high_step_model_path = f"assets/unlearn/{unlearn}/{unlearn}_{unlearn}{unlearn}{unlearn}_{dataset}_{arch}_[-1]_num{num_indexes_to_replace}_groupidNone_proxy{proxy}_high_seqTrue_step{unlearn_step}_seed{seed}.pth.tar"

    # Define commands with dynamic mask parameters, below is an example for NegGrad+ --> NegGrad+ --> NegGrad+ RUM experiment (unlearning step 1)
    runs = [
        # NG -> 5 5 10
        # SCRUB -> 10 10 10
        # RUM: NegGrad+ --> NegGrad+ --> NegGrad+ (low-medium-high memorization order or the corresponding proxy order)

        f"python3.10 main_forget.py --seed {seed} --no_aug --sequential --mem_proxy {proxy} --mem mix --unlearn {unlearn} --unlearn_step {unlearn_step} --alpha 0.99 --unlearn_epochs 10 --unlearn_lr 0.01 --num_indexes_to_replace {num_indexes_to_replace} --class_to_replace {class_to_replace} --dataset {dataset} --arch {arch} --data {data} --epochs {epochs} --lr {lr} --decreasing_lr {decreasing_lr} --batch_size {batch_size} --wandb-mode offline --beta {beta} --gamma {gamma} --msteps {mstep} --kd_T 4.0",
     ]

    for command in runs:
        run_command(command)


if __name__ == "__main__":
    main()
