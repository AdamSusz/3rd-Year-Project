import copy
import argparse
import sys
import os
from collections import OrderedDict
import shutil
import time
import numpy as np
import pandas as pd
import random
import re
import csv
import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet, ImageFolder
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split, ConcatDataset, Subset
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.autonotebook import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torch.nn._reduction')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import utils
from utils import *
from scipy.stats import norm
from scipy.stats import pearsonr,spearmanr
import scipy.stats as stats
import arg_parser
import unlearn.impl
from collections import OrderedDict
from trainer import train, validate
from unlearn.impl import wandb_init, wandb_finish
import wandb

def main():
    global args, best_sa
    args = arg_parser.parse_args()
    print(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")
    args.device = device

    if args.unlearn == 'RL':
        args.save_dir = 'assets/unlearn/salun'
    elif args.unlearn == 'seq_mix':
        args.save_dir = 'assets/unlearn/seq_mix'
    else:
        args.save_dir = f'assets/unlearn/{args.unlearn}'
    print('args.save_dir = ', args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)

    args.wandb_group_name = f"{args.dataset}_{args.arch}_analysis"
    logger = wandb_init(args)
    files_to_save = []


    seeds = [1374, 1522, 652379, 1792, 532]
    class_to_replace_configs = [[-1]]
    group_ids = ['mix']
    results = OrderedDict(((config,), []) for config in group_ids)
    results_rt = OrderedDict(((config,), []) for config in group_ids)
    df_dict = {}

    for class_to_replace in class_to_replace_configs:
        for group_id in group_ids:
            if args.mem is not None:
                args.mem = group_id
            else:
                args.group_index = group_id
            all_dfs = []
            for seed in seeds:
                args.seed = seed
                args.class_to_replace = class_to_replace
                if args.seed:
                    setup_seed(args.seed)
                args.train_seed = args.seed
                print(f'------seed: {args.seed}, class to forget: {args.class_to_replace}------')

                print('Loading results for approx. unlearning...')
                if args.unlearn == 'seq_mix':
                    if args.mem_proxy is not None:
                        if args.unlearn_step == 1 or args.unlearn_step is None:
                            filename = (f'{args.unlearn}_None_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_'
                                        f'groupidNone_proxy{args.mem_proxy}_{args.mem}_step{args.unlearn_step}_seed{args.seed}.pth.tar')
                        elif args.unlearn_step > 1:
                            filename = (f'{args.unlearn}_None_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_'
                                        f'groupidNone_proxy{args.mem_proxy}_{args.mem}_step{args.unlearn_step}_seed{args.seed}.pth.tar')

                    else:
                        filename = '{}_NGNGNG_{}_{}_{}_num{}_groupidNone_mem{}_seed{}.pth.tar'.format(args.unlearn,
                            args.dataset,args.arch,args.class_to_replace,args.num_indexes_to_replace,args.mem,args.seed)
                    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn, filename=filename)
                    eval_results = checkpoint.get("evaluation_result")
                    print(f'evaluation results for seed {args.seed} and class:{args.class_to_replace}:')
                    print(eval_results)
                    acc = eval_results['accuracy']
                    df = pd.DataFrame(acc, index=[0])
                    mia = eval_results['SVC_MIA_forget_efficacy']
                    mia_df = pd.DataFrame(mia, index=[0])
                    mia_df = mia_df.add_prefix('MIA_')
                    df = pd.concat([df, mia_df], axis=1)
                    result_dict = {f'acc_{k}': v for k, v in acc.items()}
                    result_dict.update({f'mia_{k}': v for k, v in mia.items()})
                    results[(group_id,)].append(result_dict)

                    print('check 1')
                else:
                    if args.mem_proxy is not None:
                        """
                        if args.sequential is not None:
                            if args.unlearn_step == 1 or args.unlearn_step is None:
                                uname = f'{args.unlearn}{args.unlearn}{args.unlearn}'
                                filename = (f'{args.unlearn}_{uname}_{args.dataset}_{args.arch}_[-1]_num1000_'
                                            f'groupidNone_proxy{args.mem_proxy}_{group_id}_seqTrue_step1_seed{args.seed}.pth.tar')
                                #NG_NGNGNG_cifar100_vgg16_bn_lth_[-1]_num1000_groupidNone_proxyconfidence_high_seqTrue_step2_seed652379.pth
                            else:
                                uname = f'{args.unlearn}{args.unlearn}{args.unlearn}'
                                filename = (f'{args.unlearn}_{uname}_{args.dataset}_{args.arch}_[-1]_num1000_'
                                            f'groupidNone_proxy{args.mem_proxy}_{group_id}_seqTrue_step{args.unlearn_step}_seed{args.seed}.pth.tar')
                                #NG_NGNGNG_cifar100_vgg16_bn_lth_[-1]_num1000_groupidNone_proxyconfidence_high_seqTrue_step2_seed652379.pth
                        else:                     
                        """    
                        if args.unlearn_step == 1 or args.unlearn_step is None:
                            print("step 1 proxy")
                            #filename = "NG_cifar10_resnet18_[-1]_num3000_groupidNone_proxymax_conf_mix_step1_seed1374.pth.tar"
                            filename = (f'{args.unlearn}_{args.dataset}_{args.arch}_[-1]_num3000_'
                                        f'groupidNone_proxy{args.mem_proxy}_mix_step{args.unlearn_step}_seed{args.seed}.pth.tar')
                        elif args.unlearn_step > 1:
                            print("step 1++++ proxy")
                            #filename = (f'{args.unlearn}_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_groupidNone_proxy{args.mem_proxy}_{args.mem}_step{args.unlearn_step}_seed{args.seed}.pth.tar')
                            filename = (f'{args.unlearn}_{args.dataset}_{args.arch}_[-1]_num3000_'
                                        f'groupidNone_proxy{args.mem_proxy}_mix_step{args.unlearn_step}_seed{args.seed}.pth.tar')
                            #NG_cifar10_resnet18_[-1]_num3000_groupidNone_proxymax_conf_mix_step4_seed1374.pth.tar
                    else:
                        filename = '{}_{}_{}_{}_num{}_groupid{}_mem{}_seed{}.pth.tar'.format(args.unlearn,
                            args.dataset,args.arch,args.class_to_replace,args.num_indexes_to_replace,args.group_index,args.mem,args.seed)

                    checkpoint = utils.load_checkpoint(device, args.save_dir, args.unlearn, filename=filename)
                    eval_results = checkpoint.get("evaluation_result")
                    print(f'evaluation results for seed {args.seed} and class:{args.class_to_replace}:')
                    print(eval_results)
                    acc = eval_results['accuracy']
                    df = pd.DataFrame(acc, index=[0])
                    mia = eval_results['SVC_MIA_forget_efficacy']
                    mia_df = pd.DataFrame(mia, index=[0])
                    mia_df = mia_df.add_prefix('MIA_')
                    df = pd.concat([df, mia_df], axis=1)
                    result_dict = {f'acc_{k}': v for k, v in acc.items()}
                    result_dict.update({f'mia_{k}': v for k, v in mia.items()})
                    results[(group_id,)].append(result_dict)

                print('Loading results for retraining from scratch...')
                save_dir_rt = 'assets/unlearn/retrain'
                if args.mem_proxy is not None:
                    if args.unlearn_step == 1 or args.unlearn_step is None:
                        filename_rt = (f'retrain_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_'
                                   f'groupidNone_proxy{args.mem_proxy}_{args.mem}_step{args.unlearn_step}_seed{args.seed}.pth.tar')
                    elif args.unlearn_step > 1:
                        filename_rt = (f'retrain_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_groupidNone_proxy{args.mem_proxy}_{args.mem}_step{args.unlearn_step}_seed{args.seed}.pth.tar')
                                   #retrain_cifar10_resnet18_[-1]_num3000_groupidNone_proxymax_conf_mix_step4_seed1374.pth
                else:
                    filename_rt = (f'retrain_{args.dataset}_{args.arch}_{args.class_to_replace}_num{args.num_indexes_to_replace}_'
                                   f'groupid{args.group_index}_mem{args.mem}_seed{args.seed}.pth.tar')
                checkpoint_rt = utils.load_checkpoint(device, save_dir_rt, 'retrain', filename=filename_rt)
                eval_results_rt = checkpoint_rt.get("evaluation_result")
                print(eval_results_rt)
                acc_rt = eval_results_rt['accuracy']
                df_rt = pd.DataFrame(acc_rt, index=[0])
                mia_rt = eval_results_rt['SVC_MIA_forget_efficacy']
                mia_rt_df = pd.DataFrame(mia_rt, index=[0])
                mia_rt_df = mia_rt_df.add_prefix('MIA_')
                df_rt = pd.concat([df_rt, mia_rt_df], axis=1)
                result_dict_rt = {f'acc_{k}': v for k, v in acc_rt.items()}
                result_dict_rt.update({f'mia_{k}': v for k, v in mia_rt.items()})
                results_rt[(group_id,)].append(result_dict_rt)

                columns_to_compare = ['retain', 'MIA_confidence', 'test', 'val']
                for col in columns_to_compare:
                    ratio_col_name = f'ratio_acc_{col}'
                    if col == 'MIA_confidence':
                        df[ratio_col_name] = 1 - (abs(df[col] - df_rt[col]))
                    else:
                        df[ratio_col_name] = 1 - (abs(df[col] - df_rt[col]) / 100)

                    col_index = df.columns.get_loc(col)
                    cols = df.columns.tolist()
                    ratio_col_index = cols.index(ratio_col_name)
                    cols.insert(col_index + 1, cols.pop(ratio_col_index))
                    df = df[cols]

                df['tow'] = df['ratio_acc_MIA_confidence'] * df['ratio_acc_retain'] * df['ratio_acc_test']
                all_dfs.append(df)

            dfs = pd.concat(all_dfs)
            avg_df = dfs.mean().to_frame().transpose()
            sem_tow = stats.sem(dfs['tow'])
            t_score = stats.t.ppf(0.975, len(dfs['tow']) - 1)
            ci_margin = t_score * sem_tow
            avg_df['tow_ci_margin'] = ci_margin
            ci_lower, ci_upper = stats.t.interval(0.95, len(dfs['tow']) - 1, loc=np.mean(dfs['tow']),
                                                  scale=stats.sem(dfs['tow']))
            avg_df['tow_ci_lower'] = ci_lower
            avg_df['tow_ci_upper'] = ci_upper

            columns_to_compare = ['retain', 'forget', 'test', 'val', 'MIA_confidence']
            for col in columns_to_compare:
                sem = stats.sem(dfs[col])
                t_score = stats.t.ppf(0.975, len(dfs[col]) - 1)
                ci_margin = t_score * sem
                avg_df[f'{col}_ci'] = ci_margin

            cols = avg_df.columns.tolist()
            cols = (['tow', 'tow_ci_margin', 'tow_ci_lower', 'tow_ci_upper'] +
                    [col for col in cols if col not in ['tow', 'tow_ci_margin', 'tow_ci_lower', 'tow_ci_upper']])

            new_cols = []
            seen = set(columns_to_compare)
            for col in cols:
                if col in columns_to_compare:
                    new_cols.append(col)
                    new_cols.append(f'{col}_ci')

                else:
                    first_part = col.split('_')[0]
                    if col.startswith('MIA_confidence'):
                        prefix = '_'.join(col.split('_')[:2])
                    else:
                        prefix = first_part
                    if prefix not in seen:
                        new_cols.append(col)

            avg_df = avg_df[new_cols]
            df_dict[group_id] = avg_df

    df_final = pd.concat(df_dict)
    df_final.reset_index(inplace=True)
    df_final.rename(columns={'level_0': 'group_id'}, inplace=True)
    df_final.drop(columns=['level_1'], inplace=True)
    df_final = df_final.assign(dataset=args.dataset, arch=args.arch, proxy=args.mem_proxy, unlearn_epoch=args.unlearn_epochs, alpha=args.alpha)
    #
    fname = os.path.join(args.save_dir, "sequential_scrub_mia.csv")
    #fname = os.path.join(args.save_dir, "hyper_results.csv")
    #f'results_mia_{args.dataset}_{args.arch}_{args.unlearn}_{args.mem_proxy}_step{args.unlearn_step}.csv'

    df_final.to_csv(fname, mode='a', index=False, header=not os.path.exists(fname))
    print(df_final.to_csv(sep='\t', index=False))


if __name__ == "__main__":
    main()
