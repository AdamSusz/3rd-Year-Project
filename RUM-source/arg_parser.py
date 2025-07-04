import argparse

def parse_classes(s):
    """Function to parse a comma-separated string into a list of integers."""
    try:
        return [int(item) for item in s.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError("List of classes should be integers separated by commas")

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Lottery Tickets Experiments")

    ##################################### Dataset #################################################
    parser.add_argument(
        # "--data", type=str, default="../data", help="location of the data corpus"
        "--data", type=str, default="/home/u2280917/Desktop/dataset", help="location of the data corpus"

    )
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset")
    parser.add_argument("--input_size", type=int, default=32, help="size of input images")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./tiny-imagenet-200",
        help="dir to tiny-imagenet",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    ##################################### Architecture ############################################
    parser.add_argument("--arch", type=str, default="resnet18", help="model architecture")
    parser.add_argument("--imagenet_arch", action="store_true", help="architecture for imagenet size samples",)
    parser.add_argument(
        "--train_y_file",
        type=str,
        default="./labels/train_ys.pth",
        help="labels for training files",
    )
    parser.add_argument(
        "--val_y_file",
        type=str,
        default="./labels/val_ys.pth",
        help="labels for validation files",
    )
    ##################################### General setting ############################################
    parser.add_argument("--seed", default=2, type=int, help="random seed")  # default=2
    parser.add_argument("--train_seed", default=None, type=int, help="seed for training (default value same as args.seed)",)
    parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
    parser.add_argument("--workers", type=int, default=4, help="number of workers in dataloader")
    parser.add_argument("--resume", action="store_true", help="resume from checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint file")
    parser.add_argument("--save_dir", help="The directory used to save the trained models", default=None, type=str,)
    parser.add_argument("--mask", type=str, default=None, help="sparse model")

    ##################################### Training setting #################################################
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--epochs", default=30, type=int, help="number of total epochs to run")  # default=182)
    parser.add_argument("--warmup", default=0, type=int, help="warm up epochs")
    parser.add_argument("--print_freq", default=50, type=int, help="print frequency")
    parser.add_argument("--decreasing_lr", default="91,136", help="decreasing strategy")
    parser.add_argument("--no_aug", action="store_true", default=False, help="No augmentation in training dataset (transformation).",)
    parser.add_argument("--no-l1-epochs", default=0, type=int, help="non l1 epochs")
    ##################################### Pruning setting #################################################
    parser.add_argument("--prune", type=str, default="omp", help="method to prune")
    parser.add_argument("--pruning_times", default=1, type=int, help="overall times of pruning (only works for IMP)",)
    parser.add_argument(
        "--rate", default=0.95, type=float, help="pruning rate"
    )  # pruning rate is always 20%
    parser.add_argument(
        "--prune_type",
        default="rewind_lt",
        type=str,
        help="IMP type (lt, pt or rewind_lt)",
    )
    parser.add_argument("--random_prune", action="store_true", help="whether using random prune")
    parser.add_argument("--rewind_epoch", default=0, type=int, help="rewind checkpoint")
    parser.add_argument("--rewind_pth", default=None, type=str, help="rewind checkpoint to load")

    ##################################### Unlearn setting #################################################
    parser.add_argument("--unlearn", type=str, default=None, help="method to unlearn")   # default="retrain"
    parser.add_argument("--unlearn_lr", default=0.01, type=float, help="initial learning rate")
    parser.add_argument("--unlearn_epochs",default=10,type=int,help="number of total epochs for unlearn to run",)

    parser.add_argument("--num_indexes_to_replace",type=int,default=None,help="Number of data to forget (If None, all examples of the specified class will be replaced)",)
    parser.add_argument("--class_to_replace", type=parse_classes, default=[-1],help="Specific classes to forget (comma-separated, eg.'1,2,3'. If -1, all data points are to be considered)")

    parser.add_argument("--indexes_to_replace",type=list,default=None,help="Specific index data to forget",)
    # IU: control the magnitude of the adjustment made to the model's parameters, higher alpha means stronger adjustment
    parser.add_argument("--alpha", default=0.2, type=float, help="unlearn noise")

    parser.add_argument("--path", default=None, type=str, help="mask matrix")

    ##################################### RUM setting #################################################
    parser.add_argument("--surgical", action="store_true", help="surgical fine-tuning")
    parser.add_argument("--choice", nargs="+", type=str, help="choices of layers")
    parser.add_argument("--group_index", default=None, type=int, help="select group index for finegrained embedding overlap")
    parser.add_argument("--mem", default=None, type=str, help="memorization (high or low)")
    parser.add_argument("--mem_proxy", default=None, type=str, help="memorization proxy: confidence, pd")
    parser.add_argument("--sequential", action="store_true", help="sequential fine-tuning")
    parser.add_argument("--no_save", action="store_true", help="do not save model")
    parser.add_argument("--shuffle", action="store_true", help="shuffle FS before applying RUM")

    parser.add_argument("--msteps", type=int, default=None, help="scrub")
    parser.add_argument("--kd_T", type=float, default=None, help="scrub")
    parser.add_argument("--beta", type=float, default=None, help="scrub")
    parser.add_argument("--gamma", type=float, default=None, help="scrub")

    parser.add_argument("--unlearn_step", type=int, default=None, help="Specifies the step of sequential unlearning")
    parser.add_argument("--uname", type=str, default=None, help="which unlearning algo(s) to use in RUM, eg. NGNGNG")

    ##################################### Attack setting #################################################
    parser.add_argument("--attack", type=str, default="backdoor", help="method to unlearn")
    parser.add_argument("--trigger_size", type=int, default=4, help="The size of trigger of backdoor attack",)

    ##################################### Heldout Retrain setting #################################################
    parser.add_argument("--ft_lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--ft_epochs", default=30, type=int, help="epochs for finetuning")

    ##################################### Loss Curvature setting #################################################
    parser.add_argument('--h', default='0.01', type=str, help='scale the perturbation by h')

    ######################################## WandB setting #######################################################
    parser.add_argument('--wandb-mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                        help='wandb running mode')
    parser.add_argument('--wandb-project', type=str, default=None, help='the project on wandb to add the runs')
    parser.add_argument('--wandb-entity', type=str, default=None, help='your wandb user name')
    parser.add_argument('--wandb-run-id', type=str, default=None, help='To resume a previous run with an id')
    parser.add_argument('--wandb-group-name', type=str, default=None, help='Given name to group runs together')

    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],)
    return parser.parse_args()
