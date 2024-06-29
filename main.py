import os
import argparse

from train import train
from test import test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    # train or test
    parser.add_argument(
        "--phase",
        default="test",
        type=str,
    )
    parser.add_argument(
        "--ckpt_path",
        default="/home/kats/storage/staff/eytankats/projects/reg_ssl/experiments/clw1_s32/stage16.pth",
        help="chekpoint to load",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        default="/home/kats/storage/staff/eytankats/projects/reg_ssl/experiments/original",
        help="directory to write results to",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default="0",
        metavar="FILE",
        help="gpu to train on",
        type=str,
    )
    parser.add_argument(
        "--num_warps",
        default=2,
        type=int,
    )
    # whether to use inverse consistence
    parser.add_argument(
        "--ice",
        default="true",
        type=str,
    )
    # regularization weight in Adam optimization during training
    parser.add_argument(
        "--reg_fac",
        default=10.,
        type=float,
    )
    # whether to perform difficulty-weighted data sampling during training
    parser.add_argument(
        "--sampling",
        default="true",
        type=str,
    )
    # whether to finetune pseudo labels with Adam instance optimization during training
    parser.add_argument(
        "--adam",
        default="true",
        type=str,
    )
    # whether to use affine input augmentations during training
    parser.add_argument(
        "--augment",
        default="true",
        type=str,
    )
    # whether to apply contrastive loss during training
    parser.add_argument(
        "--contrastive",
        default="true",
        type=str,
    )
    # temperature factor for infoNCE loss
    parser.add_argument(
        "--info_nce_temperature",
        default=0.1,
        type=float,
    )
    # visualize with matplotlib
    parser.add_argument(
        "--visualize",
        default="false",
        type=str,
    )
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    import torch

    if args.phase == 'test':
        test(args)

    else:
        train(args)
