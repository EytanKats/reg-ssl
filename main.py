import argparse

from test import test
from train import train

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")

    # ###########################
    # ##### SYSTEM SETTINGS #####

    # GPU to use
    parser.add_argument(
        "--gpu",
        default="0",
        metavar="FILE",
        help="gpu to train on",
        type=str,
    )

    # ############################
    # ##### GENERAL SETTINGS #####

    # train or test
    parser.add_argument(
        "--phase",
        default="train",
        type=str,
    )
    parser.add_argument(
        "--base_dir",
        default="/home/kats/storage/staff/eytankats/projects/reg_ssl/",
        help="directory to write results to",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        default="experiments/trying_reproduce/",
        help="directory to write results to",
        type=str,
    )

    # ##########################
    # ##### DATA SETTINGS #####

    # dataset: abdomenctct, radchestct
    parser.add_argument(
        "--dataset",
        default="radchestct",
        type=str,
    )
    # cache data to GPU to save training time
    parser.add_argument(
        "--cache_data_to_gpu",
        default="true",
        type=str,
    )
    # whether to choose random samples for training data loader
    parser.add_argument(
        "--random_samples",
        default="false",
        type=str,
    )

    # ##########################
    # ##### TEST SETTINGS #####

    parser.add_argument(
        "--ckpt_path_1",
        default=["/home/kats/storage/staff/eytankats/projects/reg_ssl/experiments/dataloader_abdomenct_baseline_regcyc_noclamp_1/student_stage10.pth",
                 "/home/kats/storage/staff/eytankats/projects/reg_ssl/experiments/dataloader_abdomenct_baseline_regcyc_noclamp_2/student_stage10.pth",
                 "/home/kats/storage/staff/eytankats/projects/reg_ssl/experiments/dataloader_abdomenct_baseline_regcyc_noclamp_3/student_stage10.pth"],
        help="chekpoint to load",
        type=str,
    )
    parser.add_argument(
        "--ckpt_path_2",
        default=["/home/kats/storage/staff/eytankats/projects/reg_ssl/experiments/dataloader_abdomenct_comete_noclamp_1/student_stage10.pth",
                 "/home/kats/storage/staff/eytankats/projects/reg_ssl/experiments/dataloader_abdomenct_comete_noclamp_2/student_stage10.pth",
                 "/home/kats/storage/staff/eytankats/projects/reg_ssl/experiments/dataloader_abdomenct_comete_noclamp_3/student_stage10.pth"],
        help="chekpoint to load",
        type=str,
    )

    # #############################
    # ##### TRAINING SETTINGS #####

    parser.add_argument(
        "--num_iterations",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--training_batch_size",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--use_optim_with_restarts",
        default="true",
        type=str,
    )
    # regularization weight in Adam optimization during training
    parser.add_argument(
        "--reg_fac",
        default=1.,
        type=float,
    )
    # whether to use affine input augmentations during training
    parser.add_argument(
        "--augment",
        default="true",
        type=str,
    )
    # whether to use teacher-student approach during the training
    parser.add_argument(
        "--ema",
        default="false",
        type=str,
    )
    # whether to use mind loss
    parser.add_argument(
        "--use_mind",
        default="false",
        type=str,
    )
    # whether to apply contrastive loss during training
    parser.add_argument(
        "--contrastive",
        default="true",
        type=str,
    )
    # number of positive pairs for contrastive loss
    parser.add_argument(
        "--num_sampled_featvecs",
        default=1000,
        type=int,
    )
    # temperature factor for infoNCE loss
    parser.add_argument(
        "--info_nce_temperature",
        default=0.1,
        type=float,
    )
    # strength of affine augmentations for contrastive loss
    parser.add_argument(
        "--strength",
        default=0.25,
        type=int,
    )

    # ##########################
    # ##### DEBUG SETTINGS #####

    # visualize with matplotlib
    parser.add_argument(
        "--visualize",
        default="false",
        type=str,
    )

    # #############################
    # ##### RUN TRAIN OR TEST #####

    args = parser.parse_args()
    if args.phase == 'test':
        test(args)
    else:
        train(args)
