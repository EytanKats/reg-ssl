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

    PHASE = 'train'  # train or test

    parser.add_argument(
        "--phase",
        default=PHASE,
        type=str,
    )
    parser.add_argument(
        "--base_dir",
        default="/home/kats/storage/staff/eytankats/projects/reg_ssl",
        help="directory to write results to",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        default="experiments/trying_reproduce",
        help="directory to write results to",
        type=str,
    )

    # ##########################
    # ##### DATA SETTINGS #####

    DATASET = 'radchestct'  # dataset: abdomenctct, radchestct

    parser.add_argument(
        "--dataset",
        default=DATASET,
        type=str,
    )
    # cache data to GPU to save training time
    parser.add_argument(
        "--cache_data_to_gpu",
        default="true",
        type=str,
    )
    # number of samples used in training data loader
    parser.add_argument(
        "--max_samples_num",
        default=128,
        type=int
    )
    # whether to choose random samples for training data loader
    parser.add_argument(
        "--random_samples",
        default="false",
        type=str,
    )
    # whether to update samples in data loader during training
    parser.add_argument(
        "--refresh_data_loader",
        default="false",
        type=str,
    )

    # #####################################
    # ##### DATASET SPECIFIC SETTINGS #####

    if DATASET == "radchestct":
        parser.add_argument(
            "--root_dir",
            default="data/radchest_ct",
            help="directory with the data files",
            type=str,
        )
        parser.add_argument(
            "--data_file",
            default="data/radchest_ct/radchest_ct_fold0.json",
            help="data .json file",
            type=str,
        )
        parser.add_argument(
            "--num_labels",
            default=22,
            help="number of segmentation labels in dataset used to assess registration performance",
            type=int,
        )
        # whether to apply ct abdomen window during validation
        parser.add_argument(
            "--apply_ct_abdomen_window",
            default="false",
            help="apply ct abdomen window during validation",
            type=str,
        )
    elif DATASET == "abdomenctct":
        parser.add_argument(
            "--root_dir",
            default="data/abdomen_ctct",
            help="directory with the data files",
            type=str,
        )
        parser.add_argument(
            "--data_file",
            default="data/abdomen_ctct/abdomen_ct_orig.json",
            help="data .json file",
            type=str,
        )
        parser.add_argument(
            "--num_labels",
            default=14,
            help="number of segmentation labels in dataset used to assess registration performance",
            type=int,
        )
        # whether to apply ct abdomen window during validation
        parser.add_argument(
            "--apply_ct_abdomen_window",
            default="true",
            help="apply ct abdomen window during validation",
            type=str,
        )

    # ##########################
    # ##### TEST SETTINGS #####

    if PHASE == 'test':
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
        "--learning_rate",
        default=0.001,
        type=float,
    )
    parser.add_argument(
        "--min_learning_rate",
        default=0.00001,
        type=float,
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
    # weight of contrastive loss
    parser.add_argument(
        "--cl_coeff",
        default=0.01,
        type=float,
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
        default=0.05,
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

    # ################################
    # ##### RUN TRAINING OR TEST #####

    args = parser.parse_args()
    if args.phase == 'test':
        test(args)
    else:
        train(args)
