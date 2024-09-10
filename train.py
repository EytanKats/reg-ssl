import os
import sys
import time
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from info_nce import InfoNCE
from data_utils import augment_affine_nl, augment_affine_nl_v2, resize_with_grid_sample_3d, get_rand_affine
from dataloader_radchestct import get_data_loader
from registration_pipeline import update_fields
from coupled_convex import coupled_convex


def train(args):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Initialize wandb
    wandb.init(
        project="reg-ssl",
        name=os.path.basename(args.out_dir),
        config=args
    )

    # Create output directory
    out_dir = os.path.join(args.base_dir, args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Parse arguments
    max_samples_num = args.max_samples_num
    random_samples = True if args.random_samples == 'true' else False
    refresh_data_loader = True if args.refresh_data_loader == 'true' else False
    cache_data_to_gpu = True if args.cache_data_to_gpu == 'true' else False
    root_dir = os.path.join(args.base_dir, args.root_dir)
    data_file = os.path.join(args.base_dir, args.data_file)
    num_labels = args.num_labels
    apply_ct_abdomen_window = True if args.apply_ct_abdomen_window == 'true' else False

    iterations = args.num_iterations
    training_batch_size = args.training_batch_size

    learning_rate = args.learning_rate
    min_learning_rate = args.min_learning_rate
    do_augment = True if args.augment == 'true' else False
    main_loss = args.main_loss
    apply_contrastive_loss = True if args.contrastive == 'true' else False
    cl_coeff = args.cl_coeff
    info_nce_temperature = args.info_nce_temperature
    strength = args.strength
    num_sampled_featvecs = args.num_sampled_featvecs

    visualize = True if args.visualize == 'true' else False

    # Get training data loader that is used to calculate pseudo labels
    pseudo_labels_data_loader = get_data_loader(
        root_dir=root_dir,
        data_file=data_file,
        key='training',
        batch_size=1,
        num_workers=4,
        shuffle=False,
        drop_last=False,
        max_samples_num=max_samples_num
    )

    # Get validation data loader
    val_data_loader = get_data_loader(
        root_dir=root_dir,
        data_file=data_file,
        key='test',
        batch_size=1,
        num_workers=4,
        shuffle=False,
        drop_last=False
    )

    _, H, W, D = val_data_loader.dataset[0]['image_1'].shape

    # Create pseudo-labels for the first stage
    feature_net = nn.Sequential(
        nn.Conv3d(1, 32, 3, padding=1, stride=2), nn.BatchNorm3d(32), nn.ReLU(),
        nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
        nn.Conv3d(64, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
        nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
        nn.Conv3d(128, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
        nn.Conv3d(128, 16, 1)).cuda()

    pseudo_fields, _, _, _, _, _, _ = update_fields(
        pseudo_labels_data_loader,
        feature_net,
        use_adam=True,
        num_warps=2,
        ice=True,
        reg_fac=1.0,
        compute_jacobian=True,
        num_labels=num_labels,
        clamp=apply_ct_abdomen_window
    )

    # Reinitialize feature net with novel random weights
    feature_net = nn.Sequential(nn.Conv3d(1, 32, 3, padding=1, stride=2), nn.BatchNorm3d(32), nn.ReLU(),
                           nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
                           nn.Conv3d(64, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 16, 1)).cuda()

    proj_net = nn.Sequential(nn.Conv3d(128, 128, 1)).cuda()

    # Instantiate InfoNCE loss
    info_loss = InfoNCE(temperature=info_nce_temperature)

    optimizer = torch.optim.Adam(list(feature_net.parameters()) + list(proj_net.parameters()), lr=learning_rate)
    if main_loss == 'tre':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1000, 1, eta_min=min_learning_rate)
    elif main_loss == 'mind':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations, eta_min=min_learning_rate)

    # placeholders for input images, pseudo labels, and affine augmentation matrices
    img0 = torch.zeros(training_batch_size, 1, H, W, D).cuda()
    img1 = torch.zeros(training_batch_size, 1, H, W, D).cuda()
    img0_aug = torch.zeros(training_batch_size, 1, H, W, D).cuda()
    img1_aug = torch.zeros(training_batch_size, 1, H, W, D).cuda()
    affine1 = torch.zeros(training_batch_size, H, W, D, 3).cuda()
    affine2 = torch.zeros(training_batch_size, H, W, D, 3).cuda()
    affine1_aug = torch.zeros(training_batch_size, H, W, D, 3).cuda()
    affine2_aug = torch.zeros(training_batch_size, H, W, D, 3).cuda()
    mind0 = torch.zeros(training_batch_size, 12, H//2, W//2, D//2).cuda()
    mind1 = torch.zeros(training_batch_size, 12, H//2, W//2, D//2).cuda()
    target = torch.zeros(training_batch_size, 3, H // 2, W // 2, D // 2).cuda()
    target_aug = torch.zeros(training_batch_size, 3, H // 2, W // 2, D // 2).cuda()

    grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda().repeat(2, 1, 1), [2, 1, H//2, W//2, D//2], align_corners=False)

    i = 0
    stage = 0
    t0 = time.time()
    train_data_loader = None
    update_data_loader = True
    with tqdm(total=iterations, file=sys.stdout, colour="red", ncols=200) as pbar:
        while i < iterations:

            if update_data_loader:
                train_data_loader = get_data_loader(
                        root_dir=root_dir,
                        data_file=data_file,
                        key='training',
                        batch_size=training_batch_size,
                        fast=cache_data_to_gpu,
                        max_samples_num=max_samples_num,
                        random_samples=random_samples
                    )
                update_data_loader = False

            for data_pair in train_data_loader:
                optimizer.zero_grad()

                img0_ = (data_pair['image_1'] / 500).cuda()
                img1_ = (data_pair['image_2'] / 500).cuda()
                mind0_ = data_pair['mind_1'].cuda()
                mind1_ = data_pair['mind_2'].cuda()
                indices = data_pair['idx'].numpy().tolist()

                for j in range(training_batch_size):
                    target[j:j + 1] = pseudo_fields[indices[j]:indices[j] + 1].cuda()

                img0_.requires_grad_(True)
                img1_.requires_grad_(True)
                mind0_.requires_grad_(True)
                mind1_.requires_grad_(True)

                # visualize input data
                if visualize:
                    for j in range(training_batch_size):
                        image_0_ = img0_.data.cpu().numpy()[j, 0, ...].copy()
                        image_0 = img0.data.cpu().numpy()[j, 0, ...].copy()

                        image_1_ = img1_.data.cpu().numpy()[j, 0, ...].copy()
                        image_1 = img1.data.cpu().numpy()[j, 0, ...].copy()

                        center_slice = image_0.shape[2] // 2

                        f, axarr = plt.subplots(2, 2)
                        axarr[0, 0].imshow(image_0_[:, :, center_slice], cmap='gray')
                        axarr[0, 1].imshow(image_0[:, :, center_slice], cmap='gray')
                        axarr[1, 0].imshow(image_1_[:, :, center_slice], cmap='gray')
                        axarr[1, 1].imshow(image_1[:, :, center_slice], cmap='gray')

                        plt.show()
                        plt.close()

                if do_augment:
                    with torch.no_grad():
                        for j in range(training_batch_size):
                            min_val_0 = torch.min(img0_[j:j + 1])
                            min_val_1 = torch.min(img1_[j:j + 1])

                            disp_field = target[j:j + 1]
                            disp_field_aff, affine1[j:j + 1], affine2[j:j + 1] = augment_affine_nl(disp_field, shape=(1, 1, H, W, D))
                            img0[j:j + 1] = F.grid_sample(img0_[j:j + 1] - min_val_0, affine1[j:j + 1]) + min_val_0
                            img1[j:j + 1] = F.grid_sample(img1_[j:j + 1] - min_val_1, affine2[j:j + 1]) + min_val_1
                            target_aug[j:j + 1] = disp_field_aff

                            h, w, d = mind0.shape[-3], mind0.shape[-2], mind0.shape[-1]
                            affine1_mind = resize_with_grid_sample_3d(affine1[j:j + 1].permute(0, 4, 1, 2, 3), h, w, d).permute(0, 2, 3, 4, 1)
                            affine2_mind = resize_with_grid_sample_3d(affine2[j:j + 1].permute(0, 4, 1, 2, 3), h, w, d).permute(0, 2, 3, 4, 1)

                            mind0[j:j + 1] = F.grid_sample(mind0_[j:j + 1], affine1_mind)
                            mind1[j:j + 1] = F.grid_sample(mind1_[j:j + 1], affine2_mind)
                else:
                    with torch.no_grad():
                        disp_field = target[j:j + 1]
                        disp_field_aff, affine1[j:j + 1], affine2[j:j + 1] = augment_affine_nl(disp_field, shape=(1, 1, H, W, D), strength=0.)
                        img0[j:j + 1] = F.grid_sample(img0_[j:j + 1] - min_val_0, affine1[j:j + 1]) + min_val_0
                        img1[j:j + 1] = F.grid_sample(img1_[j:j + 1] - min_val_1, affine2[j:j + 1]) + min_val_1
                        target_aug[j:j + 1] = disp_field_aff

                img0.requires_grad_(True)
                img1.requires_grad_(True)
                mind0.requires_grad_(True)
                mind1.requires_grad_(True)

                # feature extraction with feature net g
                features_fix = feature_net(img0)
                features_mov = feature_net(img1)

                if main_loss == 'mind':
                    disp_pred = coupled_convex(features_fix, features_mov, use_ice=False, img_shape=(H // 2, W // 2, D // 2))
                    mind_warp = F.grid_sample(mind1.cuda().float(), grid0 + disp_pred.permute(0, 2, 3, 4, 1))
                    loss = nn.MSELoss()(mind0.cuda().float()[:, :, 8:-8, 8:-8, 8:-8], mind_warp[:, :, 8:-8, 8:-8, 8:-8]) * 1.5
                elif main_loss == 'tre':
                    disp_pred = coupled_convex(features_fix, features_mov, use_ice=False, img_shape=(H // 2, W // 2, D // 2))
                    tre = ((disp_pred[:, :, 8:-8, 8:-8, 8:-8] - target_aug[:, :, 8:-8, 8:-8, 8:-8]) * torch.tensor([D / 2, W / 2, H / 2]).cuda().view(1, -1, 1, 1, 1)).pow(2).sum(1).sqrt() * 1.5
                    loss = tre.mean()

                wandb.log({"reg_loss": loss.detach().cpu().numpy()}, step=i)

                # apply contrastive loss
                if apply_contrastive_loss:

                    with torch.no_grad():
                        for j in range(training_batch_size):
                            min_val_0 = torch.min(img0_[j:j + 1])
                            min_val_1 = torch.min(img1_[j:j + 1])

                            affine1_aug[j:j + 1], affine2_aug[j:j + 1] = augment_affine_nl_v2(shape=(1, 1, H, W, D), strength=strength)
                            img0_aug[j:j + 1] = F.grid_sample(img0_[j:j + 1] - min_val_0, affine1_aug[j:j + 1], align_corners=False) + min_val_0
                            img1_aug[j:j + 1] = F.grid_sample(img1_[j:j + 1] - min_val_1, affine2_aug[j:j + 1], align_corners=False) + min_val_1

                        # visualize data for contrastive loss
                        if visualize:
                            for j in range(training_batch_size):
                                image_0_ = img0_.data.cpu().numpy()[j, 0, ...].copy()
                                image_0_aug = img0_aug.data.cpu().numpy()[j, 0, ...].copy()

                                image_1_ = img1_.data.cpu().numpy()[j, 0, ...].copy()
                                image_1_aug = img1_aug.data.cpu().numpy()[j, 0, ...].copy()

                                center_slice = image_0_.shape[2] // 2

                                f, axarr = plt.subplots(2, 2)
                                axarr[0, 0].imshow(image_0_[:, :, center_slice], cmap='gray')
                                axarr[0, 1].imshow(image_0_aug[:, :, center_slice], cmap='gray')
                                axarr[1, 0].imshow(image_1_[:, :, center_slice], cmap='gray')
                                axarr[1, 1].imshow(image_1_aug[:, :, center_slice], cmap='gray')

                                plt.show()
                                plt.close()

                        features_fix_aug = proj_net(feature_net[:-4](img0_aug))
                        features_mov_aug = proj_net(feature_net[:-4](img1_aug))

                    features_fix = proj_net(feature_net[:-4](img0_))
                    features_mov = proj_net(feature_net[:-4](img1_))

                    features_fix_warped = torch.zeros((training_batch_size, 128, H // 4, W // 4, D // 4)).cuda()
                    features_mov_warped = torch.zeros((training_batch_size, 128, H // 4, W // 4, D // 4)).cuda()

                    h, w, d = features_fix.shape[-3], features_fix.shape[-2], features_fix.shape[-1]
                    affine1_feat = resize_with_grid_sample_3d(affine1_aug.permute(0, 4, 1, 2, 3), h, w, d).permute(0, 2, 3, 4, 1)
                    affine2_feat = resize_with_grid_sample_3d(affine2_aug.permute(0, 4, 1, 2, 3), h, w, d).permute(0, 2, 3, 4, 1)

                    featvecs_aug_list = []
                    featvecs_warped_list = []
                    for j in range(training_batch_size):
                        features_fix_warped[j:j + 1] = F.grid_sample(features_fix[j:j + 1], affine1_feat[j:j + 1], align_corners=False)
                        features_mov_warped[j:j + 1] = F.grid_sample(features_mov[j:j + 1], affine2_feat[j:j + 1], align_corners=False)

                        # Get locations to sample from feature masks
                        ids = torch.argwhere(torch.zeros(h, w, d) > -1)
                        ids = ids[(ids[:, 0] > 4) & (ids[:, 1] > 4) & (ids[:, 2] > 4) & (ids[:, 0] < h - 5) & (ids[:, 1] < w - 5) & (ids[:, 2] < d - 5)]

                        # Sample feature vectors
                        ids = ids[torch.multinomial(torch.ones(ids.shape[0]), num_samples=num_sampled_featvecs)]
                        featvecs_aug_list.append(features_fix_aug[j, :].permute(1, 2, 3, 0)[torch.unbind(ids, dim=1)])
                        featvecs_warped_list.append(features_fix_warped[j, :].permute(1, 2, 3, 0)[torch.unbind(ids, dim=1)])

                        ids = ids[torch.multinomial(torch.ones(ids.shape[0]), num_samples=num_sampled_featvecs)]
                        featvecs_aug_list.append(features_mov_aug[j, :].permute(1, 2, 3, 0)[torch.unbind(ids, dim=1)])
                        featvecs_warped_list.append(features_mov_warped[j, :].permute(1, 2, 3, 0)[torch.unbind(ids, dim=1)])

                    # visualize features
                    if visualize:
                        for j in range(training_batch_size):
                            f_fix = features_fix.data.cpu().numpy()[j, 64, ...].copy()
                            f_fix_aug = features_fix_aug.data.cpu().numpy()[j, 64, ...].copy()
                            f_fix_warped = features_fix_warped.data.cpu().numpy()[j, 64, ...].copy()

                            f_mov = features_mov.data.cpu().numpy()[j, 64, ...].copy()
                            f_mov_aug = features_mov_aug.data.cpu().numpy()[j, 64, ...].copy()
                            f_mov_warped = features_mov_warped.data.cpu().numpy()[j, 64, ...].copy()

                            center_slice = f_fix.shape[2] // 2

                            f, axarr = plt.subplots(2, 3)
                            axarr[0, 0].imshow(f_fix[:, :, center_slice], cmap='gray')
                            axarr[0, 1].imshow(f_fix_aug[:, :, center_slice], cmap='gray')
                            axarr[0, 2].imshow(f_fix_warped[:, :, center_slice], cmap='gray')
                            axarr[1, 0].imshow(f_mov[:, :, center_slice], cmap='gray')
                            axarr[1, 1].imshow(f_mov_aug[:, :, center_slice], cmap='gray')
                            axarr[1, 2].imshow(f_mov_warped[:, :, center_slice], cmap='gray')

                            plt.show()
                            plt.close()

                    cl_loss = info_loss(torch.concat(featvecs_aug_list), torch.concat(featvecs_warped_list)) * cl_coeff
                    wandb.log({"infoNCE_loss": cl_loss.detach().cpu().numpy()}, step=i)

                    loss = cl_loss + loss



                loss.backward()
                optimizer.step()
                scheduler.step()

                lr = float(scheduler.get_last_lr()[0])
                wandb.log({"learning_rate": lr}, step=i)

                if i == 0 or i % 100 == 99:

                    if i == 0 or i % 1000 == 999:
                        log_to_wandb = True
                    else:
                        log_to_wandb = False

                    _, d_all_net_test, d_all0_test, d_all_adam_test, d_all_ident_test, test_sdlogj, test_sdlogj_adam = update_fields(
                        val_data_loader, feature_net, use_adam=True, num_warps=2, ice=True, reg_fac=10.,
                        log_to_wandb=log_to_wandb, iteration=i, compute_jacobian=True, num_labels=num_labels, clamp=apply_ct_abdomen_window
                    )

                    print(f'DICE: {d_all0_test.sum() / (d_all_ident_test > 0.1).sum()} -> {d_all_net_test.sum() / (d_all_ident_test > 0.1).sum()} -> {d_all_adam_test.sum() / (d_all_ident_test > 0.1).sum()}')
                    print(f'SDLOGJ: {test_sdlogj} -> {test_sdlogj_adam}')
                    wandb.log({"dice": d_all_net_test.sum() / (d_all_ident_test > 0.1).sum()}, step=i)
                    wandb.log({"dice_after_instance_optimisation": d_all_adam_test.sum() / (d_all_ident_test > 0.1).sum()}, step=i)
                    wandb.log({"sdlogj": test_sdlogj}, step=i)
                    wandb.log({"sdlogj__after_instance_optimisation": test_sdlogj_adam}, step=i)

                if i % 1000 == 999:

                    # end of stage
                    stage += 1

                    torch.save(feature_net.cpu(), os.path.join(out_dir, 'stage' + str(stage) + '.pth'))
                    feature_net.cuda()

                    # Generate pseudo-labels for the next stage
                    pseudo_fields, _, _, _, _, _, _ = update_fields(
                        pseudo_labels_data_loader,
                        feature_net,
                        use_adam=True,
                        num_warps=2,
                        ice=True,
                        reg_fac=1.0,
                        compute_jacobian=True,
                        num_labels=num_labels,
                        clamp=apply_ct_abdomen_window
                    )

                feature_net.train()
                proj_net.train()

                pbar_desciprtion = f"iter: {i}, runtime: {'%0.3f' % (time.time() - t0)} sec, GPU max/memory: {'%0.2f' % (torch.cuda.max_memory_allocated() * 1e-9)} GByte"
                pbar.set_description(pbar_desciprtion)
                pbar.update(1)

                i += 1

                if i % 1000 == 0:

                    if refresh_data_loader:
                        update_data_loader = True

                    break


