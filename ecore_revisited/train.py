import sys
import time
from tqdm import tqdm
import os
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from info_nce import InfoNCE
from ecore_revisited.data_utils import prepare_data, augment_affine_nl, resize_with_grid_sample_3d
from ecore_revisited.registration_pipeline import update_fields
from ecore_revisited.coupled_convex import coupled_convex


def train(args):

    # Initialize wandb
    wandb.init(
        project="ecore",
        name=os.path.basename(args.out_dir[:-1]),
        config=args
    )

    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_warps = args.num_warps
    reg_fac = args.reg_fac
    use_ice = True if args.ice == 'true' else False
    use_adam = True if args.adam == 'true' else False
    do_sampling = True if args.sampling == 'true' else False
    do_augment = True if args.augment == 'true' else False
    apply_contrastive_loss = True if args.contrastive_loss == 'true' else False

    # Loading training data (segmentations only used for validation after each stage)
    data = prepare_data(data_split='train')
    data_test = prepare_data(data_split='test')

    # initialize feature net
    feature_net = nn.Sequential(nn.Conv3d(1, 32, 3, padding=1, stride=2), nn.BatchNorm3d(32), nn.ReLU(),
                           nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
                           nn.Conv3d(64, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 16, 1)).cuda()
    print()

    N, _, H, W, D = data['images'].shape

    # generate initial pseudo labels with random features
    if use_adam:
        # w/o Adam finetuning
        all_fields_noadam, d_all_net, d_all0, _, _ = update_fields(data, feature_net, use_adam=False, num_warps=num_warps,
                                                                 ice=use_ice, reg_fac=reg_fac)
        # w/ Adam finetuning
        all_fields, _, _, d_all_adam, d_all_ident = update_fields(data, feature_net, use_adam=True, num_warps=num_warps, ice=use_ice,
                                                        reg_fac=reg_fac)
        # compute difference between finetuned and non-finetuned fields for difficulty sampling --> the larger the difference, the more difficult the sample
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                tre_adam = ((all_fields_noadam[:, :, 8:-8, 8:-8, 8:-8] - all_fields[:, :, 8:-8, 8:-8,
                                                                              8:-8]) * torch.tensor(
                    [D / 2, W / 2, H / 2]).view(1, -1, 1, 1, 1)).pow(2).sum(1).sqrt() * 1.5
                tre_adam1 = (tre_adam.mean(-1).mean(-1).mean(-1))

        # Log training metrics
        print(f'TRAIN DICE for iter 0:', d_all0.sum() / (d_all_ident > 0.1).sum(), '>', d_all_net.sum() / (d_all_ident > 0.1).sum(), '>', d_all_adam.sum() / (d_all_ident > 0.1).sum())
        wandb.log({"train_dice": d_all_net.sum() / (d_all_ident > 0.1).sum()}, step=0)
        wandb.log({"train_dice_io": d_all_adam.sum() / (d_all_ident > 0.1).sum()}, step=0)

    else:
        # w/o Adam finetuning
        all_fields, d_all_net, d_all0, _, _ = update_fields(data, feature_net, use_adam=False, num_warps=num_warps,
                                                            ice=use_ice, reg_fac=reg_fac)
        print('fields updated val error:', d_all0[:3].mean(), '>', d_all_net[:3].mean())

    # reinitialize feature net with novel random weights
    feature_net = nn.Sequential(nn.Conv3d(1, 32, 3, padding=1, stride=2), nn.BatchNorm3d(32), nn.ReLU(),
                           nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(),
                           nn.Conv3d(64, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 128, 3, padding=1, stride=2), nn.BatchNorm3d(128), nn.ReLU(),
                           nn.Conv3d(128, 16, 1)).cuda()

    # Define projection layer and loss for contrastive learning
    info_loss = InfoNCE(temperature=0.1)
    proj_net = nn.Sequential(nn.Conv3d(128, 128, 1)).cuda()

    # Calculate initial results on test data
    all_fields_test, d_all_net_test, d_all0_test, d_all_adam_test, d_all_ident_test = update_fields(data_test, feature_net, True, num_warps=2, compute_jacobian=True, ice=True, reg_fac=10.)

    # Log validation metrics
    print(f'VAL DICE for iter 0:', d_all0_test.sum() / (d_all_ident_test > 0.1).sum(), '>', d_all_net_test.sum() / (d_all_ident_test > 0.1).sum(), '>', d_all_adam_test.sum() / (d_all_ident_test > 0.1).sum())
    wandb.log({"val_dice": d_all_net_test.sum() / (d_all_ident_test > 0.1).sum()}, step=0)
    wandb.log({"val_dice_io": d_all_adam_test.sum() / (d_all_ident_test > 0.1).sum()}, step=0)

    # perform overall 8 (2x4) cycle of self-training
    for repeat in range(2):
        stage = 0 + repeat * 4

        feature_net.cuda()
        feature_net.train()
        print()

        optimizer = torch.optim.Adam(list(feature_net.parameters()) + list(proj_net.parameters()), lr=0.001)
        eta_min = 0.00001
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 500 * 2, 1, eta_min=eta_min)
        run_lr = torch.zeros(2000 * 2)
        half_iterations = 2000 * 2
        run_loss = torch.zeros(half_iterations)
        scaler = torch.cuda.amp.GradScaler()

        # placeholders for input images, pseudo labels, and affine augmentation matrices
        img0 = torch.zeros(2, 1, H, W, D).cuda()
        img1 = torch.zeros(2, 1, H, W, D).cuda()
        img0_aug = torch.zeros(2, 1, H, W, D).cuda()
        img1_aug = torch.zeros(2, 1, H, W, D).cuda()
        target = torch.zeros(2, 3, H // 2, W // 2, D // 2).cuda()
        affine1 = torch.zeros(2,  H, W, D, 3).cuda()
        affine2 = torch.zeros(2,  H, W, D, 3).cuda()
        affine1_aug = torch.zeros(2, H, W, D, 3).cuda()
        affine2_aug = torch.zeros(2, H, W, D, 3).cuda()

        t0 = time.time()
        with tqdm(total=half_iterations, file=sys.stdout, colour="red") as pbar:
            for i in range(half_iterations):
                optimizer.zero_grad()
                # difficulty weighting
                if use_adam and do_sampling:
                    q = torch.zeros(len(data['pairs']))
                    q[torch.argsort(tre_adam1)] = torch.sigmoid(torch.linspace(5, -5, len(data['pairs'])))
                else:
                    q = torch.ones(len(data['pairs']))
                idx = torch.tensor(list(WeightedRandomSampler(q, 2, replacement=True))).long()

                with torch.cuda.amp.autocast():
                    # image selection and augmentation
                    img0_ = data['images'][data['pairs'][idx, 0]].cuda()
                    img1_ = data['images'][data['pairs'][idx, 1]].cuda()
                    if do_augment:
                        with torch.no_grad():
                            for j in range(len(idx)):
                                disp_field = all_fields[idx[j]:idx[j] + 1].cuda()
                                disp_field_aff, affine1[j:j + 1], affine2[j:j + 1] = augment_affine_nl(disp_field)
                                img0[j:j + 1] = F.grid_sample(img0_[j:j + 1], affine1[j:j + 1])
                                img1[j:j + 1] = F.grid_sample(img1_[j:j + 1], affine2[j:j + 1])
                                target[j:j + 1] = disp_field_aff
                    else:
                        with torch.no_grad():
                            for j in range(len(idx)):
                                input_field = all_fields[idx[j]:idx[j] + 1].cuda()
                                disp_field_aff, affine1[j:j + 1], affine2[j:j + 1] = augment_affine_nl(input_field, strength=0.)
                                img0[j:j + 1] = F.grid_sample(img0_[j:j + 1], affine1[j:j + 1])
                                img1[j:j + 1] = F.grid_sample(img1_[j:j + 1], affine2[j:j + 1])
                                target[j:j + 1] = disp_field_aff
                    img0.requires_grad = True
                    img1.requires_grad = True

                    # feature extraction with feature net g
                    features_fix = feature_net(img0)
                    features_mov = feature_net(img1)

                    # differentiable optimization with optimizer h (coupled convex)
                    disp_pred = coupled_convex(features_fix, features_mov, use_ice=False, img_shape=(H//2, W//2, D//2))

                    # consistency loss between prediction and pseudo label
                    tre = ((disp_pred[:, :, 8:-8, 8:-8, 8:-8] - target[:, :, 8:-8, 8:-8, 8:-8]) * torch.tensor(
                        [D / 2, W / 2, H / 2]).cuda().view(1, -1, 1, 1, 1)).pow(2).sum(1).sqrt() * 1.5
                    loss = tre.mean()

                    wandb.log({"reg_loss": loss.detach().cpu().numpy()}, step=repeat * half_iterations + i)

                    # apply contrastive loss
                    if apply_contrastive_loss:

                        with torch.no_grad():
                            for j in range(2):

                                disp_field = all_fields[idx[j]:idx[j] + 1].cuda()
                                _, affine1_aug[j:j + 1], affine2_aug[j:j + 1] = augment_affine_nl(disp_field)
                                img0_aug[j:j + 1] = F.grid_sample(img0_[j:j + 1], affine1_aug[j:j + 1])
                                img1_aug[j:j + 1] = F.grid_sample(img1_[j:j + 1], affine2_aug[j:j + 1])

                            features_fix_aug = proj_net(feature_net[:-4](img0_aug))
                            features_mov_aug = proj_net(feature_net[:-4](img1_aug))

                        img0_.requires_grad = True
                        img1_.requires_grad = True

                        features_fix = proj_net(feature_net[:-4](img0_))
                        features_mov = proj_net(feature_net[:-4](img1_))

                        features_fix_warped = torch.zeros((2, 128, H // 4, W // 4, D // 4)).cuda()
                        features_mov_warped = torch.zeros((2, 128, H // 4, W // 4, D // 4)).cuda()

                        h, w, d = features_fix.shape[-3], features_fix.shape[-2], features_fix.shape[-1]
                        affine1_feat = resize_with_grid_sample_3d(affine1_aug.permute(0, 4, 1, 2, 3), h, w, d).permute(0, 2, 3, 4, 1)
                        affine2_feat = resize_with_grid_sample_3d(affine2_aug.permute(0, 4, 1, 2, 3), h, w, d).permute(0, 2, 3, 4, 1)

                        featvecs_aug_list = []
                        featvecs_warped_list = []
                        for j in range(2):
                            features_fix_warped[j:j + 1] = F.grid_sample(features_fix[j:j + 1], affine1_feat[j:j + 1], align_corners=False)
                            features_mov_warped[j:j + 1] = F.grid_sample(features_mov[j:j + 1], affine2_feat[j:j + 1], align_corners=False)

                            # Get locations to sample from feature masks
                            ids = torch.argwhere(torch.zeros(h, w, d) > -1)
                            ids = ids[(ids[:, 0] > 4) & (ids[:, 1] > 4) & (ids[:, 2] > 4) & (ids[:, 0] < h - 5) & (ids[:, 1] < w - 5) & (ids[:, 2] < d - 5)]

                            # Sample feature vectors
                            ids = ids[torch.multinomial(torch.ones(ids.shape[0]), num_samples=1000)]
                            featvecs_aug_list.append(features_fix_aug[j, :].permute(1, 2, 3, 0)[torch.unbind(ids, dim=1)])
                            featvecs_warped_list.append(features_fix_warped[j, :].permute(1, 2, 3, 0)[torch.unbind(ids, dim=1)])

                            ids = ids[torch.multinomial(torch.ones(ids.shape[0]), num_samples=1000)]
                            featvecs_aug_list.append(features_mov_aug[j, :].permute(1, 2, 3, 0)[torch.unbind(ids, dim=1)])
                            featvecs_warped_list.append(features_mov_warped[j, :].permute(1, 2, 3, 0)[torch.unbind(ids, dim=1)])

                        cl_loss = info_loss(torch.concat(featvecs_aug_list), torch.concat(featvecs_warped_list))
                        wandb.log({"infoNCE_loss": cl_loss.detach().cpu().numpy()}, step=repeat * half_iterations + i)

                        loss = cl_loss + loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                lr1 = float(scheduler.get_last_lr()[0])
                run_lr[i] = lr1

                if ((i % 1000 == 999)):
                    # end of stage
                    stage += 1
                    torch.save(feature_net.cpu(), os.path.join(out_dir, 'stage' + str(stage) + '.pth'))
                    feature_net.cuda()
                    torch.save(run_loss, os.path.join(out_dir, 'run_loss_rep={}.pth'.format(repeat)))
                    print()

                    #  recompute pseudo-labels with current model weights
                    if use_adam:
                        # w/o Adam finetuning
                        all_fields_noadam, d_all_net, d_all0, _, _ = update_fields(data, feature_net, use_adam=False,
                                                                                 num_warps=num_warps, ice=use_ice,
                                                                                 reg_fac=reg_fac)
                        # w Adam finetuning
                        all_fields, _, _, d_all_adam, d_all_ident = update_fields(data, feature_net, use_adam=True, num_warps=num_warps,
                                                                        ice=use_ice, reg_fac=reg_fac)

                        # recompute difference between finetuned and non-finetuned fields for difficulty sampling --> the larger the difference, the more difficult the sample
                        with torch.no_grad():
                            with torch.cuda.amp.autocast():
                                tre_adam = ((all_fields_noadam[:, :, 8:-8, 8:-8, 8:-8] - all_fields[:, :, 8:-8,
                                                                                              8:-8,
                                                                                              8:-8]) * torch.tensor(
                                    [D / 2, W / 2, H / 2]).view(1, -1, 1, 1, 1)).pow(2).sum(1).sqrt() * 1.5
                                tre_adam1 = (tre_adam.mean(-1).mean(-1).mean(-1))

                        # Log training metrics
                        print(f'TRAIN DICE for iter {repeat * half_iterations + i + 1}:', d_all0.sum() / (d_all_ident > 0.1).sum(), '>', d_all_net.sum() / (d_all_ident > 0.1).sum(), '>', d_all_adam.sum() / (d_all_ident > 0.1).sum())
                        wandb.log({"train_dice": d_all_net.sum() / (d_all_ident > 0.1).sum()}, step=repeat * half_iterations + i + 1)
                        wandb.log({"train_dice_io": d_all_adam.sum() / (d_all_ident > 0.1).sum()}, step=repeat * half_iterations + i + 1)

                        # Calculate intermediate results on test data
                        all_fields_test, d_all_net_test, d_all0_test, d_all_adam_test, d_all_ident_test = update_fields(data_test, feature_net, True, num_warps=2, compute_jacobian=True, ice=True, reg_fac=10.)

                        # Log validation metrics
                        print(f'VAL DICE for iter {repeat * half_iterations + i + 1}:', d_all0_test.sum() / (d_all_ident_test > 0.1).sum(), '>', d_all_net_test.sum() / (d_all_ident_test > 0.1).sum(), '>', d_all_adam_test.sum() / (d_all_ident_test > 0.1).sum())
                        wandb.log({"val_dice": d_all_net_test.sum() / (d_all_ident_test > 0.1).sum()}, step=repeat * half_iterations + i + 1)
                        wandb.log({"val_dice_io": d_all_adam_test.sum() / (d_all_ident_test > 0.1).sum()}, step=repeat * half_iterations + i + 1)

                    else:
                        # w/o Adam finetuning
                        all_fields, d_all_net, d_all0, _, _ = update_fields(data, feature_net, use_adam=False,
                                                                            num_warps=num_warps, ice=use_ice, reg_fac=reg_fac)
                        print('fields updated val error:', d_all0[:3].mean(), '>', d_all_net[:3].mean())

                    feature_net.train()

                run_loss[i] = loss.item()

                str1 = f"iter: {i}, loss: {'%0.3f' % (run_loss[i - 34:i - 1].mean())}, runtime: {'%0.3f' % (time.time() - t0)} sec, GPU max/memory: {'%0.2f' % (torch.cuda.max_memory_allocated() * 1e-9)} GByte"
                pbar.set_description(str1)
                pbar.update(1)
