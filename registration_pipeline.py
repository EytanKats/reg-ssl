import wandb
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from coupled_convex import coupled_convex
from eval_utils import dice_coeff, jacobian_determinant
from adam_instance_opt import AdamReg

# compute displacement fields with current model and evaluate Dice score: called after each stage and at test time
def update_fields(data, feature_net, use_adam, num_warps=1, compute_jacobian=False, ice=False, reg_fac=1., log_to_wandb=False, iteration=0):
    all_img = data['images']
    all_seg = data['segmentations']
    pairs = data['pairs']

    # placeholders for dice scores and SDlogJ
    d_all0 = torch.empty(0, 13)
    d_all_ident = torch.empty(0, 13)
    d_all_net = torch.empty(0, 13)
    d_all_adam = torch.empty(0, 13)
    sdlogj_net = []
    sdlog_adam = []

    _, _, H, W, D = all_img.shape
    all_fields = torch.zeros(len(pairs), 3, H // 2, W // 2, D // 2)
    grid0 = F.affine_grid(torch.eye(3, 4).unsqueeze(0).cuda(), (1, 1, H, W, D))

    feature_net.eval()

    for idx in trange(len(pairs)):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # select image pair and segmentations
                img0 = torch.clamp(all_img[pairs[idx, 0]].cuda().unsqueeze(0), -.4, .6)  # .repeat(1,2,1,1,1)
                img1 = torch.clamp(all_img[pairs[idx, 1]].cuda().unsqueeze(0), -.4, .6)  # .repeat(1,2,1,1,1)
                img1_orig = img1.clone()
                fixed_seg = all_seg[pairs[idx, 0]].cuda()
                moving_seg = all_seg[pairs[idx, 1]].cuda()
                moving_seg_orig = moving_seg.clone()

                # feature extraction with feature net g
                features_fix = feature_net(img0)
                features_mov = feature_net(img1)

                # differentiable optimization with optimizer h (coupled convex)
                disp = coupled_convex(features_fix, features_mov, use_ice=ice, img_shape=(H, W, D))

                # warp moving segmentation according to computed disp field
                warped_seg = F.grid_sample(
                    moving_seg.view(1, 1, H, W, D).float(),
                    grid0 + disp.permute(0, 2, 3, 4, 1), mode='nearest'
                ).squeeze(1)

                disp_0 = disp.clone()

                # compute DSC
                dsc_0 = dice_coeff(fixed_seg.contiguous(), moving_seg.contiguous(), 14).cpu()
                dsc_1 = dice_coeff(fixed_seg.contiguous(), warped_seg.contiguous(), 14).cpu()
                dsc_ident = dice_coeff(fixed_seg.contiguous(), fixed_seg.contiguous(), 14).cpu() * dice_coeff(moving_seg.contiguous(), moving_seg.contiguous(), 14).cpu()

                for _ in range(num_warps - 1):
                    # warp moving image with first disp field to generate input for 2nd warp
                    warped_img = F.grid_sample(img1, grid0 + disp.permute(0, 2, 3, 4, 1), mode='nearest')
                    img1 = torch.clamp(warped_img, -.4, .6)
                    moving_seg = warped_seg[0]

                    # feature extraction with feature net g
                    features_fix = feature_net(img0)
                    features_mov = feature_net(img1)

                    # differentiable optimization with optimizer h (coupled convex)
                    disp = coupled_convex(features_fix, features_mov, use_ice=ice, img_shape=(H, W, D))

                    # warp moving segmentation according to computed disp field
                    warped_seg = F.grid_sample(moving_seg.view(1, 1, H, W, D).float(), grid0 + disp.permute(0, 2, 3, 4, 1), mode='nearest').squeeze(1)

                    # compute DSC
                    dsc_1 = dice_coeff(fixed_seg.contiguous(), warped_seg.contiguous(), 14).cpu()

                    disp_1 = disp.clone()
                    disp = disp_0 + F.grid_sample(disp_1, disp_0.permute(0,2,3,4,1) + grid0)

                d_all0 = torch.cat((d_all0, dsc_0.view(1, -1)), 0)
                d_all_net = torch.cat((d_all_net, dsc_1.view(1, -1)), 0)
                d_all_ident = torch.cat((d_all_ident, dsc_ident.view(1, -1)), 0)

                if compute_jacobian:
                    dense_flow = disp * torch.tensor([H - 1, W - 1, D - 1]).cuda().view(1, 3, 1, 1, 1) / 2
                    jac = jacobian_determinant(dense_flow)
                    sdlogj_net.append(torch.log((jac + 3).clamp_(0.000000001, 1000000000)).std().item())

            if use_adam:
                # instance optimization with Adam
                with torch.no_grad():
                    # random projection network
                    proj = nn.Conv3d(64, 32, 1, bias=False)
                    proj.cuda()
                    # feature extraction with g and projection to 32 channels
                    feat_fix = proj(feature_net[:6](img0))
                    feat_mov = proj(feature_net[:6](img1_orig))
                    # finetuning of displacement field with Adam
                    flow = AdamReg(5 * feat_fix, 5 * feat_mov, disp, reg_fac=reg_fac)

                if compute_jacobian:
                    dense_flow = flow * torch.tensor([H - 1, W - 1, D - 1]).cuda().view(1, 3, 1, 1, 1) / 2
                    jac = jacobian_determinant(dense_flow)
                    sdlog_adam.append(torch.log((jac + 3).clamp_(0.000000001, 1000000000)).std().item())

                warped_seg = F.grid_sample(moving_seg_orig.view(1, 1, H, W, D).float(), grid0 + flow.permute(0, 2, 3, 4, 1), mode='nearest').squeeze(1)

                if log_to_wandb and idx % 5 == 0:

                    # warp moving image and visualize central slice of warped and fixed image
                    warped_img_adam = F.grid_sample(img1_orig, grid0 + flow.permute(0, 2, 3, 4, 1), mode='nearest')
                    warped_img_adam = torch.clamp(warped_img_adam, -.4, .6)

                    fixed = img0.data.cpu().numpy()[0, 0, ...].copy()
                    moving = img1_orig.data.cpu().numpy()[0, 0, ...].copy()
                    warped = warped_img_adam.data.cpu().numpy()[0, 0, ...].copy()

                    center_slice = fixed.shape[2] // 2

                    f, axarr = plt.subplots(1, 3, figsize=(15, 15))
                    axarr[0].imshow(fixed[:, :, center_slice], cmap='gray')
                    axarr[1].imshow(moving[:, :, center_slice], cmap='gray')
                    axarr[2].imshow(warped[:, :, center_slice], cmap='gray')

                    wandb.log({f'fixed-warped-pair-{idx}': f}, iteration)

                    plt.close(f)

                dsc_2 = dice_coeff(fixed_seg.contiguous(), warped_seg.contiguous(), 14).cpu()
                all_fields[idx] = F.interpolate(flow, scale_factor=.5, mode='trilinear').cpu()
                d_all_adam = torch.cat((d_all_adam, dsc_2.view(1, -1)), 0)

            else:
                all_fields[idx] = F.interpolate(disp, scale_factor=.5, mode='trilinear').cpu()

    return all_fields, d_all_net, d_all0, d_all_adam, d_all_ident, np.mean(sdlogj_net), np.mean(sdlog_adam)
