import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from eval_utils import dice_coeff

RAD_CHEST_CT = '/home/kats/storage/staff/eytankats/projects/reg_ssl/data/radchest_ct'
IMAGES = 'imgs'
LABELS = 'lbls_ts'
PAIRS = 'list_of_pairs.pth'


pairs = torch.load(os.path.join(RAD_CHEST_CT, PAIRS))

for pair in pairs[:10]:

    img_1 = nib.load(os.path.join(RAD_CHEST_CT, IMAGES, f'{pair[0]}.nii.gz')).get_fdata()
    img_2 = nib.load(os.path.join(RAD_CHEST_CT, IMAGES, f'{pair[1]}.nii.gz')).get_fdata()

    lbl_1 = nib.load(os.path.join(RAD_CHEST_CT, LABELS, f'{pair[0]}.nii.gz')).get_fdata()
    lbl_2 = nib.load(os.path.join(RAD_CHEST_CT, LABELS, f'{pair[1]}.nii.gz')).get_fdata()

    dice = dice_coeff(torch.tensor(lbl_1).contiguous(), torch.tensor(lbl_2).contiguous(), max_label=22)

    print(f'____________')
    print(f'img_1 shape: {lbl_1.shape}')
    print(f'img_2 shape: {lbl_2.shape}')
    print(f'dice: {torch.mean(dice).numpy()}')

    center_slice_1_0 = img_1.shape[0] // 2
    center_slice_2_0 = img_2.shape[0] // 2
    center_slice_1_1 = img_1.shape[1] // 2
    center_slice_2_1 = img_2.shape[1] // 2
    center_slice_1_2 = img_1.shape[2] // 2
    center_slice_2_2 = img_2.shape[2] // 2

    f, axarr = plt.subplots(2, 6, figsize=(15, 10))

    axarr[0, 0].imshow(img_1[center_slice_1_0, :, :], cmap='gray')
    axarr[0, 1].imshow(lbl_1[center_slice_1_0, :, :])
    axarr[0, 2].imshow(img_1[:, center_slice_1_1, :], cmap='gray')
    axarr[0, 3].imshow(lbl_1[:, center_slice_1_1, :])
    axarr[0, 4].imshow(img_1[:, :, center_slice_1_2], cmap='gray')
    axarr[0, 5].imshow(lbl_1[:, :, center_slice_1_2])

    axarr[1, 0].imshow(img_2[center_slice_2_0, :, :], cmap='gray')
    axarr[1, 1].imshow(lbl_2[center_slice_2_0, :, :])
    axarr[1, 2].imshow(img_2[:, center_slice_2_1, :], cmap='gray')
    axarr[1, 3].imshow(lbl_2[:, center_slice_2_1, :])
    axarr[1, 4].imshow(img_2[:, :, center_slice_2_2], cmap='gray')
    axarr[1, 5].imshow(lbl_2[:, :, center_slice_2_2])

    plt.show()
    plt.close()

print('End of Script')



