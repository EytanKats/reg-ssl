import os
import torch
import numpy as np
import nibabel as nib

from tqdm import tqdm

RAD_CHEST_CT = '/home/xxxx/storage/staff/yyyyyxxxx/data/radchest_ct'
OUTPUT_DIR = '/home/xxxx/storage/staff/yyyyyxxxx/projects/reg_ssl/data/radchest_ct'
IMAGES = 'imgs'
LABELS = 'lbls_ts'
PAIRS = 'list_of_pairs.pth'
SHAPE = (256, 256, 224)
TS_MAP = {
    13: 1,  # "lung_upper_lobe_left"
    14: 2,  # "lung_lower_lobe_left"
    15: 3,  # "lung_upper_lobe_right"
    16: 4,  # "lung_middle_lobe_right"
    17: 5,  # "lung_lower_lobe_right"
    23: 6,  # '"vertebrae_T12"
    24: 7,  # '"vertebrae_T11"
    25: 8,  # '"vertebrae_T10"
    26: 9,  # '"vertebrae_T9"
    27: 10,  # '"vertebrae_T8"
    28: 11,  # '"vertebrae_T7"
    29: 12,  # '"vertebrae_T6"
    30: 13,  # '"vertebrae_T5"
    31: 14,  # '"vertebrae_T4"
    32: 15,  # '"vertebrae_T3"
    33: 16,  # '"vertebrae_T2"
    34: 17,  # '"vertebrae_T1"
    44: 18,  # "heart_myocardium": 44,
    45: 19,  # "heart_atrium_left": 45,
    46: 20,  # "heart_ventricle_left": 46,
    47: 21,  # "heart_atrium_right": 47,
    48: 22,  # "heart_ventricle_right": 48,
}

def to_shape(a, shape):
    x_, y_, z_ = shape
    x, y, z = a.shape
    x_pad = max((x_ - x), 0)
    y_pad = max((y_ - y), 0)
    z_pad = max((z_ - z), 0)
    img = np.pad(
        a,
        ((x_pad // 2, x_pad // 2 + x_pad % 2),
         (y_pad // 2, y_pad // 2 + y_pad % 2),
         (z_pad // 2, z_pad // 2 + z_pad % 2)),
        mode='constant',
        constant_values=np.min(a)
    )

    x, y, z = img.shape
    x_crop = max((x - x_), 0)
    y_crop = max((y - y_), 0)
    z_crop = max((z - z_), 0)
    img = img[x_crop // 2:x - x_crop // 2 - x_crop % 2,
          y_crop // 2:y - y_crop // 2 - y_crop % 2,
          z_crop // 2:z - z_crop // 2 - z_crop % 2]

    return img


def map_ts_labels(lbl, map):

    lbl_mapped = np.zeros_like(lbl)
    for key, val in map.items():
        lbl_mapped[lbl == key] = val
    return lbl_mapped


pairs = torch.load(os.path.join(RAD_CHEST_CT, PAIRS))

for pair in tqdm(pairs):

    img_1_nib = nib.load(os.path.join(RAD_CHEST_CT, IMAGES, f'{pair[0]}.nii.gz'))
    img_1 = img_1_nib.get_fdata()
    img_1 = to_shape(img_1, SHAPE)
    nib.save(nib.Nifti1Image(img_1.astype(np.float32), affine=img_1_nib.affine), os.path.join(OUTPUT_DIR, IMAGES, f'{pair[0]}.nii.gz'))

    img_2_nib = nib.load(os.path.join(RAD_CHEST_CT, IMAGES, f'{pair[1]}.nii.gz'))
    img_2 = img_2_nib.get_fdata()
    img_2 = to_shape(img_2, SHAPE)
    nib.save(nib.Nifti1Image(img_2.astype(np.float32), affine=img_2_nib.affine), os.path.join(OUTPUT_DIR, IMAGES, f'{pair[1]}.nii.gz'))

    lbl_1_nib = nib.load(os.path.join(RAD_CHEST_CT, LABELS, f'{pair[0]}_ts.nii.gz'))
    lbl_1 = lbl_1_nib.get_fdata()
    lbl_1 = to_shape(lbl_1, SHAPE)
    lbl_1 = map_ts_labels(lbl_1, TS_MAP)
    nib.save(nib.Nifti1Image(lbl_1.astype(np.float32), affine=lbl_1_nib.affine), os.path.join(OUTPUT_DIR, LABELS, f'{pair[0]}.nii.gz'))

    lbl_2_nib = nib.load(os.path.join(RAD_CHEST_CT, LABELS, f'{pair[1]}_ts.nii.gz'))
    lbl_2 = lbl_2_nib.get_fdata()
    lbl_2 = to_shape(lbl_2, SHAPE)
    lbl_2 = map_ts_labels(lbl_2, TS_MAP)
    nib.save(nib.Nifti1Image(lbl_2.astype(np.float32), affine=lbl_2_nib.affine), os.path.join(OUTPUT_DIR, LABELS, f'{pair[1]}.nii.gz'))

print('End of Script')



