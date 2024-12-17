import scipy.linalg
import nibabel as nib
import torch
import torch.nn.functional as F


def prepare_data(data_split):
    path = '/home/xxxx/storage/staff/eytanxxxx/projects/reg_ssl/data/abdomen_ctct'

    # idx of train and val samples
    if data_split == 'train':
        data_idx = (2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30)
    else:
        data_idx = (1, 4, 7, 10, 13, 16, 19, 22, 25, 28)

    # load images and segmentation masks
    all_img = torch.zeros(len(data_idx), 1, 192, 160, 256).cuda()  # pin_memory()
    all_seg = torch.zeros(len(data_idx), 192, 160, 256).long().cuda()  # pin_memory()
    for i in range(len(data_idx)):
        nu1 = data_idx[i]
        all_img[i, 0] = torch.from_numpy(nib.load(
            path + '/imagesTr/AbdomenCTCT_' + str(nu1).zfill(4) + '_0000.nii.gz').get_fdata()).cuda().float() / 500
        all_seg[i] = torch.from_numpy(  # 3 cases used for validation
            nib.load(
                path + '/labelsTr/AbdomenCTCT_' + str(nu1).zfill(4) + '_0000.nii.gz').get_fdata()).cuda().long()

    # build pairings to be registered
    pairs = torch.empty(0, 2).long()
    for i in range(len(data_idx)):
        for j in range(len(data_idx)):
            if (i >= j):
                continue
            pairs = torch.cat((pairs, torch.tensor([i, j]).long().view(1, 2)), 0)

    data = {'images': all_img,
            'segmentations': all_seg,
            'pairs': pairs}

    return data


# affine augmentation during training
def augment_affine_nl(disp_field2, strength=.05):
    field_lr = F.interpolate(disp_field2, scale_factor=0.5, mode='trilinear')
    field_hr = F.interpolate(field_lr, scale_factor=4, mode='trilinear')
    A1 = (torch.randn(1, 4, 4) * strength * 1.5 + torch.eye(4, 4).unsqueeze(0)).cuda();
    A1[:, 3, :3] = 0
    A2 = (torch.randn(1, 4, 4) * strength + torch.eye(4, 4).unsqueeze(0)).cuda();
    A2[:, 3, :3] = 0
    A2 = A2.matmul(A1)

    affine1 = F.affine_grid(A1[:, :3], (1, 1, 192, 160, 256))
    affine2 = F.affine_grid(A2[:, :3], (1, 1, 192, 160, 256))

    A12 = (torch.from_numpy(scipy.linalg.expm(
        scipy.linalg.logm(A1[0].cpu().double().numpy(), disp=False)[0] - scipy.linalg.logm(A2[0].cpu().double().numpy(), disp=False)[
            0]))).unsqueeze(0).cuda().float()
    affine12 = F.affine_grid(A12[:, :3], (1, 1, 192, 160, 256))
    grid0 = F.affine_grid(torch.eye(3, 4).cuda().unsqueeze(0), (1, 1, 192, 160, 256))

    field_hr2 = F.grid_sample(field_hr.cuda(), affine12.cuda())

    disp_field_aff = F.interpolate(field_hr2.cuda() + (affine12.cuda() - grid0).permute(0, 4, 1, 2, 3),
                                   scale_factor=0.5, mode='trilinear')
    return disp_field_aff, affine1, affine2


def resize_with_grid_sample_3d(tensor_to_resize, out_d, out_h, out_w):
    """
    Resize a 3D tensor using grid_sample.

    Args:
        tensor_to_resize: Tensor of shape (N, C, D_in, H_in, W_in) representing the 3D image.
        out_d: Desired output depth.
        out_h: Desired output height.
        out_w: Desired output width.

    Returns:
        resized_tensor: Resized tensor of shape (N, C, out_d, out_h, out_w).
    """
    # Get original dimensions

    N, C, in_d, in_h, in_w = tensor_to_resize.size()

    # Create normalized coordinates for the output space (between 0 and 1)
    norm_z = torch.linspace(-1, 1, out_d, device=tensor_to_resize.device)
    norm_y = torch.linspace(-1, 1, out_h, device=tensor_to_resize.device)
    norm_x = torch.linspace(-1, 1, out_w, device=tensor_to_resize.device)

    # Create mesh grids
    grid_z, grid_y, grid_x = torch.meshgrid(norm_z, norm_y, norm_x)

    # Combine them into a grid with shape (1, D_out, H_out, W_out, 3)
    grid = torch.stack((grid_x, grid_y, grid_z), dim=-1).unsqueeze(0)

    # Resize the grid to match the tensor_to_resize batch size
    grid = grid.repeat(N, 1, 1, 1, 1)

    # Use grid_sample to sample from the tensor_to_resize at the specified locations
    resized_tensor = F.grid_sample(tensor_to_resize, grid, mode='bilinear', align_corners=False)

    return resized_tensor