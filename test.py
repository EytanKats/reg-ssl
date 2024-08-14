import torch
from scipy.stats import wilcoxon

from dataloader_radchestct import get_data_loader
from registration_pipeline import update_fields


def test(args):

    if args.dataset == 'abdomenctct':
        root_dir = f'/home/kats/storage/staff/eytankats/projects/reg_ssl/data/abdomen_ctct'
        data_file = f'/home/kats/storage/staff/eytankats/projects/reg_ssl/data/abdomen_ctct/abdomen_ct_orig.json'

        data = get_data_loader(
            root_dir=root_dir,
            data_file=data_file,
            key='test',
            batch_size=1,
            num_workers=4,
            shuffle=False,
            drop_last=False)

        num_labels = 14
        apply_ct_abdomen_window = True

    elif args.dataset == 'radchestct':
        root_dir = f'/home/kats/storage/staff/eytankats/projects/reg_ssl/data/radchest_ct/'
        data_file = f'/home/kats/storage/staff/eytankats/projects/reg_ssl/data/radchest_ct/radchest_ct_fold0.json'

        data = get_data_loader(
            root_dir=root_dir,
            data_file=data_file,
            key='test',
            batch_size=1,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            max_samples_num=None
        )

        num_labels = 22
        apply_ct_abdomen_window = False

    feature_net = torch.load(args.ckpt_path_1).cuda()

    all_fields, d_all_net, d_all0, d_all_adam, d_all_ident, sdlogj, sdloj_adam = update_fields(data, feature_net, True, num_warps=2, compute_jacobian=True, ice=True, reg_fac=10., num_labels=num_labels, clamp=apply_ct_abdomen_window)
    print('DSC:', d_all0.sum() / (d_all_ident > 0.1).sum(), '>', d_all_net.sum() / (d_all_ident > 0.1).sum(), '>', d_all_adam.sum() / (d_all_ident > 0.1).sum())
    print('SDLJ:', sdlogj, '>', sdloj_adam)

    feature_net = torch.load(args.ckpt_path_2).cuda()
    all_fields_, d_all_net_, d_all0_, d_all_adam_, d_all_ident_, sdlogj_, sdloj_adam_ = update_fields(data, feature_net, True, num_warps=2, compute_jacobian=True, ice=True, reg_fac=10., num_labels=num_labels, clamp=apply_ct_abdomen_window)
    print('DSC:', d_all0_.sum() / (d_all_ident_ > 0.1).sum(), '>', d_all_net_.sum() / (d_all_ident_ > 0.1).sum(), '>', d_all_adam_.sum() / (d_all_ident_ > 0.1).sum())
    print('SDLJ:', sdlogj_, '>', sdloj_adam_)

    x = torch.flatten(d_all_adam).numpy()
    y = torch.flatten(d_all_adam_).numpy()

    res = wilcoxon(x, y)
    print(res)


