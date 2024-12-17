import torch
from scipy.stats import wilcoxon

from dataloader_radchestct import get_data_loader
from registration_pipeline import update_fields


def test(args):

    if args.dataset == 'abdomenctct':
        root_dir = f''
        data_file = f''

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
        root_dir = f''
        data_file = f''

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

    dice_1 = None
    for ckpt_path in args.ckpt_path_1:
        feature_net = torch.load(ckpt_path).cuda()

        working_point_found = False
        direction = None
        update = 2.5
        reg_fac = 10
        while not working_point_found:
            all_fields, d_all_net, d_all0, d_all_adam, d_all_ident, sdlogj, sdloj_adam = update_fields(data, feature_net, True, num_warps=2, compute_jacobian=True, ice=True, reg_fac=10, num_labels=num_labels, clamp=apply_ct_abdomen_window)
            print('DSC:', d_all0.sum() / (d_all_ident > 0.1).sum(), '>', d_all_net.sum() / (d_all_ident > 0.1).sum(), '>', d_all_adam.sum() / (d_all_ident > 0.1).sum())
            print('SDLJ:', sdlogj, '>', sdloj_adam)
            print(f'REG_FAC = {reg_fac}, UPDATE = {update}')

            if sdloj_adam > 0.110:  # 0.155 for abdomenct, 0.110 for radchestct

                if direction is None:
                    reg_fac = reg_fac + update
                    direction = "down"
                    continue

                if direction == "down":
                    reg_fac = reg_fac + update
                    continue

                if direction == "up":
                    update = update / 2
                    reg_fac = reg_fac + update
                    direction = "down"
                    continue

            elif sdloj_adam < 0.090:  # 0.135 for abdomenct, 0.090 for radchestct

                if direction is None:
                    reg_fac = reg_fac - update
                    direction = "up"
                    continue

                if direction == "up":
                    reg_fac = reg_fac - update
                    continue

                if direction == "down":
                    update = update / 2
                    reg_fac = reg_fac - update
                    direction = "up"
                    continue
            else:
                print("WORKING POINT FOUND")
                working_point_found = True

        if dice_1 is None:
            dice_1 = d_all_adam
        else:
            dice_1 = dice_1 + d_all_adam

    dice_1 = dice_1 / len(args.ckpt_path_1)

    dice_2 = None
    for ckpt_path in args.ckpt_path_2:
        feature_net = torch.load(ckpt_path).cuda()

        working_point_found = False
        direction = None
        update = 2.5
        reg_fac = 10
        while not working_point_found:
            all_fields_, d_all_net_, d_all0_, d_all_adam_, d_all_ident_, sdlogj_, sdloj_adam_ = update_fields(data, feature_net, True, num_warps=2, compute_jacobian=True, ice=True, reg_fac=reg_fac, num_labels=num_labels, clamp=apply_ct_abdomen_window)
            print('DSC:', d_all0_.sum() / (d_all_ident_ > 0.1).sum(), '>', d_all_net_.sum() / (d_all_ident_ > 0.1).sum(), '>', d_all_adam_.sum() / (d_all_ident_ > 0.1).sum())
            print('SDLJ:', sdlogj_, '>', sdloj_adam_)
            print(f'REG_FAC = {reg_fac}, UPDATE = {update}')

            if sdloj_adam_ > 0.155:  # 0.155 for abdomenct, 0.110 for radchestct

                if direction is None:
                    reg_fac = reg_fac + update
                    direction = "down"
                    continue

                if direction == "down":
                    reg_fac = reg_fac + update
                    continue

                if direction == "up":
                    update = update / 2
                    reg_fac = reg_fac + update
                    direction = "down"
                    continue

            elif sdloj_adam_ < 0.135:  # 0.135 for abdomenct,  0.090 for radchestct

                if direction is None:
                    reg_fac = reg_fac - update
                    direction = "up"
                    continue

                if direction == "up":
                    reg_fac = reg_fac - update
                    continue

                if direction == "down":
                    update = update / 2
                    reg_fac = reg_fac - update
                    direction = "up"
                    continue
            else:
                print("WORKING POINT FOUND")
                working_point_found = True

        if dice_2 is None:
            dice_2 = d_all_adam_
        else:
            dice_2 = dice_2 + d_all_adam_

    dice_2 = dice_2 / len(args.ckpt_path_2)

    print('DSC:', (dice_1.sum() / (d_all_ident > 0.1).sum()), '>', (dice_2.sum() / (d_all_ident_ > 0.1).sum()))

    x = torch.flatten(dice_1).numpy()
    y = torch.flatten(dice_2).numpy()

    res = wilcoxon(x, y)  # alternative='less'
    print(res)


