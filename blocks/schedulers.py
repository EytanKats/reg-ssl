from torch.optim.lr_scheduler import CosineAnnealingLR


def get_scheduler(
        settings,
        optimizer):

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=settings['scheduler']['max_epochs'],
        eta_min=settings['scheduler']['min_lr']
    )

    return scheduler
