from core.tre_loss import TRELoss
from core.info_nce import InfoNCE


def get_loss_functions(settings):

    loss_fns = []

    tre_loss = TRELoss(
        settings['dataset']['d1'],
        settings['dataset']['d2'],
        settings['dataset']['d3']
    )
    loss_fns.append(tre_loss)

    info_nce_loss = InfoNCE(settings['loss']['temperature'])
    loss_fns.append(info_nce_loss)

    return loss_fns
