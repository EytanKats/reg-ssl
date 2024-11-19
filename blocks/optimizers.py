from torch.optim import AdamW


def get_optimizer(
        settings,
        model):

    if isinstance(model, list):
        params = list(model[0].parameters()) + list(model[1].parameters())
    else:
        params = model.parameters()

    optimizer = AdamW(
        params,
        lr=settings['optimizer']['lr'],
        weight_decay=settings['optimizer']['weight_decay']
    )

    return optimizer
