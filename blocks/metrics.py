import torch
from monai.utils import MetricReduction
from torchmetrics.classification import MulticlassAccuracy

from domain_adversarial_segmentation.core.DiceMetricWrapper import DiceMetricWrapper


def get_metrics(settings):

    metrics = []

    metric = DiceMetricWrapper(settings, mode='mean', reduction=MetricReduction.NONE)
    metric.__name__ = f'{settings["metrics"]["ct_metric_name"]}_mean'
    metrics.append(metric)

    for label in settings['dataset']['labels']:
        metric = DiceMetricWrapper(settings, mode=label)
        metric.__name__ = f'{settings["metrics"]["ct_metric_name"]}_{label}'
        metrics.append(metric)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric = MulticlassAccuracy(num_classes=settings["metrics"]["domains_num"]).to(device)
    metric.__name__ = f'{settings["metrics"]["accuracy_metric_name"]}'
    metrics.append(metric)

    return metrics
