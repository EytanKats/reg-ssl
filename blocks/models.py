import os
import torch

from core.regcyc_featurenet import FeatureNet


def get_feature_extractor(settings):

    if settings['model']['architecture'] == 'regcyc_featurenet':
        model = FeatureNet()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model


def get_model():
    return get_feature_extractor



