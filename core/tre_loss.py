import torch
import torch.nn as nn


class TRELoss(nn.Module):
    def __init__(self, d1, d2, d3, margin=8):

        super(TRELoss).__init__()

        # registration error will be multiplied by actual dimensions of the image
        self.multiplier = torch.tensor([d1 / 2, d2 / 2, d3 / 2]).cuda().view(1, -1, 1, 1, 1)

        # registration error will be calculated for cropped volume to eliminate border effect
        self.margin = margin

    def forward(self, pred, gt):

        pred_womargin = pred[:, :, self.margin:-self.margin, self.margin:-self.margin, self.margin:-self.margin]
        gt_womargin = gt[:, :, self.margin:-self.margin, self.margin:-self.margin, self.margin:-self.margin]
        tre = ((pred_womargin[:, :, 8:-8, 8:-8, 8:-8] - gt_womargin[:, :, 8:-8, 8:-8, 8:-8]) * self.multiplier).pow(2).sum(1).sqrt() * 1.5

        return tre


