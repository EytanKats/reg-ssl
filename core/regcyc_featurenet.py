import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, input_channels, output_channels, stride):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv3d(input_channels, output_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm = nn.BatchNorm3d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x, get_conv=False):

        conv = self.conv(x)
        if get_conv:  # return results right after the convolution layer
            return conv

        norm = self.batch_norm(conv)
        relu = self.relu(norm)

        return relu


class FeatureNet(nn.Module):

    def __init__(self):
        super(FeatureNet, self).__init__()

        self.conv_block1 = ConvBlock(1, 32, 2)
        self.conv_block2 = ConvBlock(32, 64, 1)
        self.conv_block3 = ConvBlock(64, 128, 2)
        self.conv_block4 = ConvBlock(64, 128, 1)
        self.conv_block5 = ConvBlock(64, 128, 2)
        self.feat_head = nn.Conv3d(128, 16, 1)

        self.proj_head = nn.Conv3d(128, 128, 1)

    def forward(self, x, get_convblock4=False, get_conv=False):

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)

        if get_convblock4:  # return results right after the 4th convolution block
            x = self.conv_block4(x, get_conv)
            return x

        x = self.conv_block5(x)
        x = self.feat_head(x)

        return x
