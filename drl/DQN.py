import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import timm


class DQN(nn.Module):
    def __init__(self, h, w, c, outputs):
        super(DQN, self).__init__()
        self.upsampler = nn.UpsamplingBilinear2d(size=(h * 8, w * 8))
        self.input_norm = nn.BatchNorm2d(c)
        self.body = timm.create_model('mobilenetv3_large_100', pretrained=True)
        # change input channels from RGB to C
        self.body.conv_stem.weight = nn.Parameter(self.body.conv_stem.weight.mean(dim=1, keepdim=True).repeat(1, c, 1, 1))
        # retrain last layer
        self.body.classifier = nn.Linear(self.body.classifier.in_features, outputs)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.upsampler(x)
        x = self.input_norm(x)
        out = self.body(x)
        return out
