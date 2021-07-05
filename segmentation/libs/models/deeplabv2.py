import numpy as np
import torch
from torch import nn
from torchvision import models
from collections import OrderedDict

class Classifier_Module(nn.Module):

    def __init__(self, dims_in, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(nn.Conv2d(dims_in, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias = True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out


class DeeplabResNet(nn.Module):
    def __init__(self, num_classes = 21):
        super(DeeplabResNet, self).__init__()
        model = models.resnet101(pretrained=True)

        # Layer 3 (OS=16 -> OS=8)
        model.layer3[0].conv2.stride = (1, 1)
        model.layer3[0].downsample[0].stride = (1, 1)
        for m in model.layer3[1:]:
            m.conv2.padding = (2, 2)
            m.conv2.dilation = (2, 2)

        # Layer 4 (OS=32 -> OS=8)
        model.layer4[0].conv2.stride = (1, 1)
        model.layer4[0].downsample[0].stride = (1, 1)
        for m in model.layer4[1:]:
            m.conv2.padding = (4, 4)
            m.conv2.dilation = (4, 4)

        # Remove "avgpool" and "fc", and add ASPP
        model = list(model.named_children())[:-2]

        self.features = nn.Sequential(OrderedDict(model))

        self.classifier = Classifier_Module(2048, [6,12,18,24],[6,12,18,24],num_classes)


    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def optim_parameters(self, ):
        return self.parameters()
