import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
import cv2
import numpy as np
import os


model_urls = {'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'}

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True, att_dir='./runs/', training_epoch=15):
        super(VGG, self).__init__()
        self.features = features
        self.extra_convs = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512,20,1)            
        )
        self._initialize_weights()
        self.training_epoch = training_epoch
        self.att_dir = att_dir
        if not os.path.exists(self.att_dir):
            os.makedirs(self.att_dir)

    
    def forward(self, x, sal=None, epoch=2, label=None, index=None):
        x = self.features(x)
        x = self.extra_convs(x)
        
        self.map1 = x.clone()
        x_ori = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0)
        x_ori = x_ori.view(-1, 20)

        
        center_loss1 = 0
        center_loss2 = 0
        cam_center_dist_loss = 0

        if index != None:
            num_class = label.sum(1)
            ind_batch = (num_class==1)
            if ind_batch.sum()>0:
                ind_batch = torch.nonzero(ind_batch).squeeze(1)
                x = x[ind_batch]
                sal = sal[ind_batch]
                label = label[ind_batch]

                center_loss1, center1 = self.centerLoss(x,sal)
                center_loss2, center2 = self.centerLoss(x,1-sal)


                cam_center_dist = (center2 - center1)*label
                cam_center_dist_loss = cam_center_dist.mean()

        return x_ori, center_loss1, center_loss2, cam_center_dist_loss

    def get_heatmaps(self):
        return self.map1

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for name, value in self.named_parameters():

            if 'extra' in name:
                if 'weight' in name:
                    groups[2].append(value)
                else:
                    groups[3].append(value)
            else:
                if 'weight' in name:
                    groups[0].append(value)
                else:
                    groups[1].append(value)
        return groups
    
    def getFeatures(self, fts, mask):
        fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        masked_fts = torch.sum(fts * mask.unsqueeze(1), dim=(2, 3)) \
            / (mask.unsqueeze(1).sum(dim=(2, 3)) + 1e-5) # N x C
        return masked_fts


    def centerLoss(self, x, sal):
        center = self.getFeatures(x,sal)
        center_copy = center
        x = F.interpolate(x, size=sal.shape[-2:], mode='bilinear')
        n,c,h,w = x.shape
        center = center.view(n,c,1,1)
        center = center.repeat(1,1,h,w)
        diff = x - center
        sal = sal.unsqueeze(1)
        masked_diff = diff * sal
        masked_diff = torch.pow(masked_diff,2)
        center_loss = torch.sum(masked_diff, dim=(2, 3)) / (sal.sum(dim=(2, 3)) + 1e-5) # N x C
        center_loss = center_loss.mean()
        return center_loss, center_copy


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'N':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        else:
            if i > 13:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, dilation=2, padding=2)            
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'D1': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'N', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(pretrained=False, **kwargs):
    model = VGG(make_layers(cfg['D1']), **kwargs)  
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model
