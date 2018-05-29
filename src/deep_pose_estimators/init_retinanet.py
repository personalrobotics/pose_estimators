#!/usr/bin/env python

'''Init RestinaNet50 with pretrained ResNet50 model.

Download pretrained ResNet50 params from:
  https://download.pytorch.org/models/resnet50-19c8e357.pth
'''

import torch
import torch.nn as nn

import math
import os
import sys

from model.fpn import FPN50
from model.retinanet import RetinaNet


model_base_dir = './pretrained'
model_filename = os.path.join(model_base_dir, 'resnet50.pth')
model_url = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'

net_filename = os.path.join(model_base_dir, 'net_new.pth')


def load_pretrained_model():
    if not os.path.isdir(model_base_dir):
        os.makedirs(model_base_dir)

    if not os.path.exists(model_filename):
        print('Cannot find saved params: {}'.format(model_filename))
        try:
            import urllib.request as urll
        except ImportError:
            import urllib as urll

        print('Downloading pretrained ResNet50 params from {}'.format(
            model_url))
        urll.urlretrieve(model_url, model_filename)
        print('Saved in {}'.format(model_filename))

    return torch.load(model_filename)


def import_pretrained_resnet():
    print('Loading pretrained ResNet50 model..')

    d = load_pretrained_model()

    print('Loading into FPN50..')
    fpn = FPN50()
    dd = fpn.state_dict()
    for k in d.keys():
        if not k.startswith('fc'):  # skip fc layers
            dd[k] = d[k]

    print('Saving RetinaNet..')
    net = RetinaNet(num_classes=49)
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    pi = 0.01
    nn.init.constant_(net.cls_head[-1].bias, -math.log((1 - pi) / pi))

    net.fpn.load_state_dict(dd)
    torch.save(net.state_dict(), net_filename)
    print('Done!')


if __name__ == '__main__':
    import_pretrained_resnet()
