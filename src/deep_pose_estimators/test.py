#!/usr/bin/env python

import sys
import os
import argparse

import torch
import torchvision.transforms as transforms

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw, ImageFont


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Testing')
parser.add_argument('--gpus', '-g', default='0',
                    help='cuda visible devices')
parser.add_argument('--img_path', default='./image/000050.jpg',
                    help='test image path')
parser.add_argument('--num_classes', default=20, type=int,
                    help='number of classes')
parser.add_argument('--checkpoint', default='./checkpoint/ckpt.pth',
                    help='saved checkpoint path')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus


def run_test():
    print('Loading model..')
    net = RetinaNet(args.num_classes)

    ckpt = torch.load(args.checkpoint)
    net.load_state_dict(ckpt['net'])
    net.eval()
    net.cuda()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    print('Loading image..')
    img = Image.open(args.img_path)
    w, h = img.size

    print('Predicting..')
    x = transform(img)
    x = x.unsqueeze(0)
    with torch.no_grad():
        loc_preds, cls_preds = net(x.cuda())

        print('Decoding..')
        encoder = DataEncoder()
        boxes, labels = encoder.decode(
            loc_preds.cpu().data.squeeze(),
            cls_preds.cpu().data.squeeze(),
            (w, h))


        draw = ImageDraw.Draw(img)
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 15)
        for idx in range(len(boxes)):
            box = boxes[idx]
            label = labels[idx]
            draw.rectangle(list(box), outline='red')
            draw.text(list(box[:2]), str(label.item()), font=fnt, fill=(255, 0, 0, 255))
        img.show()


if __name__ == '__main__':
    run_test()
