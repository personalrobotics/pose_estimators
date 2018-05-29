#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division

import os
import math
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss.focal_loss import FocalLoss
from model.retinanet import RetinaNet
from datagen import ListDataset


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--gpus', '-g', default='0',
                    help='cuda visible devices')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--skip_checkpoint', '-s', action='store_true',
                    help='skip checkpoint and retrain')
parser.add_argument('--img_dir', default='./datasets/voc_all_images',
                    help='image directory path')
parser.add_argument('--train_list', default='./datasets/voc12_train.txt',
                    help='annotations for training dataset')
parser.add_argument('--test_list', default='./datasets/voc12_val.txt',
                    help='annotations for test dataset')
parser.add_argument('--train_batch_size', default=8, type=int,
                    help='batch size of training')
parser.add_argument('--test_batch_size', default=4, type=int,
                    help='batch size of testing')
parser.add_argument('--num_classes', default=20, type=int,
                    help='number of classes')
parser.add_argument('--net', default='./pretrained/net.pth',
                    help='saved state dict of the model')
parser.add_argument('--checkpoint', default='./checkpoint/ckpt.pth',
                    help='saved checkpoint path')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

best_loss = float('inf')  # best test loss


def run_train():
    assert torch.cuda.is_available(), 'Error: CUDA not found!'
    start_epoch = 0  # start from epoch 0 or last epoch

    # Data
    print('==> Preparing data..')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    trainset = ListDataset(
        root=args.img_dir,
        list_file=args.train_list,
        train=True, transform=transform, input_size=600)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.train_batch_size,
        shuffle=True, num_workers=8,
        collate_fn=trainset.collate_fn)

    testset = ListDataset(
        root=args.img_dir,
        list_file=args.test_list,
        train=False, transform=transform, input_size=600)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.test_batch_size,
        shuffle=False, num_workers=8,
        collate_fn=testset.collate_fn)

    # Model
    net = RetinaNet(num_classes=args.num_classes)
    net.load_state_dict(torch.load(args.net))

    if args.skip_checkpoint:
        print('==> Skipping checkpoint and retraining..')
    elif os.path.exists(args.checkpoint):
        print('==> Resuming from checkpoint \"{}\"..'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['net'])
        best_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch']

    net = torch.nn.DataParallel(
        net, device_ids=range(torch.cuda.device_count()))
    net.cuda()

    criterion = FocalLoss(num_classes=args.num_classes)
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=1e-4)

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        net.module.freeze_bn()
        train_loss = 0

        total_batches = int(math.ceil(
            trainloader.dataset.num_samples / trainloader.batch_size))

        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(trainloader):
            inputs = inputs.cuda()
            loc_targets = loc_targets.cuda()
            cls_targets = cls_targets.cuda()

            optimizer.zero_grad()
            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            print('[%d| %d/%d] train_loss: %.3f | avg_loss: %.3f' %
                  (epoch, batch_idx, total_batches,
                   loss.data, train_loss / (batch_idx + 1)))

    # Test
    def test(epoch):
        print('\nTest')
        net.eval()
        test_loss = 0

        total_batches = int(math.ceil(
            testloader.dataset.num_samples / testloader.batch_size))

        for batch_idx, (inputs, loc_targets, cls_targets) in enumerate(testloader):
            inputs = inputs.cuda()
            loc_targets = loc_targets.cuda()
            cls_targets = cls_targets.cuda()

            loc_preds, cls_preds = net(inputs)
            loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            test_loss += loss.data
            print('[%d| %d/%d] test_loss: %.3f | avg_loss: %.3f' %
                  (epoch, batch_idx, total_batches,
                   loss.data, test_loss / (batch_idx + 1)))

        # Save checkpoint
        global best_loss
        test_loss /= len(testloader)
        if test_loss < best_loss:
            print('Saving..')
            state = {
                'net': net.module.state_dict(),
                'loss': test_loss,
                'epoch': epoch,
            }
            if not os.path.exists(os.path.dirname(args.checkpoint)):
                os.makedirs(os.path.dirname(args.checkpoint))
            torch.save(state, args.checkpoint)
            best_loss = test_loss

    for epoch in range(start_epoch, start_epoch + 200):
        train(epoch)
        test(epoch)


if __name__ == '__main__':
    run_train()
