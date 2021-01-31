'''Train CIFAR10/CIFAR100 with PyTorch.'''
from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.network_utils import get_network
from utils.data_utils import get_dataloader
from utils.common_utils import PresetLRScheduler, makedirs

from utils.compute_flops import print_model_param_flops, print_model_param_flops

import numpy as np


def count_parameters(model):
    """The number of trainable parameters.
    It will exclude the rotation matrix in bottleneck layer.
    If those parameters are not trainiable.
    """
    return sum(p.numel() for p in model.parameters())


# fetch args
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', default=0.1, type=float)
parser.add_argument('--weight_decay', default=3e-3, type=float)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--network', default='vgg', type=str)
parser.add_argument('--depth', default=19, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--epoch', default=150, type=int)
parser.add_argument('--decay_every', default=60, type=int)
parser.add_argument('--decay_ratio', default=0.1, type=float)
parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--resume', '-r', default=None, type=str)
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--log_dir', default='cifar10_result/pretrain', type=str)
args = parser.parse_args()

# init model
net = get_network(network=args.network,
                  depth=args.depth,
                  dataset=args.dataset)
print(net)
# net = net.to(args.device)
net = nn.DataParallel(net).to(args.device)

# init dataloader
dataset = 'imagenet_vgg' if args.dataset == 'imagenet' and args.network == 'vgg' else args.dataset
trainloader, testloader = get_dataloader(dataset=dataset,
                                         train_batch_size=args.batch_size,
                                         test_batch_size=256)

# init optimizer and lr scheduler
optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
lr_schedule = {0: args.learning_rate,
               int(args.epoch*0.5): args.learning_rate*0.1,
               int(args.epoch*0.75): args.learning_rate*0.01}
lr_scheduler = PresetLRScheduler(lr_schedule)
# lr_scheduler = #StairCaseLRScheduler(0, args.decay_every, args.decay_ratio)

# init criterion
criterion = nn.CrossEntropyLoss()

start_epoch = 0
best_acc = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    # assert os.path.isdir('checkpoint/pretrain'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(f'{args.resume}')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print(args.dataset, args.network, args.depth)
    print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))
    raise Exception('Test for Acc.')

# init summary writter
log_dir = os.path.join(args.log_dir, '%s_%s%s' % (args.dataset,
                                                  args.network,
                                                  args.depth))
makedirs(log_dir)
writer = SummaryWriter(log_dir)

if args.dataset == 'tiny_imagenet':
    total_flops, rotation_flops = print_model_param_flops(net, 64, cuda=True)
elif args.dataset == 'imagenet':
    total_flops, rotation_flops = print_model_param_flops(net, 224, cuda=True)
else:
    total_flops, rotation_flops = print_model_param_flops(net, 32, cuda=True)
num_params = count_parameters(net)
print(f"Total Flops: {total_flops}")
print(f"Total Params: {num_params}")


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    lr_scheduler(optimizer, epoch)
    desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (lr_scheduler.get_lr(optimizer), 0, 0, correct, total))

    writer.add_scalar('train/lr', lr_scheduler.get_lr(optimizer), epoch)

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (lr_scheduler.get_lr(optimizer), train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)
    print(f'Train Loss: {train_loss/total}')
    print(f'Train Acc: {np.around(correct/total*100, 2)}')
    writer.add_scalar('train/loss', train_loss/(batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (lr_scheduler.get_lr(optimizer), test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (lr_scheduler.get_lr(optimizer), test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)
        print(f'Test Loss: {test_loss/total}')
        print(f'Test Acc: {np.around(correct/total*100, 2)}')
    # save checkpoint
    acc = 100.*correct/total

    writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('test/acc', 100. * correct / total, epoch)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'loss': loss,
            'args': args
        }
        if not os.path.isdir(f'{args.log_dir}'):
            os.mkdirs(f'{args.log_dir}')
        # if not os.path.isdir('checkpoint/pretrain'):
            # os.mkdir('checkpoint/pretrain')
        torch.save(state, f'{args.log_dir}/best.t7')
        best_acc = acc


for epoch in range(start_epoch, args.epoch):
    train(epoch)
    test(epoch)
