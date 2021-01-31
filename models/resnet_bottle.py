import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from utils.prune_utils import (ConvLayerRotation,
                               LinearLayerRotation,
                               register_bottleneck_layer,
                               update_QQ_dict)
from utils.common_utils import try_cuda


__all__ = ['resnet_bottle', 'BottleneckResNet_bottle']  # , 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
_AFFINE = True
#_AFFINE = False

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
    elif isinstance(m, LinearLayerRotation):
        if m.trainable:
            print('* init Linear rotation')
            init.kaiming_normal(m.rotation_matrix)
    elif isinstance(m, ConvLayerRotation):
        if m.trainable:
            print('* init Conv rotation')
            init.kaiming_normal(m.rotation_matrix)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=False):
        super(BasicBlock, self).__init__()

        mid_channels = planes // 4

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.conv3 = nn.Conv2d(planes, planes*self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes*self.expansion, affine=_AFFINE)

        self.downsample = None
        self.bn4 = None
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes*self.expansion, kernel_size=1, stride=stride, bias=False))
            self.bn4 = nn.BatchNorm2d(planes*self.expansion, affine=_AFFINE)

        self.speed_tensor = None
        self.create_flag = True
        self.speed_tensor_indices = [[], []]

    def forward(self, x):
        ##############
        # This is a new version
        ##############

        # x: batch_size * in_c * h * w
        is_pruned = hasattr(self.conv1, 'in_indices')

        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.bn4(self.downsample(x))

        if is_pruned:

            indices = []
            indices.append(self.conv3.out_indices)

            if self.downsample is not None:
                indices.append(self.downsample[0].out_indices)
            else:
                indices.append(self.conv1.in_indices)

            n_c = len(set(indices[0] + indices[1]))
            all_indices = list(set(indices[0] + indices[1]))
            # the following part is used for iterative pruning
            if self.create_flag or (not set(self.speed_tensor_indices[1]) == set(all_indices)) or (not set(self.speed_tensor_indices[0]) == set(self.conv1.in_indices)) or (self.speed_tensor.size(0) < x.size(0)):

                self.speed_tensor_indices[0] = self.conv1.in_indices
                self.speed_tensor_indices[1] = all_indices
                self.create_flag = False

                self.r_indices = []
                self.o_indices = []

                for i in range(n_c):
                    idx = all_indices[i]
                    if idx in indices[0] and idx in indices[1]:
                        self.r_indices.append(i)
                        self.o_indices.append(i)
                    elif idx in indices[0]:
                        self.o_indices.append(i)
                    elif idx in indices[1]:
                        self.r_indices.append(i)
                self.speed_tensor = try_cuda(torch.zeros(x.size(0), n_c, residual.size(2), residual.size(3)))



        if is_pruned:
            tmp_tensor = self.speed_tensor[:x.size(0), :, :, :] + 0. # +0 is used for preventing copy issue
            tmp_tensor[:, self.r_indices, :, :] += residual
            tmp_tensor[:, self.o_indices, :, :] += out
            out = tmp_tensor
        else:
            out += residual
        out = F.relu(out)
        return out




class ResNet_bottle(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet_bottle, self).__init__()
        _outputs = [64, 128, 256, 512]
        self.in_planes = _outputs[0]

        self.conv1 = nn.Conv2d(3, _outputs[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(_outputs[0], affine=_AFFINE)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # previous stride is 2
        self.layer1 = self._make_layer(block, _outputs[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, _outputs[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, _outputs[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, _outputs[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(_outputs[3]*block.expansion, num_classes)


        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [1]*(num_blocks-1)
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample=True))
        self.in_planes = planes * block.expansion
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet_bottle(depth=50, dataset='imagenet'):

    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == "imagenet":
        num_classes = 1000
    else:
        raise NotImplementedError
    if depth==50:
        model =  ResNet_bottle(BasicBlock, [3,4,6,3], num_classes)
    return model


class BottleneckResNet_bottle(nn.Module):
    def __init__(self, net_prev, fix_rotation=True):
        super(BottleneckResNet_bottle, self).__init__()
        self.conv1 = net_prev.conv1
        self.bn = net_prev.bn
        self.layer1 = net_prev.layer1
        self.layer2 = net_prev.layer2
        self.layer3 = net_prev.layer3
        self.linear = net_prev.linear

        self.fix_rotation = fix_rotation
        self._is_registered = False

    def _update_bottleneck(self, bneck, modules, Q_g, Q_a, W_star, use_patch, fix_rotation):
        m = bneck.conv1
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            bneck.conv1 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, bneck.conv1[1])

        m = bneck.conv2
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            bneck.conv2 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, bneck.conv2[1])

        m = bneck.downsample
        if m is not None:
            if len(m) == 1 and m[0] in modules:
                m = m[0]
                bneck.downsample = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
                update_QQ_dict(Q_g, Q_a, m, bneck.downsample[1])
            elif len(m) == 3 and m[1] in modules:
                m = m[1]
                bneck.downsample = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
                update_QQ_dict(Q_g, Q_a, m, bneck.downsample[1])
            else:
                assert len(m) == 1 or len(m) == 3, 'Upexpected layer %s' % m

    def register(self, modules, Q_g, Q_a, W_star, use_patch, fix_rotation, re_init):
        for m in self.modules():
            if isinstance(m, BasicBlock):
                self._update_bottleneck(m, modules, Q_g, Q_a, W_star, use_patch, fix_rotation)

        m = self.conv1
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            self.conv1 = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, self.conv1[1])

        m = self.linear
        if isinstance(m, nn.Sequential):
            m = m[1]
        if m in modules:
            self.linear = register_bottleneck_layer(m, Q_g[m], Q_a[m], W_star[m], use_patch, fix_rotation)
            update_QQ_dict(Q_g, Q_a, m, self.linear[1])
        self._is_registered = True
        if re_init:
            self.apply(_weights_init)

    def forward(self, x):
        assert self._is_registered
        out = F.relu(self.bn(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
