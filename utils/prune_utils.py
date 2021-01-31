import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from collections import OrderedDict
from torch.nn.modules.utils import _pair


# ======================================================
# Find layer dependency
# Update input indices (adapt to previous layers)
# Update output indices
# ======================================================
def update_resnet_block_dependencies(prev_modules, block, dependencies):
    for m in prev_modules:
        assert isinstance(m, (nn.Conv2d, nn.Linear)), 'Only conv or linear layer can be previous modules.'
    dependencies[block.conv1] = prev_modules
    dependencies[block.bn1] = [block.conv1]
    dependencies[block.conv2] = [block.conv1]
    dependencies[block.bn2] = [block.conv2]

    if block.downsample is not None:
        dependencies[block.downsample[0]] = prev_modules
        dependencies[block.downsample[1]] = [block.downsample[0]]


def update_resnet_layer_dependencies(prev_modules, layer, dependencies):
    num_blocks = len(layer)
    for block_idx in range(num_blocks):
        block = layer[block_idx]
        update_resnet_block_dependencies(prev_modules, block, dependencies)
        prev_modules = [block.conv2]
        if block.downsample is not None:
            prev_modules.append(block.downsample[0])
        else:
            prev_modules.extend(dependencies[block.conv1])


def update_resnet_bottle_block_dependencies(prev_modules, block, dependencies):
    for m in prev_modules:
        assert isinstance(m, (nn.Conv2d, nn.Linear)), 'Only conv or linear layer can be previous modules.'
    dependencies[block.conv1] = prev_modules
    dependencies[block.bn1] = [block.conv1]
    dependencies[block.conv2] = [block.conv1]
    dependencies[block.bn2] = [block.conv2]
    dependencies[block.conv3] = [block.conv2]
    dependencies[block.bn3] = [block.conv3]

    if block.downsample is not None:
        dependencies[block.downsample[0]] = prev_modules
        dependencies[block.downsample[1]] = [block.downsample[0]]


def update_resnet_bottle_layer_dependencies(prev_modules, layer, dependencies):
    num_blocks = len(layer)
    for block_idx in range(num_blocks):
        block = layer[block_idx]
        update_resnet_bottle_block_dependencies(prev_modules, block, dependencies)
        prev_modules = [block.conv3]
        if block.downsample is not None:
            prev_modules.append(block.downsample[0])
        else:
            prev_modules.extend(dependencies[block.conv1])


def update_presnet_block_dependencies(prev_modules, block, dependencies):
    # TODO: presnet
    for m in prev_modules:
        assert isinstance(m, (nn.Conv2d, nn.Linear)), 'Only conv or linear layer can be previous modules.'
    dependencies[block.bn1] = prev_modules
    dependencies[block.conv1] = prev_modules
    dependencies[block.bn2] = [block.conv1]
    dependencies[block.conv2] = [block.conv1]
    dependencies[block.bn3] = [block.conv2]
    dependencies[block.conv3] = [block.conv2]

    if block.downsample is not None:
        dependencies[block.downsample[0]] = prev_modules


def update_presnet_layer_dependencies(prev_modules, layer, dependencies):
    # TODO: presnet
    num_blocks = len(layer)
    for block_idx in range(num_blocks):
        block = layer[block_idx]
        update_presnet_block_dependencies(prev_modules, block, dependencies)
        prev_modules = [block.conv3]
        if block.downsample is not None:
            prev_modules.append(block.downsample[0])
        else:
            prev_modules.extend(dependencies[block.bn1])


def get_layer_dependencies(model, network):
    # Helper function; ad-hoc fix
    dependencies = OrderedDict()
    if 'vgg' in network:
        modules = model.modules()
        prev_layers = []
        for m in modules:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                dependencies[m] = prev_layers
                prev_layers = [m]
            elif isinstance(m, nn.BatchNorm2d):
                dependencies[m] = prev_layers
    elif 'presnet' in network:
        dependencies[model.module.conv1] = []
        prev_modules = [model.module.conv1]

        # update first layer's dependencies
        update_presnet_layer_dependencies(prev_modules, model.module.layer1, dependencies)

        # update second layer's dependencies
        prev_modules = [model.module.layer1[-1].conv3]
        if model.module.layer1[-1].downsample is not None:
            prev_modules.append(model.module.layer1[-1].downsample[0])
        else:
            prev_modules = [model.module.layer1[-1].conv3] + dependencies[model.module.layer1[-1].bn1]
        update_presnet_layer_dependencies(prev_modules, model.module.layer2, dependencies)

        # update third layer's dependencies
        prev_modules = [model.module.layer2[-1].conv3]
        if model.module.layer2[-1].downsample is not None:
            prev_modules.append(model.module.layer2[-1].downsample[0])
        else:
            prev_modules = [model.module.layer2[-1].conv3] + dependencies[model.module.layer2[-1].bn1]
        update_presnet_layer_dependencies(prev_modules, model.module.layer3, dependencies)

        # update bn and fc layer's dependencies
        prev_modules = [model.module.layer3[-1].conv3]
        if model.module.layer3[-1].downsample is not None:
            prev_modules.append(model.module.layer3[-1].downsample[0])
        else:
            prev_modules = [model.module.layer3[-1].conv3] + dependencies[model.module.layer3[-1].bn1]
        dependencies[model.module.bn] = prev_modules
        dependencies[model.module.fc] = prev_modules


    elif 'bottle' in network:
        dependencies[model.module.conv1] = []
        dependencies[model.module.bn1] = [model.module.conv1]

        prev_modules = [model.module.conv1]
        update_resnet_bottle_layer_dependencies(prev_modules, model.module.layer1, dependencies)

        prev_modules = [model.module.layer1[-1].conv3]
        if model.module.layer1[-1].downsample is not None:
            prev_modules.append(model.module.layer1[-1].downsample[0])
        else:
            prev_modules = [model.module.layer1[-1].conv3] + dependencies[model.module.layer1[-1].conv1]
        update_resnet_bottle_layer_dependencies(prev_modules, model.module.layer2, dependencies)

        prev_modules = [model.module.layer2[-1].conv3]
        if model.module.layer2[-1].downsample is not None:
            prev_modules.append(model.module.layer2[-1].downsample[0])
        else:
            prev_modules = [model.module.layer2[-1].conv3] + dependencies[model.module.layer2[-1].conv1]
        update_resnet_bottle_layer_dependencies(prev_modules, model.module.layer3, dependencies)

        prev_modules = [model.module.layer3[-1].conv3]
        if model.module.layer3[-1].downsample is not None:
            prev_modules.append(model.module.layer3[-1].downsample[0])
        else:
            prev_modules = [model.module.layer3[-1].conv3] + dependencies[model.module.layer3[-1].conv1]
        dependencies[model.module.fc] = prev_modules
        update_resnet_bottle_layer_dependencies(prev_modules, model.module.layer4, dependencies)

        prev_modules = [model.module.layer4[-1].conv3]
        if model.module.layer4[-1].downsample is not None:
            prev_modules.append(model.module.layer4[-1].downsample[0])
        else:
            prev_modules = [model.module.layer4[-1].conv3] + dependencies[model.module.layer4[-1].conv1]
        dependencies[model.module.fc] = prev_modules



    elif 'resnet' in network:
        dependencies[model.module.conv1] = []
        dependencies[model.module.bn1] = [model.module.conv1]

        prev_modules = [model.module.conv1]
        update_resnet_layer_dependencies(prev_modules, model.module.layer1, dependencies)

        prev_modules = [model.module.layer1[-1].conv2]
        if model.module.layer1[-1].downsample is not None:
            prev_modules.append(model.module.layer1[-1].downsample[0])
        else:
            prev_modules = [model.module.layer1[-1].conv2] + dependencies[model.module.layer1[-1].conv1]
        update_resnet_layer_dependencies(prev_modules, model.module.layer2, dependencies)

        prev_modules = [model.module.layer2[-1].conv2]
        if model.module.layer2[-1].downsample is not None:
            prev_modules.append(model.module.layer2[-1].downsample[0])
        else:
            prev_modules = [model.module.layer2[-1].conv2] + dependencies[model.module.layer2[-1].conv1]
        update_resnet_layer_dependencies(prev_modules, model.module.layer3, dependencies)

        prev_modules = [model.module.layer3[-1].conv2]
        if model.module.layer3[-1].downsample is not None:
            prev_modules.append(model.module.layer3[-1].downsample[0])
        else:
            prev_modules = [model.module.layer3[-1].conv2] + dependencies[model.module.layer3[-1].conv1]
        dependencies[model.module.fc] = prev_modules
        try:
            update_resnet_layer_dependencies(prev_modules, model.module.layer4, dependencies)
            prev_modules = [model.module.layer4[-1].conv2]
            if model.module.layer4[-1].downsample is not None:
                prev_modules.append(model.module.layer4[-1].downsample[0])
            else:
                prev_modules = [model.module.layer4[-1].conv2] + dependencies[model.module.layer4[-1].conv1]
            dependencies[model.module.fc] = prev_modules
        except Exception as e:
            pass


    return dependencies


def update_indices(model, network):
    print("updating indices")
    dependencies = get_layer_dependencies(model, network)
    update_out_indices(model, dependencies)
    update_in_dinces(dependencies)


def update_out_indices(model, dependencies):
    pass


def update_in_dinces(dependencies):
    for m, deps in dependencies.items():
        if len(deps) > 0:
            indices = set()
            for d in deps:
                indices = indices.union(d.out_indices)
            m.in_indices = sorted(list(indices))


# ======================================================
# For building vgg net: generate cfgs and generate mask
# as well as copying weights.
# ======================================================
def gen_network_cfgs(filter_nums, network):
    # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    if network == 'vgg19':
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
        counts = 0
        for idx in range(len(cfg)):
            c = cfg[idx]
            if c == 'M':
                counts += 1
                continue
            cfg[idx] = filter_nums[idx-counts]
    else:
        raise NotImplementedError
    return cfg


def copy_weights(m0, m1, ):
    if isinstance(m0, nn.BatchNorm2d):
        pass
    elif isinstance(m0, nn.Conv2d):
        pass
    elif isinstance(m0, nn.Linear):
        pass
    else:
        raise NotImplementedError


def get_threshold(values, percentage):
    v_sorted = sorted(values)
    n = int(len(values) * percentage)
    threshold = v_sorted[n]
    return threshold


def filter_indices(values, threshold):
    indices = []
    for idx, v in enumerate(values):
        if v > threshold:
            indices.append(idx)
    if len(indices) <= 1:
        # we make it at least 1 filters in each laer
        indices = [0]
    return indices


def filter_indices_ni(values, threshold, ni_threshold):
    ni_indices = []
    pruned_indices = []
    remained_indices = []
    for idx, v in enumerate(values):
        if v > threshold:
            remained_indices.append(idx)
        elif v > ni_threshold and v<= threshold:
            ni_indices.append(idx)
        else:
            pruned_indices.append(idx)
    if len(remained_indices) <= 1:
        # we make it at least 1 filters in each laer
        remained_indices = [0]
        try:
            ni_indices.remove(0)
        except Exception as e:
            pruned_indices.remove(0)
    return remained_indices, ni_indices, pruned_indices


def get_rotation_layer_weights(model, qm):
    for m in model.modules():
        if (isinstance(m, nn.Sequential)
                and len(m) == 3
                and isinstance(m[0], (LinearLayerRotation, ConvLayerRotation))
                and isinstance(m[2], (LinearLayerRotation, ConvLayerRotation))):
            if qm is m[1]:
                return m[0].rotation_matrix.data, m[2].rotation_matrix.data
    raise ValueError('%s not found in the model. Potential bug!' % qm)


def update_QQ_dict(Q_g, Q_a, m, n):
    if n is not m:
        Q_g[n] = Q_g[m]
        Q_a[n] = Q_a[m]
        Q_a.pop(m)
        Q_g.pop(m)


def get_block_sum(m, imps):
    importances = []
    if isinstance(m, nn.Conv2d):
        kernel_size = m.kernel_size
        k = kernel_size[0] * kernel_size[1]
        l = imps.squeeze().size(0)
        bias = 1 if m.bias is not None else 0
        assert ((l-bias) // k) * k == (l-bias)
        for idx in range(0, l, k):
            s = min(idx+k, l)
            s = imps[idx:idx+k].sum().item()
            importances.append(s)
        return imps.new(importances)
    elif isinstance(m, nn.Linear):
        return imps


def count_module_params(m):
    counts = m.weight.view(-1).size(0)
    if m.bias is not None:
        counts += m.bias.size(0)
    return counts


class LinearLayerRotation(nn.Module):
    def __init__(self, rotation_matrix, bias=0, trainable=False):
        super(LinearLayerRotation, self).__init__()
        self.rotation_matrix = rotation_matrix
        self.rotation_matrix.requires_grad_(trainable)
        if trainable:
            self.rotation_matrix = nn.Parameter(self.rotation_matrix)

        self.trainable = trainable
        self.bias = bias

    def forward(self, x):
        if self.bias != 0:
            x = torch.cat([x, x.new(x.size(0), 1).fill_(self.bias)], 1)
        return x @ self.rotation_matrix

    def parameters(self):
        return [self.rotation_matrix]

    def extra_repr(self):
        return "in_features=%s, out_features=%s, trainable=%s" % (self.rotation_matrix.size(1),
                                                                  self.rotation_matrix.size(0),
                                                                  self.trainable)


class ConvLayerRotation(nn.Module):
    def __init__(self, rotation_matrix, bias=0, trainable=False):
        super(ConvLayerRotation, self).__init__()
        self.rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3)  # out_dim * in_dim
        self.rotation_matrix.requires_grad_(trainable)
        if trainable:
            self.rotation_matrix = nn.Parameter(self.rotation_matrix)
        self.trainable = trainable
        self.bias = bias

    def forward(self, x):
        # x: batch_size * in_dim * w * h
        if self.bias != 0:
            x = torch.cat([x, x.new(x.size(0), 1, x.size(2), x.size(3)).fill_(self.bias)], 1)
        return F.conv2d(x, self.rotation_matrix, None, _pair(1), _pair(0), _pair(1), 1)

    def parameters(self):
        return [self.rotation_matrix]

    def extra_repr(self):
        return "in_channels=%s, out_channels=%s, trainable=%s" % (self.rotation_matrix.size(1),
                                                                  self.rotation_matrix.size(0),
                                                                  self.trainable)


def register_bottleneck_layer(m, Q_g, Q_a, W_star, use_patch, trainable=False):
    assert use_patch
    if isinstance(m, nn.Linear):
        scale = nn.Linear(W_star.size(1), W_star.size(0), bias=False).cuda()
        scale.weight.data.copy_(W_star)
        bias = 1.0 if m.bias is not None else 0
        return nn.Sequential(
            LinearLayerRotation(Q_a, bias, trainable),
            scale,
            LinearLayerRotation(Q_g.t(), trainable=trainable))
    elif isinstance(m, nn.Conv2d):
        # if it is a conv layer, W_star should be out_c * in_c * h * w
        W_star = W_star.view(W_star.size(0), m.kernel_size[0], m.kernel_size[1], -1)
        W_star = W_star.transpose(2, 3).transpose(1, 2).contiguous()
        scale = nn.Conv2d(W_star.size(1), W_star.size(0), m.kernel_size,
                          m.stride, m.padding, m.dilation, m.groups, False).cuda()
        scale.weight.data.copy_(W_star)
        patch_size = m.kernel_size[0] * m.kernel_size[1]
        bias = 1.0/patch_size if m.bias is not None else 0
        return nn.Sequential(
            ConvLayerRotation(Q_a.t(), bias, trainable),
            scale,
            ConvLayerRotation(Q_g, trainable=trainable))
    else:
        raise NotImplementedError


# ====== for normalization ========
def normalize_factors(A, B):
    eps = 1e-10

    trA = torch.trace(A) + eps
    trB = torch.trace(B) + eps
    assert trA > 0, 'Must PD. A not PD'
    assert trB > 0, 'Must PD. B not PD'
    return A * (trB/trA)**0.5, B * (trA/trB)**0.5


# ====== Neuron Implant ====== #
def prune_model_ni(model):
    # Recursively prune the model
    if type(model) == nn.Conv2d:
        # return DW_NIConv2d(model)
        # return NIConv2d_fast(model)
        return NIConv2d(model)

    elif type(model) == nn.Sequential:
        mods = []
        for n, m in model.named_children():
            mods.append(prune_model_ni(m))
        return nn.Sequential(*mods)

    else:
        try:
            newmodel = copy.deepcopy(model)
        except Exception as e:
            print(model)
            print(e)
            exit()

        for attr in dir(model):
            mod = getattr(model, attr)
            if isinstance(mod, nn.Module) and 'norm' not in attr:
                setattr(newmodel, attr, prune_model_ni(mod))
        return newmodel


# ======= NI ======
class NIConv2d(nn.Module):
    def __init__(self, conv):
        super(NIConv2d, self).__init__()
        self.out_channels       = conv.out_channels
        self.in_channels        = conv.in_channels
        self.kernel_size        = conv.kernel_size
        self.stride             = conv.stride
        self.padding            = conv.padding
        self.pruned             = 1
        self.remained_indices   = conv.remained_indices
        self.ni_indices         = conv.ni_indices
        self.pruned_indices     = conv.pruned_indices

        self.in_indices         = conv.in_indices
        self.out_indices        = conv.out_indices

        if self.in_indices == None:
            self.in_indices = list(range(conv.weight.size(1)))
        self.in_channels        = len(self.in_indices)


        middle = (self.kernel_size[0]-1) // 2

        self.conv_indices = sorted(self.remained_indices + self.ni_indices)
        print(len(self.remained_indices), len(self.ni_indices), len(self.pruned_indices))

        self.conv1 = nn.Conv2d(self.in_channels, len(self.remained_indices), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True if conv.bias is not None else False)
        self.conv1.weight.data = conv.weight.data[self.remained_indices, :, :, :][:, self.in_indices,: ,:].clone()

        # new indices points to the position of 1x1 indices in conv_indices
        self.ni_indices_new = []
        if len(self.ni_indices)>0:
            for idx in self.ni_indices:
                self.ni_indices_new.append(self.conv_indices.index(idx))

        # new indices points to the position of 3x3 indices in conv_indices
        self.remained_indices_new = []
        if len(self.remained_indices)>0:
            for idx in self.remained_indices:
                self.remained_indices_new.append(self.conv_indices.index(idx))


        if len(self.ni_indices)>0:
            self.conv2 = nn.Conv2d(self.in_channels, len(self.ni_indices), kernel_size=1, stride=self.stride, bias=False)
            self.conv2.weight.data = conv.weight.data[self.ni_indices, :, middle:middle+1, middle:middle+1][:, self.in_indices,: ,:].clone().fill_(0)

        self.out_channels = len(self.conv_indices)

        if conv.bias is not None:
            self.conv1.bias.data = conv.bias.data[m.out_indices]
            self.conv1.bias.grad = None

    def forward(self, x):
        out1 = self.conv1(x)
        out = out1[:,0:1,:,:].expand(out1.shape[0], self.out_channels, out1.shape[2], out1.shape[3]).clone().fill_(0)
        out[:,self.remained_indices_new,:,:] = out1
        if len(self.ni_indices)>0:
            out2 = self.conv2(x)
            out[:,self.ni_indices_new,:,:] = out2
        return out


class NIConv2d_fast(nn.Module):
    def __init__(self, conv):
        super(NIConv2d_fast, self).__init__()
        self.out_channels       = conv.out_channels
        self.in_channels        = conv.in_channels
        self.kernel_size        = conv.kernel_size
        self.stride             = conv.stride
        self.padding            = conv.padding
        self.pruned             = 1
        self.remained_indices   = conv.remained_indices
        self.ni_indices         = conv.ni_indices
        self.pruned_indices     = conv.pruned_indices
        self.mask_indices       = []

        self.in_indices         = conv.in_indices
        self.out_indices        = conv.out_indices

        if self.in_indices == None:
            self.in_indices = list(range(conv.weight.size(1)))
        self.in_channels        = len(self.in_indices)


        middle = (self.kernel_size[0]-1) // 2

        conv_indices = sorted(self.remained_indices + self.ni_indices)
        print(len(self.remained_indices), len(self.ni_indices), len(self.pruned_indices) )

        self.conv1 = nn.Conv2d(self.in_channels, len(conv_indices), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True if conv.bias is not None else False)
        self.conv1.weight.data = conv.weight.data[conv_indices, :, :, :][:, self.in_indices,: ,:].clone()

        if len(self.ni_indices)>0:
            for idx in self.ni_indices:
                self.mask_indices.append(conv_indices.index(idx))
            tmp_mask = torch.ones(self.conv1.weight.data.shape)
            tmp_mask[self.mask_indices] = 0
            tmp_mask[self.mask_indices, :, middle, middle] = 1

            self.register_buffer("conv_mask", tmp_mask)


        self.out_channels = len(sorted(self.remained_indices + self.ni_indices))

        if conv.bias is not None:
            self.conv1.bias.data = conv.bias.data[m.out_indices]
            self.conv1.bias.grad = None

    def forward(self, x):
        w = self.conv1.weight
        if len(self.ni_indices)>0:
            w = w * self.conv_mask.cuda()
        res =  F.conv2d(x, w, self.conv1.bias, self.conv1.stride, self.conv1.padding,
                        self.conv1.dilation, self.conv1.groups)
        return res


class DW_NIConv2d(nn.Module):
    def __init__(self, conv):
        super(DW_NIConv2d, self).__init__()
        self.out_channels       = conv.out_channels
        self.in_channels        = conv.in_channels
        self.kernel_size        = conv.kernel_size
        self.stride             = conv.stride
        self.padding            = conv.padding
        self.pruned             = 1
        self.remained_indices   = conv.remained_indices
        self.ni_indices         = conv.ni_indices
        self.pruned_indices     = conv.pruned_indices

        self.in_indices         = conv.in_indices
        self.out_indices        = conv.out_indices

        self.use_depthwise      = (conv.weight.size(0)==conv.weight.size(1))

        if self.in_indices == None:
            self.in_indices = list(range(conv.weight.size(1)))


        self.in_channels        = len(self.in_indices)


        middle = (self.kernel_size[0]-1) // 2

        print(len(self.remained_indices), len(self.ni_indices), len(self.pruned_indices))

        self.conv1 = nn.Conv2d(self.in_channels, len(self.remained_indices), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True if conv.bias is not None else False)
        self.conv1.weight.data = conv.weight.data[self.remained_indices, :, :, :][:, self.in_indices,: ,:].clone()

        # new indices points to the position of 1x1 indices in conv_indices
        self.ni_indices_in  = []
        self.ni_indices_out = []
        self.ni_indices_new = []
        if len(self.ni_indices)>0:
            for idx in self.ni_indices:
                self.ni_indices_new.append(self.out_indices.index(idx))
                if idx in self.in_indices:
                    self.ni_indices_out.append(self.out_indices.index(idx))
                    self.ni_indices_in.append(self.in_indices.index(idx))

        # new indices points to the position of 3x3 indices in conv_indices
        self.remained_indices_new = []
        if len(self.remained_indices)>0:
            for idx in self.remained_indices:
                self.remained_indices_new.append(self.out_indices.index(idx))


        if len(self.ni_indices)>0:
            self.pw_conv = nn.Conv2d(self.in_channels, len(self.ni_indices), kernel_size=1, stride=self.stride, bias=False)
            self.pw_conv.weight.data = conv.weight.data[self.ni_indices, :, middle:middle+1, middle:middle+1][:, self.in_indices,: ,:].clone().fill_(0)
            if self.use_depthwise and len(self.ni_indices_in):
                self.dw_conv = nn.Conv2d(len(self.ni_indices_in), len(self.ni_indices_in), groups=len(self.ni_indices_in), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=True if conv.bias is not None else False)
                self.dw_conv.weight.data = conv.weight.data[self.ni_indices_in, :, :, :].mean(1, keepdim=True).clone().fill_(0)

        self.out_channels = len(self.out_indices)

        if conv.bias is not None:
            self.conv1.bias.data = conv.bias.data[m.out_indices]
            self.conv1.bias.grad = None

    def forward(self, x):
        out1 = self.conv1(x)
        out = out1[:,0:1,:,:].expand(out1.shape[0], self.out_channels, out1.shape[2], out1.shape[3]).clone().fill_(0)
        out[:,self.remained_indices_new,:,:] = out1
        if len(self.ni_indices)>0:
            out2_1 = self.pw_conv(x)
            out[:,self.ni_indices_new,:,:] = out2_1
            if self.use_depthwise and len(self.ni_indices_in):
                out2_2 = self.dw_conv(x[:,self.ni_indices_in,:,:])
                out[:,self.ni_indices_out,:,:] = out2_2
        return out
