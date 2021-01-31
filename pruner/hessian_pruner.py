import torch
import torch.nn as nn
from collections import OrderedDict
from models.resnet import _weights_init
from utils.kfac_utils import fetch_mat_weights
from utils.common_utils import (tensor_to_list, PresetLRScheduler)
from utils.prune_utils import (filter_indices,
                               filter_indices_ni,
                               get_threshold,
                               update_indices,
                               normalize_factors,
                               prune_model_ni)
from utils.network_utils import stablize_bn
from tqdm import tqdm

from .hessian_fact import get_trace_hut
from .pyhessian import hessian
from .pyhessian import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal, cpu2gpu, gpu2cpu

import numpy as np
import time
import scipy.linalg
import os.path
from os import path


class HessianPruner:

        def __init__(self,
                     model,
                     builder,
                     config,
                     writer,
                     logger,
                     prune_ratio_limit,
                     network,
                     batch_averaged=True,
                     use_patch=False,
                     fix_layers=0,
                     hessian_mode='Trace',
                     use_decompose=False):
            print('Using patch is %s' % use_patch)
            self.iter = 0
            self.logger = logger
            self.writer = writer
            self.config = config
            self.prune_ratio_limit = prune_ratio_limit
            self.network = network

            self.batch_averaged = batch_averaged
            self.use_decompose = use_decompose
            self.known_modules = {'Linear', 'Conv2d'}
            if self.use_decompose:
                self.known_modules = {'Conv2d'}
            self.modules = []
            self.model = model
            self.builder = builder
            self.fix_layers = fix_layers
            self.steps = 0
            self.use_patch = False  # use_patch

            self.W_pruned = {}
            self.S_l = None

            self.hessian_mode = hessian_mode

            self.importances = {}
            self._inversed = False
            self._cfgs = {}
            self._indices = {}

        def make_pruned_model(self, dataloader, criterion, device, fisher_type, prune_ratio, is_loader=False, normalize=True, re_init=False, n_v=300):
            self.prune_ratio = prune_ratio # use for some special case, particularly slq_full, slq_layer
            self._prepare_model()
            self.init_step()
            if self.config.dataset == 'imagenet':
                is_loader = True
            self._compute_hessian_importance(dataloader, criterion, device, is_loader, n_v=n_v)

            if self.use_decompose:
                self._do_prune_ni(prune_ratio, self.config.ni_ratio ,re_init)
                self._build_pruned_model_ni(re_init)
            else:
                self._do_prune(prune_ratio, re_init)
                self._build_pruned_model(re_init)

            self._rm_hooks()
            self._clear_buffer()
            return str(self.model)

        def _prepare_model(self):
            count = 0
            for module in self.model.modules():
                classname = module.__class__.__name__
                if classname in self.known_modules:
                    self.modules.append(module)
                    count += 1
            self.modules = self.modules[self.fix_layers:]

        def _compute_hessian_importance(self, dataloader, criterion, device, is_loader, n_v=300):
            ###############
            # Here, we use the fact that Conv does not have bias term
            ###############
            if self.hessian_mode == 'trace':
                for m in self.model.parameters():
                    # set requires_grad for convolution layers only
                    shape_list = [2, 4]
                    if self.use_decompose:
                        shape_list = [4]
                    if len(m.shape) in shape_list:
                        m.requires_grad = True
                    else:
                        m.requires_grad = False

                trace_dir = f"../HAPresults/{self.config.dataset}_result/{self.config.network}{self.config.depth}/tract.npy"
                print(trace_dir)
                if os.path.exists(trace_dir):
                    print(f"Loading trace from {trace_dir}")
                    results = np.load(trace_dir, allow_pickle=True)
                else:
                    results = get_trace_hut(self.model, dataloader, criterion, n_v=n_v, loader=is_loader, channelwise=True, layerwise=False)
                    np.save(trace_dir, results)


                for m in self.model.parameters():
                    m.requires_grad = True

                channel_trace, weighted_trace = [], []
                for k, layer in enumerate(results):
                    channel_trace.append(torch.zeros(len(layer)))
                    weighted_trace.append(torch.zeros(len(layer)))
                    for cnt, channel in enumerate(layer):
                        channel_trace[k][cnt] = sum(channel) / len(channel)

                for k, m in enumerate(self.modules):
                    tmp = []
                    for cnt, channel in enumerate(m.weight.data):
                        tmp.append( (channel_trace[k][cnt] * channel.detach().norm()**2 / channel.numel()).cpu().item())
                    self.importances[m] = (tmp, len(tmp))
                    self.W_pruned[m] = fetch_mat_weights(m, False)

            elif self.hessian_mode == 'random':
                # get uniform baseline
                for k, m in enumerate(self.modules):
                    tmp = []
                    for cnt, channel in enumerate(m.weight.data):
                        tmp.append( np.random.randn() )
                    self.importances[m] = (tmp, len(tmp))
                    self.W_pruned[m] = fetch_mat_weights(m, False)

        def _do_prune(self, prune_ratio, re_init):
            # get threshold
            all_importances = []
            for m in self.modules:
                imp_m = self.importances[m]
                imps = imp_m[0]
                all_importances += imps
            all_importances = sorted(all_importances)
            idx = int(prune_ratio * len(all_importances))
            threshold = all_importances[idx]

            threshold_recompute = get_threshold(all_importances, prune_ratio)
            idx_recomputed = len(filter_indices(all_importances, threshold))
            print('=> The threshold is: %.5f (%d), computed by function is: %.5f (%d).' %
                (threshold, idx, threshold_recompute, idx_recomputed))

            # do pruning
            print('=> Conducting network pruning. Max: %.5f, Min: %.5f, Threshold: %.5f' %
                (max(all_importances), min(all_importances), threshold))
            self.logger.info("[Weight Importances] Max: %.5f, Min: %.5f, Threshold: %.5f." %
                (max(all_importances), min(all_importances), threshold))

            for idx, m in enumerate(self.modules):
                imp_m = self.importances[m]
                n_r = imp_m[1]
                row_imps = imp_m[0]
                row_indices = filter_indices(row_imps, threshold)
                r_ratio = 1 - len(row_indices) / n_r

                # compute row indices (out neurons)
                if r_ratio > self.prune_ratio_limit:
                    r_threshold = get_threshold(row_imps, self.prune_ratio_limit)
                    row_indices = filter_indices(row_imps, r_threshold)
                    print('* row indices empty!')
                if isinstance(m, nn.Linear) and idx == len(self.modules) - 1:
                    row_indices = list(range(self.W_pruned[m].size(0)))

                m.out_indices = row_indices
                m.in_indices = None
            update_indices(self.model, self.network)

        def _build_pruned_model(self, re_init):
            for m_name, m in self.model.named_modules():
                if isinstance(m, nn.BatchNorm2d):
                    idxs = m.in_indices
                    m.num_features = len(idxs)
                    m.weight.data = m.weight.data[idxs]
                    m.bias.data = m.bias.data[idxs].clone()
                    m.running_mean = m.running_mean[idxs].clone()
                    m.running_var = m.running_var[idxs].clone()
                    m.weight.grad = None
                    m.bias.grad = None
                elif isinstance(m, nn.Conv2d):
                    in_indices = m.in_indices
                    if m.in_indices is None:
                        in_indices = list(range(m.weight.size(1)))
                    m.weight.data = m.weight.data[m.out_indices, :, :, :][:, in_indices, :, :].clone()
                    if m.bias is not None:
                        m.bias.data = m.bias.data[m.out_indices]
                        m.bias.grad = None
                    m.in_channels = len(in_indices)
                    m.out_channels = len(m.out_indices)
                    m.weight.grad = None

                elif isinstance(m, nn.Linear):
                    in_indices = m.in_indices
                    if m.in_indices is None:
                        in_indices = list(range(m.weight.size(1)))

                    m.weight.data = m.weight.data[m.out_indices, :][:, in_indices].clone()

                    if m.bias is not None:
                        m.bias.data = m.bias.data[m.out_indices].clone()
                        m.bias.grad = None

                    m.in_features = len(in_indices)
                    m.out_features = len(m.out_indices)
                    m.weight.grad = None
            if re_init:
                self.model.apply(_weights_init)


        def _do_prune_ni(self, prune_ratio, ni_ratio, re_init):
            # get threshold
            all_importances = []
            for m in self.modules:
                imp_m = self.importances[m]
                imps = imp_m[0]
                all_importances += imps
            all_importances = sorted(all_importances)
            idx = int(prune_ratio * len(all_importances))
            ni_idx = int( (1-ni_ratio) *prune_ratio * len(all_importances))
            threshold = all_importances[idx]
            ni_threshold  = all_importances[ni_idx]

            # do pruning
            print('=> Conducting network pruning. Max: %.5f, Min: %.5f, Threshold: %.5f' %
                (max(all_importances),  min(all_importances), threshold))
            self.logger.info("[Weight Importances] Max: %.5f, Min: %.5f, Threshold: %.5f." %
                (max(all_importances), min(all_importances), threshold))

            for idx, m in enumerate(self.modules):
                imp_m = self.importances[m]
                n_r = imp_m[1]
                row_imps = imp_m[0]
                remained_indices, ni_indices, pruned_indices = filter_indices_ni(row_imps, threshold, ni_threshold)
                r_ratio = (len(remained_indices) + len(ni_indices)) / n_r

                # compute row indices (out neurons)
                if r_ratio > self.prune_ratio_limit:
                    row_imps = sorted(row_imps)
                    idx = int(self.prune_ratio_limit * len(row_imps))
                    ni_idx = int( (1-ni_ratio) *prune_ratio * len(row_imps))
                    tmp_threshold = row_imps[idx]
                    tmp_ni_threshold  = row_imps[ni_idx]
                    remained_indices, ni_indices, pruned_indices = filter_indices_ni(row_imps, tmp_threshold, tmp_ni_threshold)
                    print('* row indices empty!')
                if isinstance(m, nn.Linear) and idx == len(self.modules) - 1:
                    row_indices = list(range(self.W_pruned[m].size(0)))

                m.remained_indices = remained_indices
                m.ni_indices       = ni_indices
                m.pruned_indices   = pruned_indices

                m.out_indices = sorted(m.remained_indices + m.ni_indices)
                m.in_indices = None
            update_indices(self.model, self.network)


        def _build_pruned_model_ni(self, re_init):
            for m in self.model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    idxs = m.in_indices
                    # print(len(idxs))
                    m.num_features = len(idxs)
                    m.weight.data = m.weight.data[idxs]
                    m.bias.data = m.bias.data[idxs].clone()
                    m.running_mean = m.running_mean[idxs].clone()
                    m.running_var = m.running_var[idxs].clone()
                    m.weight.grad = None
                    m.bias.grad = None

                elif isinstance(m, nn.Linear):
                    in_indices = m.in_indices
                    if m.in_indices is None:
                        in_indices = list(range(m.weight.size(1)))
                    m.weight.data = m.weight.data[:, in_indices].clone()

                    if m.bias is not None:
                        m.bias.data = m.bias.data.clone()
                        m.bias.grad = None

                    m.in_features = len(in_indices)
                    m.weight.grad = None

            self.model = prune_model_ni(self.model.module)
            if re_init:
                self.model.apply(_weights_init)

        def init_step(self):
            self.steps = 0

        def step(self):
            self.steps += 1

        def _rm_hooks(self):
            for m in self.model.modules():
                classname = m.__class__.__name__
                if classname in self.known_modules:
                    m._backward_hooks = OrderedDict()
                    m._forward_pre_hooks = OrderedDict()

        def _clear_buffer(self):
            self.m_aa = {}
            self.m_gg = {}
            self.d_a = {}
            self.d_g = {}
            self.Q_a = {}
            self.Q_g = {}
            self.modules = []
            if self.S_l is not None:
                self.S_l = {}

        def fine_tune_model(self, trainloader, testloader, criterion, optim, learning_rate, weight_decay, nepochs=10,
                            device='cuda'):
            self.model = self.model.train()
            self.model = self.model.cpu()
            self.model = self.model.to(device)

            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
            # optimizer = optim.Adam(self.model.parameters(), weight_decay=5e-4)
            if self.config.dataset == "cifar10":
                lr_schedule = {0: learning_rate,
                            int(nepochs * 0.5): learning_rate * 0.1,
                            int(nepochs * 0.75): learning_rate * 0.01}

            elif self.config.dataset == "imagenet":
                lr_schedule = {0 : learning_rate,
                   30: learning_rate * 0.1,
                   60: learning_rate * 0.01}


            lr_scheduler = PresetLRScheduler(lr_schedule)
            best_test_acc, best_test_loss = 0, 100
            iterations = 0

            for epoch in range(nepochs):
                self.model = self.model.train()
                correct = 0
                total = 0
                all_loss = 0
                lr_scheduler(optimizer, epoch)
                desc = ('[LR: %.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                lr_scheduler.get_lr(optimizer), 0, 0, correct, total))
                prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)

                for batch_idx, (inputs, targets) in prog_bar:
                    optimizer.zero_grad()
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    self.writer.add_scalar('train_%d/loss' % self.iter, loss.item(), iterations)
                    iterations += 1
                    all_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    desc = ('[%d][LR: %.5f, WD: %.5f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                            (epoch, lr_scheduler.get_lr(optimizer), weight_decay, all_loss / (batch_idx + 1),
                             100. * correct / total, correct, total))
                    prog_bar.set_description(desc, refresh=True)

                test_loss, test_acc, top5_acc = self.test_model(testloader, criterion, device)
                self.logger.info(f'{epoch} Test Loss: %.3f, Test Top1 %.2f%%(test), Test Top5 %.2f%%(test).' % (test_loss, test_acc, top5_acc))

                if test_acc > best_test_acc:
                    best_test_loss = test_loss
                    best_test_acc  = test_acc
                    network = self.config.network
                    depth   = self.config.depth
                    dataset = self.config.dataset
                    path = os.path.join(self.config.checkpoint, '%s_%s%s.pth.tar' % (dataset, network, depth))
                    save = {
                        'args': self.config,
                        'net': self.model,
                        'acc': test_acc,
                        'loss': test_loss,
                        'epoch': epoch
                    }
                    torch.save(save, path)
            print('** Finetuning finished. Stabilizing batch norm and test again!')
            stablize_bn(self.model, trainloader)
            test_loss, test_acc, top5_acc = self.test_model(testloader, criterion, device)
            best_test_loss = best_test_loss if best_test_acc > test_acc else test_loss
            best_test_acc = max(test_acc, best_test_acc)
            return best_test_loss, best_test_acc

        def test_model(self, dataloader, criterion, device='cuda'):
            self.model = self.model.eval()
            self.model = self.model.cpu()
            self.model = self.model.to(device)
            correct = 0
            top_1_correct = 0
            top_5_correct = 0
            total = 0
            all_loss = 0
            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (0, 0, correct, total))
            prog_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=desc, leave=True)
            for batch_idx, (inputs, targets) in prog_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                all_loss += loss.item()

                total += targets.size(0)
                _, pred = outputs.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))
                top_1_correct += correct[:1].contiguous().view(-1).float().sum(0)
                top_5_correct += correct[:5].contiguous().view(-1).float().sum(0)
                desc = ('Loss: %.3f | Top1: %.3f%% | Top5: %.3f%% ' %
                        (all_loss / (batch_idx + 1), 100. * top_1_correct / total, 100. * top_5_correct / total))

                prog_bar.set_description(desc, refresh=True)
            return all_loss / (batch_idx + 1), 100. * float(top_1_correct / total), 100. * float(top_5_correct / total)

        def speed_model(self, dataloader, criterion, device='cuda'):
            self.model = self.model.eval()
            self.model = self.model.cpu()
            self.model = self.model.to(device)
            # warm-up
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.model(inputs)
                    if batch_idx == 999:
                        break
            # time maesure
            start = time.time()
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(dataloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = self.model(inputs)
                    if batch_idx == 999:
                        break
            end = time.time()

            return end - start
