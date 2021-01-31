import argparse
import json
import time
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.resnet_imagenet import BottleneckResNetImagenet
running_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

from models import VGG
from pruner.hessian_pruner import HessianPruner

from tensorboardX import SummaryWriter
from utils.common_utils import (get_config_from_json,
                                get_logger,
                                makedirs,
                                process_config,
                                str_to_list)
from utils.data_utils import get_dataloader, get_hessianloader
from utils.network_utils import (get_bottleneck_builder,
                                 get_network)
from utils.prune_utils import (ConvLayerRotation,
                               LinearLayerRotation)
from utils.compute_flops import compute_model_param_flops
from models.resnet_imagenet import *


def count_parameters(model):
    """The number of trainable parameters.
    It will exclude the rotation matrix in bottleneck layer.
    If those parameters are not trainiable.
    """
    return sum(p.numel() for p in model.parameters())


def count_rotation_numels(model):
    """Count how many parameters in the rotation matrix.
    Call this only when they are not trainable for complementing
    the number of parameters.
    """
    total = 0
    for m in model.modules():
        if isinstance(m, (ConvLayerRotation, LinearLayerRotation)):
            total += m.rotation_matrix.numel()
    return total


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    config = process_config(args.config)

    return config


def init_dataloader(config):
    trainloader, testloader = get_dataloader(dataset=config.dataset,
                                             train_batch_size=config.batch_size,
                                             test_batch_size=256, returnset=config.data_distributed)
    return trainloader, testloader


def init_network(config, logger, device, imagenet=True):
    net = get_network(network=config.network,
                      depth=config.depth,
                      dataset=config.dataset)

    if imagenet:
        if config.network == "resnet" or config.network == "resnet_bottle":
            if config.depth==18:
                net = resnet18(pretrained=True)
            elif config.depth==50:
                net = resnet50(pretrained=True)
            bottleneck_net=BottleneckResNetImagenet

    else:
        print('==> Loading checkpoint from %s.' % config.load_checkpoint)
        logger.info('==> Loading checkpoint from %s.' % config.load_checkpoint)
        checkpoint = torch.load(config.load_checkpoint)
        if checkpoint.get('args', None) is not None:
            args = checkpoint['args']
            print('** [%s-%s%d] Acc: %.2f%%, Epoch: %d, Loss: %.4f' % (args.dataset, args.network, args.depth,
                                                                   checkpoint['acc'], checkpoint['epoch'],
                                                                   checkpoint['loss']))
            logger.info('** [%s-%s%d] Acc: %.2f%%, Epoch: %d, Loss: %.4f' % (args.dataset, args.network, args.depth,
                                                                         checkpoint['acc'], checkpoint['epoch'],
                                                                         checkpoint['loss']))
        state_dict = checkpoint['net'] if checkpoint.get('net', None) is not None else checkpoint['state_dict']
        for key in list(state_dict.keys()):
            if key.startswith('module'):
                state_dict[key[7:]] = state_dict[key]
                state_dict.pop(key)

        net.load_state_dict(state_dict)
        bottleneck_net = get_bottleneck_builder(config.network)

    if config.data_distributed:
        net = nn.parallel.DistributedDataParallel(net.cuda(), device_ids=[config.local_rank], output_device=config.local_rank)
        return net, bottleneck_net
    else:
        net = nn.DataParallel(net)

    return net.to(device), bottleneck_net


def init_pruner(net, bottleneck_net, config, writer, logger):
    if config.fisher_mode == 'eigen':
        pruner = KFACEigenPruner(net,
                                 bottleneck_net,
                                 config,
                                 writer,
                                 logger,
                                 config.prune_ratio_limit,
                                 batch_averaged=True,
                                 use_patch=config.get('use_patch', True),
                                 fix_layers=config.fix_layers,
                                 fix_rotation=config.fix_rotation)
    elif config.fisher_mode == 'hessian':
        pruner = HessianPruner(net,
                               VGG,
                               config,
                               writer,
                               logger,
                               config.prune_ratio_limit,
                               '%s%d'%(config.network, config.depth),
                               batch_averaged=True,
                               use_patch=False,
                               fix_layers=0,
                               hessian_mode=config.hessian_mode,
                               use_decompose=config.use_decompose)


    else:
        raise NotImplementedError

    return pruner


def init_summary_writer(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)
    print(config.checkpoint, os.path.exists(config.checkpoint))
    if not os.path.exists(config.checkpoint):
        os.makedirs(config.checkpoint)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_model = os.path.join(path, 'models/%s.py' % config.network)
    path_main = os.path.join(path, 'main_prune.py')
    path_pruner = os.path.join(path, 'pruner/%s.py' % config.pruner)

    logger = get_logger(f'log{running_time}.log_time', logpath=config.saving_log,
                        filepath=path_model, package_files=[path_main, path_pruner])
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)

    return logger, writer


def save_model(config, iteration, pruner, cfg, stat):
    network = config.network
    depth = config.depth
    dataset = config.dataset
    path = os.path.join(config.checkpoint, '%s_%s%s_%d.pth.tar' % (dataset, network, depth, iteration))
    save = {
        'config': config,
        'net': pruner.model,
        'cfg': cfg,
        'stat': stat
    }
    torch.save(save, path)


def compute_ratio(model, total, fix_rotation, logger):
    indicator = 1 if fix_rotation else 0
    rotation_numel = count_rotation_numels(model)
    pruned_numel = count_parameters(model) + rotation_numel*indicator
    ratio = 100. * pruned_numel / total
    logger.info('Compression ratio: %.2f%%(%d/%d), Total: %d, Rotation: %d.' % (ratio,
                                                                                pruned_numel,
                                                                                total,
                                                                                pruned_numel,
                                                                                rotation_numel))
    unfair_ratio = 100 - 100. * (pruned_numel - rotation_numel*indicator)
    return ratio, unfair_ratio, pruned_numel, rotation_numel


def main(config):
    stats = {}
    if config.data_distributed:
        torch.distributed.init_process_group(backend="nccl")
    device = torch.device('cuda:0,1')
    criterion = torch.nn.CrossEntropyLoss()

    logger, writer = init_summary_writer(config)
    trainloader, testloader = init_dataloader(config)

    if config.data_distributed:
        trainset, testset = trainloader, testloader
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
        trainloader = torch.utils.data.DataLoader(trainset, config.batch_size, False, num_workers=config.num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
        testloader = torch.utils.data.DataLoader(testset, config.batch_size, False, num_workers=config.num_workers, pin_memory=True, drop_last=True, sampler=test_sampler)

    hess_data = []
    if config.dataset == 'imagenet':
        hessianloader = get_hessianloader(config.dataset, 64)
        num_batch = config.hessian_batch_size // 64
        i = 0
        for data, label in hessianloader:
            i += 1
            hess_data.append((data, label))
            if i == num_batch:
                break
    else:
        hessianloader = get_hessianloader(config.dataset, config.hessian_batch_size)
        for data, label in hessianloader:
            hess_data = (data, label)

    net, bottleneck_net = init_network(config, logger, device, config.dataset=="imagenet")

    pruner = init_pruner(net, bottleneck_net, config, writer, logger)

    # total FLOPs calculation
    if config.dataset == 'tiny_imagenet':
        total_flops, _ = compute_model_param_flops(pruner.model, 64, cuda=True)
    elif config.dataset == 'imagenet':
        total_flops, _ = compute_model_param_flops(pruner.model, 224, cuda=True)
    else:
        total_flops, _ = compute_model_param_flops(pruner.model, 32, cuda=True)

    # start pruning
    epochs         = str_to_list(config.epoch, ',', int)
    learning_rates = str_to_list(config.learning_rate, ',', float)
    weight_decays  = str_to_list(config.weight_decay, ',', float)
    ratios         = str_to_list(config.ratio, ',', float)

    fisher_type = config.fisher_type  # empirical|true
    fisher_mode = config.fisher_mode  # eigen|full|diagonal
    normalize = config.normalize
    prune_mode = config.prune_mode  # one-pass | iterative
    fix_rotation = config.get('fix_rotation', True)

    assert (len(epochs) == len(learning_rates) and
            len(learning_rates) == len(weight_decays) and
            len(weight_decays) == len(ratios))

    total_parameters = count_parameters(net)
    for it in range(len(epochs)):
        epoch = epochs[it]
        lr = learning_rates[it]
        wd = weight_decays[it]
        ratio = ratios[it]
        logger.info('-'*120)
        logger.info('** [%d], Ratio: %.2f, epoch: %d, lr: %.4f, wd: %.4f' % (it, ratio, epoch, lr, wd))
        logger.info('Reinit: %s, Fisher_mode: %s, fisher_type: %s, normalize: %s, fix_rotation: %s.' %
            (config.re_init, fisher_mode, fisher_type, normalize, fix_rotation))
        pruner.fix_rotation = fix_rotation

        # test pretrained model
        if config.init_test:
            train_loss_pruned, train_acc_pruned, top5_acc = pruner.test_model(trainloader, criterion, device)
            test_loss_pruned, test_acc_pruned, top5_acc = pruner.test_model(testloader, criterion, device)
            logger.info('Pretrain: Accuracy: %.2f%%(train), %.2f%%(test).' % (train_acc_pruned, test_acc_pruned))
            logger.info('          Loss:     %.2f  (train), %.2f  (test).' % (train_loss_pruned, test_loss_pruned))

        # conduct pruning
        if 'hessian' not in config.fisher_mode:
            cfg = pruner.make_pruned_model(trainloader,
                                        criterion=criterion,
                                        device=device,
                                        fisher_type=fisher_type,
                                        prune_ratio=ratio,
                                        normalize=normalize,
                                        re_init=config.re_init)
        else:
            cfg = pruner.make_pruned_model(hess_data,
                                        criterion=criterion,
                                        device=device,
                                        fisher_type=fisher_type,
                                        prune_ratio=ratio,
                                        normalize=normalize,
                                        re_init=config.re_init,
                                        n_v=config.nv)
        print(pruner.model)
        # for tracking the best accuracy
        compression_ratio, unfair_ratio, all_numel, rotation_numel = compute_ratio(pruner.model, total_parameters,
                                                                                   fix_rotation, logger)
        if config.dataset == 'tiny_imagenet':
            remained_flops, rotation_flops = compute_model_param_flops(pruner.model, 64, cuda=True)
            logger.info('  + Remained FLOPs: %.4fG(%.2f%%), Total FLOPs: %.4fG' % (remained_flops / 1e9, 100.*remained_flops/total_flops ,total_flops / 1e9) )
        elif config.dataset == 'imagenet':
            remained_flops, rotation_flops = compute_model_param_flops(pruner.model, 224, cuda=True)
            logger.info('  + Remained FLOPs: %.4fG(%.2f%%), Total FLOPs: %.4fG' % (remained_flops / 1e9, 100.*remained_flops/total_flops ,total_flops / 1e9) )
        else:
            remained_flops, rotation_flops = compute_model_param_flops(pruner.model, 32, cuda=True)
            logger.info('  + Remained FLOPs: %.4fG(%.2f%%), Total FLOPs: %.4fG' % (remained_flops / 1e9, 100.*remained_flops/total_flops ,total_flops / 1e9) )

        logger.info(f"Total Flops: {remained_flops}")

        test_loss_pruned, test_acc_pruned, top5_acc = pruner.test_model(testloader, criterion, device)
        if config.dataset != 'imagenet':
            time_loader = get_hessianloader(config.dataset, 1)
            run_time = pruner.speed_model(time_loader, criterion, device)
            logger.info(f"Total Run Time: {run_time}")

        test_loss_finetuned, test_acc_finetuned = pruner.fine_tune_model(trainloader=trainloader,
                                                                         testloader=testloader,
                                                                         criterion=criterion,
                                                                         optim=optim,
                                                                         learning_rate=lr,
                                                                         weight_decay=wd,
                                                                         nepochs=epoch)
        train_loss_finetuned, train_acc_finetuned, top5_acc = pruner.test_model(trainloader, criterion, device)
        logger.info(f'After {config.dataset, config.network, config.depth}:  Accuracy: %.2f%%(train), %.2f%%.' % (train_acc_finetuned, test_acc_finetuned))
        logger.info('        Loss:     %.2f  (train), %.2f  .' % (train_loss_finetuned, test_loss_finetuned))

        stat = {
            'total_flops': total_flops,
            'rotation_flops': rotation_flops,
            'flops_remained': float(100.*remained_flops / total_flops),
            'it': it,
            'prune_ratio': ratio,
            'cr': compression_ratio,
            'unfair_cr': unfair_ratio,
            'all_params': all_numel,
            'rotation_params': rotation_numel,
            'prune/test_loss': test_loss_pruned,
            'prune/test_acc': test_acc_pruned,
            'finetune/train_loss': train_loss_finetuned,
            'finetune/test_loss': test_loss_finetuned,
            'finetune/train_acc': train_acc_finetuned,
            'finetune/test_acc': test_acc_finetuned
        }

        print('saving checkpoint')
        save_model(config, it, pruner, cfg, stat)

        stats[it] = stat

        if prune_mode == 'one_pass':
            print('one_pass')
            del net
            del pruner
            net, bottleneck_net = init_network(config, logger, device, config.dataset=="imagenet")
            pruner = init_pruner(net, bottleneck_net, config, writer, logger)
            pruner.iter = it
        with open(os.path.join(config.saving_log, f'stats_{running_time}.json'), 'w') as f:
            json.dump(stats, f)
        if prune_mode != 'one_pass':
            with open(os.path.join(config.saving_log, f'stats{it}.json'), 'w') as f:
                json.dump(stats, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/imagenet_exps/hessian_trace.json', required=False)
    parser.add_argument('--exp-name', type=str, default='', required=False)
    parser.add_argument('--network' , type=str, default='resnet_bottle', required=False)
    parser.add_argument('--depth', type=int, default=50, required=False)
    parser.add_argument('--dataset', type=str, default='imagenet', required=False)
    parser.add_argument('--batch-size', type=int, default=64, required=False)
    parser.add_argument('--epoch', type=str, default='120', required=False)
    parser.add_argument('--learning-rate', type=str, default='1e-3', required=False)
    parser.add_argument('--weight-decay', type=str, default='1e-4', required=False)
    parser.add_argument('--ratio', type=str, default="0.5", required=False)
    parser.add_argument('--ni-ratio', type=float, default=0.1, required=False)

    parser.add_argument('--fisher-mode', type=str, default='hessian', required=False)
    parser.add_argument('--fisher-type', type=str, default='true', required=False)
    parser.add_argument('--fix_layers', type=int, default=0, required=False)
    parser.add_argument('--pruner', type=str, default='hessian_pruner', required=False)
    parser.add_argument('--prune-mode', type=str, default='one_pass', required=False)
    parser.add_argument('--prune-ratio-limit', type=float, default=0.95, required=False)
    parser.add_argument('--fix-rotation', type=bool, default=False, required=False)
    parser.add_argument('--re-init', type=bool, default=False, required=False)
    parser.add_argument('--use-patch', type=bool, default=False, required=False)
    
    parser.add_argument('--hessian-batch-size', type=int, default=512)
    parser.add_argument('--hessian-mode', type=str, default='trace', required=False)
    parser.add_argument('--use-decompose', type=int, default=0)
    parser.add_argument('--nv', type=int, default=3)
    parser.add_argument('--local_rank', type=int, default=0, required=False)
    parser.add_argument('--num_workers', type=int, default=32, required=False)
    parser.add_argument("--data_distributed",type=int, default=0)
    parser.add_argument("--gpu",type=str, default="0,1")
    parser.add_argument("--init-test",type=int, default=0)

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    print('Using config!')
    config = process_config(args.config)

    if args.dataset == "imagenet":
        args.network = "resnet_bottle"

    config.exp_name             = args.exp_name
    config.network              = args.network
    config.depth                = args.depth
    config.dataset              = args.dataset
    config.batch_size           = args.batch_size
    config.epoch                = args.epoch
    config.learning_rate        = args.learning_rate
    config.weight_decay         = args.weight_decay
    config.ratio                = args.ratio
    config.fisher_mode          = args.fisher_mode
    config.fisher_type          = args.fisher_type
    config.fix_layers           = args.fix_layers

    config.load_checkpoint      = "../../HAPresults/checkpoint/pretrain/"
    config.load_checkpoint      += f"{args.dataset}_{args.network}{args.depth}_best.t7"
    config.checkpoint           =  f"../HAPresults/{args.dataset}_result/{args.network}{args.depth}/"
    config.checkpoint           += f"pr_{args.ratio}_nir_{args.ni_ratio}/"

    config.pruner               = args.pruner
    config.prune_mode           = args.prune_mode
    config.prune_ratio_limit    = args.prune_ratio_limit
    config.fix_rotation         = args.fix_rotation
    config.re_init              = args.re_init
    config.use_patch            = args.use_patch

    config.saving_log           = config.checkpoint
    config.hessian_batch_size   = args.hessian_batch_size
    config.hessian_mode         = args.hessian_mode
    config.nv                   = args.nv
    config.num_workers          = args.num_workers
    config.data_distributed     = args.data_distributed
    config.local_rank           = args.local_rank
    config.use_decompose        = args.use_decompose
    config.ni_ratio             = args.ni_ratio
    config.init_test            = args.init_test
    if args.data_distributed:
        torch.cuda.set_device(args.local_rank)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    main(config)
