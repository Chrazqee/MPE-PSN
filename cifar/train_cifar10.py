import argparse
import math
import os
import time

import loguru
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset_utils import prepare_cifar10
from functions import seed_all, get_logger, TET_loss
from mixup import Mixup
from model.MPE_PSN import MPE_PSN
from model.spiking_resnet import spiking_resnet18, spiking_resnet19

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j',
                    '--workers',
                    default=4,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--epochs',
                    default=100,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch_size',
                    default=32,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr',
                    '--learning_rate',
                    default=0.001,
                    type=float,
                    metavar='LR',
                    help='initial learning rate',
                    dest='lr')
parser.add_argument('--seed',
                    default=42,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('--T',
                    default=10,
                    type=int,
                    metavar='N',
                    help='snn simulation time (default: 2)')
parser.add_argument('--means',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--lamb',
                    default=1e-3,
                    type=float,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument("--device", default="cuda:1", type=str, help="")
parser.add_argument('-out_dir', default='./logs_frame_sigmoid/', type=str, help='root dir for saving logs and checkpoint')
parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
# parser.add_argument('-method', type=str, default='MPE-PSN', help='use which network')
parser.add_argument('-opt', type=str, default='', help='optimizer method')
parser.add_argument('-tau', type=float, default=0.25, help='tau of LIF')
parser.add_argument('-TET', action='store_true', help='use the tet loss')
parser.add_argument("--lambda_mem", default=0.01, type=float, help="")
parser.add_argument("--parallel", action="store_true", help="")
parser.add_argument("--mixup", action="store_true", help="")
parser.add_argument("--num_classes", default=10, type=int, help="")
parser.add_argument("--dataset", default="cifar10", type=str, help="")
parser.add_argument("--model", default="resnet18", type=str, help="")


args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

def train(model, device, train_loader, criterion, optimizer, epoch, args):
    running_loss = 0
    model.train()
    total = 0
    correct = 0

    if args.mixup:
        mixup_args = dict(
                mixup_alpha=0.5, cutmix_alpha=0., cutmix_minmax=None,
                prob=0.5, switch_prob=0.5, mode="batch",
                label_smoothing=0.1, num_classes=10)
        mixup_fn = Mixup(**mixup_args)
        if epoch > 140:
            mixup_fn.mixup_enabled = False

    for i, (images, labels) in tqdm.tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.float().to(device)
        # images = images.unsqueeze_(1).repeat(1, args.T, 1, 1, 1)

        if args.mixup:
            images, labels = mixup_fn(images, labels)
            labels = labels.argmax(dim=-1)
        outputs = model(images)
        mean_out = outputs.mean(1)  # 时间步维度求均值

        if args.TET:
            loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
        else:
            loss = criterion(mean_out, labels)

        # 梯度裁剪
        # 检查梯度中的 -inf
        for n, param in model.named_parameters():
            if hasattr(param, "grad") and param.grad is not None:
                if torch.isinf(param.grad).any():
                    loguru.logger.warning(f"Infinite gradient detected in {n}, gradient clip is applied!")
                    # 处理 -inf，例如将其设置为0
                    param.grad[param.grad == -float('inf')] = 0
                    param.grad[param.grad == float('inf')] = 0
                if torch.isnan(param.grad).any():
                    loguru.logger.warning(f"Nan gradient detected in {n}, gradient clip is applied!")
                    # 处理 -inf，例如将其设置为0
                    param.grad[param.grad != param.grad] = 0

        # 进行 梯度 裁剪
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)

        # mem loss
        if epoch < 150:
            mem_loss = torch.tensor(0., device=device)
            for m in model.modules():
                if hasattr(m, "mem_loss"):
                    mem_loss += m.mem_loss
            # when using parallel, it may always zero!!!
            # assert mem_loss.item() != 0., "mem_loss should not be Zero!!!"
            mem_loss = torch.clamp(mem_loss, 0.)
            loss = (1 - args.lambda_mem) * loss + args.lambda_mem * mem_loss


        running_loss += loss.item()
        if math.isnan(running_loss):
            raise ValueError('loss is Nan')

        loss.mean().backward()
        optimizer.step()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total


@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(test_loader)):
        inputs = inputs.float().to(device)
        # inputs = inputs.unsqueeze_(1).repeat(1, args.T, 1, 1, 1)
        outputs = model(inputs)
        mean_out = outputs.mean(1)  # 时间步维度
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc


if __name__ == '__main__':
    seed_all(args.seed)
    if args.dataset == "cifar10":
        trainloader, testloader = prepare_cifar10(args)
    else:
        raise NotImplementedError
    if args.model == "resnet18":
        model = spiking_resnet18(num_classes=args.num_classes, neuron=MPE_PSN, T=args.T)
    elif args.model == "resnet19":
        model = spiking_resnet19(num_classes=args.num_classes, neuron=MPE_PSN, T=args.T)
    else:
        raise NotImplementedError

    loguru.logger.info("Creating model")
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loguru.logger.info(f"number of params: {n_parameters} = {n_parameters / 1e6:.2f} M")

    if args.parallel:
        parallel_model = torch.nn.DataParallel(model)
    else:
        parallel_model = model
    parallel_model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    if args.opt == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.opt == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    log_file_name = f'model_{args.model}_T{args.T}_opt_{args.opt}_MPE-PSN_lr{args.lr}_b_{args.batch_size}_device_{args.device}'
    if args.TET:
        log_file_name += '_TET'
    if args.mixup:
        log_file_name += "_mixup"
    log_file_name += "_pre-encoding"

    start_epoch = 0
    best_acc = 0
    best_epoch = 0
    if args.mixup:
        out_dir = "./logs_mixup"
    else:
        out_dir = "./logs"
    out_dir += "_pre-encoding/"
    out_dir = os.path.join(out_dir, log_file_name)

    if args.resume:
        print('load resume')
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

    logger = get_logger(log_file_name + '.log')
    logger.info('start training!')

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    for epoch in range(start_epoch, args.epochs):

        loss, acc = train(parallel_model, device, trainloader, criterion, optimizer, epoch, args)

        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        scheduler.step()
        facc = test(parallel_model, testloader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch, args.epochs, facc))
        writer.add_scalar('test_acc', facc, epoch)

        save_max = False
        if best_acc < facc:
            best_acc = facc
            save_max = True
            best_epoch = epoch + 1

        logger.info('Best Test acc={:.3f}'.format(best_acc))
        print('\n')

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
