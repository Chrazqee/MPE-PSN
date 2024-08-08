import argparse
import math
import os
import time

import loguru
import torch
import torch.nn as nn
import torchvision.transforms as transforms
# from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from sj.spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
from torch.utils.tensorboard import SummaryWriter

import myTransform
from functions import TET_loss, seed_all, get_logger
from models import VGGSNN, VGGPSN, VGGSSN
from mixup import Mixup

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('-j',
                    '--workers',
                    default=8,
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
                    default=100,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('-T',
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
parser.add_argument("--device", default="cuda:0", type=str, help="")
parser.add_argument('-data_path', default='/home/chrazqee/datasets/CIFAR10-DVS/events_pt_1', type=str, help='')
parser.add_argument('-out_dir', default='./logs_frameV6/', type=str, help='root dir for saving logs and checkpoint')
parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
parser.add_argument('-method', type=str, default='VGGSNN', help='use which network')
parser.add_argument('-opt', type=str, default='SGD0.1', help='optimizer method')
parser.add_argument('-tau', type=float, default=0.25, help='tau of LIF')
parser.add_argument('-TET', action='store_true', help='use the tet loss')
parser.add_argument('-fixed', action='store_true')
parser.add_argument("--lambda_mem", default=0.01, type=float, help="")
parser.add_argument("--parallel", action="store_true", help="")
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

def train(model, device, train_loader, criterion, optimizer, epoch, args):
    running_loss = 0
    start_time = time.time()
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    # mixup_args = dict(
    #         mixup_alpha=0.5, cutmix_alpha=0., cutmix_minmax=None,
    #         prob=0.5, switch_prob=0.5, mode="batch",
    #         label_smoothing=0.1, num_classes=10)
    # mixup_fn = Mixup(**mixup_args)
    # mem_loss_fn = nn.MSELoss(reduction="None").to(device)
    # if epoch > 75:
    # mixup_fn.mixup_enabled = False
        
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.float().to(device)

        # images, labels = mixup_fn(images, labels)
        # labels = labels.argmax(dim=-1)
        outputs = model(images)
        mean_out = outputs.mean(1)  # 时间步维度求均值

        if args.TET:
            loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
        else:
            loss = criterion(mean_out, labels)

        if args.method == "SSN" and epoch < 75:
            mem_loss = torch.tensor(0., device=device)
            for m in model.modules():
                if hasattr(m, "mem_loss"):
                    mem_loss += m.mem_loss
            assert mem_loss.item() != 0., "mem_loss should not be Zero!!!"

            loss = (1-args.lambda_mem) * loss + args.lambda_mem * mem_loss

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
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.float().to(device)
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


def build_dvscifar():
    transform_train = transforms.Compose([
        myTransform.ToTensor(),
        transforms.Resize(size=(48, 48)),
        transforms.RandomCrop(48, padding=4),
        transforms.RandomHorizontalFlip(), ])
    # transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],std=[n / 255. for n in [68.2, 65.4, 70.4]]),
    # Cutout(n_holes=1, length=16)])

    transform_test = transforms.Compose([
        myTransform.ToTensor(),
        transforms.Resize(size=(48, 48))])
    # transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]])
    train_set = CIFAR10DVS(root='/home/chrazqee/datasets/CIFAR10-DVS/', train=True, data_type='frame', frames_number=args.T,
                           split_by='number', transform=transform_train)
    test_set = CIFAR10DVS(root='/home/chrazqee/datasets/CIFAR10-DVS/', train=False, data_type='frame', frames_number=args.T,
                          split_by='number', transform=transform_test)

    return train_set, test_set


def build_dvscifar_():
    transform_train = transforms.Compose([
        transforms.Resize(size=(48, 48)),
        transforms.RandomCrop(48, padding=4),
        transforms.RandomHorizontalFlip()])

    transform_test = transforms.Compose([
        transforms.Resize(size=(48, 48))])

    train_dataset, val_dataset = (
        Cifar10dvs(bins=args.T, data_path=args.data_path, data_type="training", split_ratio=[0.9, 0.1], transform=True,
                   transform_compose=transform_train, repre="frame"),
        Cifar10dvs(bins=args.T, data_path=args.data_path, data_type="validation", split_ratio=[0.9, 0.1],
                   transform_compose=transform_test, transform=True, repre="frame"))

    return train_dataset, val_dataset

from data.dataset import Cifar10dvs

if __name__ == '__main__':
    seed_all(args.seed)
    train_dataset, val_dataset = build_dvscifar()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    loguru.logger.info(len(train_loader))
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)
    loguru.logger.info(len(test_loader))

    if args.method == 'PSN':
        model = VGGPSN()
    elif args.method == "SSN":
        model = VGGSSN(tau=args.tau, T=args.T)
    else:
        model = VGGSNN(tau=args.tau)
    # print(model)

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
    elif args.opt == 'SGD0.1':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)
    elif args.opt == 'SGD0.02':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, weight_decay=5e-4, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)

    log_file_name = f'T{args.T}_opt_{args.opt}_tau_{args.tau}_method_{args.method}_lr{args.lr}_b_{args.batch_size}_lambda_mem_{args.lambda_mem}'
    if args.TET:
        log_file_name += '_TET'

    num_gpus = torch.cuda.device_count()
    log_file_name += f'_{num_gpus}gpu_frameV2'

    start_epoch = 0
    best_acc = 0
    best_epoch = 0
    out_dir = os.path.join(args.out_dir, log_file_name)

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

        loss, acc = train(parallel_model, device, train_loader, criterion, optimizer, epoch, args)

        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name + '_grad', param.grad, epoch)
        #     writer.add_histogram(name + '_data', param, epoch)
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        scheduler.step()
        facc = test(parallel_model, test_loader, device)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}'.format(epoch, args.epochs, facc))
        writer.add_scalar('test_acc', facc, epoch)

        save_max = False
        if best_acc < facc:
            best_acc = facc
            save_max = True
            best_epoch = epoch + 1
            # torch.save(parallel_model.module.state_dict(), 'VGGSNN_woAP.pth')
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
