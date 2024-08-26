import argparse
import math
import os
import time
import uuid
from signal import sigwait

import loguru
import torch
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch import nn
from torch import amp
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import transforms

try:
    from DVSGusture import myTransform
    from DVSGusture.models import DVSGestureNet
    from cifar10dvs.mixup import Mixup
    from functions import TET_loss, seed_all, get_logger
except ImportError:
    import myTransform
    from models import DVSGestureNet
    from mixup import Mixup
    from functions import TET_loss, seed_all, get_logger

def build_dvs_gusture():
    transform_train = transforms.Compose([
        myTransform.ToTensor(),
        # transforms.Resize(size=(48, 48)),
        transforms.RandomCrop(128, padding=4),
        transforms.RandomHorizontalFlip(), ])
    # transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]],std=[n / 255. for n in [68.2, 65.4, 70.4]]),
    # Cutout(n_holes=1, length=16)])

    transform_test = transforms.Compose([
        myTransform.ToTensor(),
        # transforms.Resize(size=(48, 48))
    ])
    # transforms.Normalize(mean=[n / 255. for n in [129.3, 124.1, 112.4]], std=[n / 255. for n in [68.2, 65.4, 70.4]])
    train_set = DVS128Gesture(root='/home/chrazqee/datasets/DVSGesture/', train=True, data_type='frame', frames_number=args.T,
                           split_by='number', transform=transform_train)
    test_set = DVS128Gesture(root='/home/chrazqee/datasets/DVSGesture/', train=False, data_type='frame', frames_number=args.T,
                          split_by='number', transform=transform_test)

    return train_set, test_set


def train(model, device, train_loader, criterion, optimizer, epoch, scaler, args):
    running_loss = 0
    start_time = time.time()
    model.train()
    M = len(train_loader)
    total = 0
    correct = 0
    dists = 0
    mixup_args = dict(
            mixup_alpha=0.5, cutmix_alpha=0., cutmix_minmax=None,
            prob=0.5, switch_prob=0.5, mode="batch",
            label_smoothing=0.1, num_classes=11)
    mixup_fn = Mixup(**mixup_args)
    if epoch > 140:
        mixup_fn.mixup_enabled = False

    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        labels = labels.to(device)
        images = images.float().to(device)

        images, labels = mixup_fn(images, labels)
        labels = labels.argmax(dim=-1)
        if scaler is not None:
            with amp.autocast("cuda"):
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

                    loss = (1 - args.lambda_mem) * loss + args.lambda_mem * mem_loss
        else:
            outputs = model(images)
            mean_out = outputs.mean(1)  # 时间步维度求均值

            if args.TET:
                loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
            else:
                loss = criterion(mean_out, labels)

            if args.method == "MPE_PSN" and epoch < 75:
                mem_loss = torch.tensor(0., device=device)
                for m in model.modules():
                    if hasattr(m, "mem_loss"):
                        mem_loss += m.mem_loss
                assert mem_loss.item() != 0., "mem_loss should not be Zero!!!"

                loss = (1 - args.lambda_mem) * loss + args.lambda_mem * mem_loss
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
        if scaler is not None:
            scaler.scale(loss.mean()).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.mean().backward()
            optimizer.step()
        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())

        for m in model.modules():
            if hasattr(m, "dist"):
                dists += m.dist

    return running_loss, 100 * correct / total, dists


@torch.no_grad()
def test(model, test_loader, criterion, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.float().to(device)
        outputs = model(inputs)
        labels = labels.to(device)
        mean_out = outputs.mean(1)  # 时间步维度
        _, predicted = mean_out.cpu().max(1)
        total += float(labels.cpu().size(0))
        correct += float(predicted.eq(labels.cpu()).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-j',
                        '--workers',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 10)')
    parser.add_argument('--epochs',
                        default=230,
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
                        default=8,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr',
                        '--learning_rate',
                        default=0.01,
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
    # parser.add_argument('-out_dir', default='./logs_frame/', type=str, help='root dir for saving logs and checkpoint')
    parser.add_argument('--resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-method', type=str, default='MPE_PSN', help='use which network')
    parser.add_argument('-opt', type=str, default='', help='optimizer method')
    parser.add_argument('-tau', type=float, default=0.25, help='tau of LIF')
    parser.add_argument('-TET', action='store_true', help='use the tet loss')
    parser.add_argument('-fixed', action='store_true')
    parser.add_argument("--lambda_mem", default=0.001, type=float, help="")
    parser.add_argument("--parallel", action="store_true", help="")
    parser.add_argument("--channels", default=128, type=int, help="")
    parser.add_argument("--amp", action="store_true", help="")
    parser.add_argument("--use_tdbn", action="store_true", help="")

    args = parser.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = build_dvs_gusture()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    loguru.logger.info(len(train_loader))
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=True)
    loguru.logger.info(len(test_loader))

    model = DVSGestureNet(channels=args.channels, T=args.T, use_tdbn=args.use_tdbn)

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

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    log_file_name = f'T{args.T}_opt_{args.opt}_tau_{args.tau}_method_{args.method}_lr{args.lr}_b_{args.batch_size}_channels{args.channels}'
    if args.TET:
        log_file_name += '_TET'

    num_gpus = torch.cuda.device_count()
    log_file_name += f'_{num_gpus}gpu_frame'

    start_epoch = 0
    best_acc = 0
    best_epoch = 0
    if args.use_tdbn:
        out_dir = './logs_frame_tdbn/'
    else:
        out_dir = './logs_frame_dist/'
    out_dir = os.path.join(out_dir, log_file_name)

    if not args.resume:
        uid = uuid.uuid4().hex
        out_dir = os.path.join(out_dir, str(uid))
    else:
        out_dir = os.path.join(out_dir, args.resume.split('/')[-2])

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

        loss, acc, dist = train(parallel_model, device, train_loader, criterion, optimizer, epoch, scaler, args)

        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        # for name, param in model.named_parameters():
        #     writer.add_histogram(name + '_grad', param.grad, epoch)
        #     writer.add_histogram(name + '_data', param, epoch)
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        writer.add_scalar("dist", dist, epoch)
        scheduler.step()
        facc = test(parallel_model, test_loader, criterion, device)
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
