import argparse

import loguru
import tqdm

import data_loaders
from utils import *
from models.resnet_models import *
from tools.functions import *

from torch.cuda import amp
from mixup import *
from torch.utils.tensorboard import SummaryWriter
from models.MPE_PSN import MPE_PSN

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# FOR TRAINING

parser.add_argument("--epochs", default=500, type=int, help="#epoch")
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--batch_size", default=64, type=int, help="mini-batch size")
parser.add_argument("--model", default='resnet19', type=str, help="arch")
parser.add_argument("--method", default="MPE-PSN", type=str, help="")
parser.add_argument("--lambda_mem", default=0.01, type=float, help="")
parser.add_argument('--weight_decay',default=5e-4,)
parser.add_argument('--T', default=6, type=int, help="")
# Other settings
parser.add_argument('--seed', default=1000, type=int, help='random seed')
parser.add_argument('--workers', default=4, type=int, help='#threads')
parser.add_argument('--scheduler', default='cos', type=str, help='lr scheduler')
# FOR TET
parser.add_argument('--TET',
                    default=True,
                    type=bool,
                    metavar='N',
                    help='if use Temporal Efficient Training (default: True)')
parser.add_argument('--lamb',
                    default=1e-3,
                    type=float,
                    metavar='N',
                    help='adjust the norm factor to avoid outlier (default: 0.0)')
parser.add_argument('--means',
                    default=1.0,
                    type=float,
                    metavar='N',
                    help='make all the potential increment around the means (default: 1.0)')
parser.add_argument('--norm_layer', default='tdbn',type=str)
parser.add_argument('--mixup', default=False, action='store_true')
# todo: use Adam optimizer to avoid degrade when epoch large to around 150
# set --optimizer to '' for using Adam!!!
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer')
parser.add_argument('--amp', default=True, action='store_true',
                        help='Use AMP training')
parser.add_argument("--resume", default="logs_frame_sigmoid/T4_opt_SGD_model_resnet19_method_MPE-PSN_lr0.01_b_64_lambda_mem_0.01_TET_frame_sigmoid/checkpoint_latest.pth  ", type=str, help="")
args = parser.parse_args()


def train(model, device, train_loader, criterion, optimizer, args, scaler):

    model.train()
    running_loss =0
    correct = 0
    total = 0

    for batch_idx, (data, labels) in tqdm.tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        data, labels = data.to(device), labels.to(device)
        if scaler is not None:
            with amp.autocast():
                outputs = model(data)
                mean_out = outputs.mean(1)
                if args.TET:
                    loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
                else:
                    loss = criterion(mean_out, labels)
        else:
            outputs = model(data)
            mean_out = outputs.mean(1)
            if args.TET:
                loss = TET_loss(outputs, labels, criterion, args.means, args.lamb)
            else:
                loss = criterion(mean_out, labels)
        running_loss += loss.item()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += float(labels.size(0))
        _, predicted = mean_out.cpu().max(1)

        if args.mixup:
            correct += float(predicted.eq(labels.argmax(dim=-1).cpu()).sum().item())
        else:
            correct += float(predicted.eq(labels.cpu()).sum().item())
    return running_loss, 100 * correct / total

@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(test_loader)):
        inputs = inputs.to(device)
        outputs = model(inputs)

        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())

        if batch_idx % 10 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100. * float(correct) / float(total)
    return final_acc


if __name__ == '__main__':
    seed_all(args.seed)
    train_dataset, test_dataset = data_loaders.build_cifar(root=r"./datasets/cifar-100-python", use_cifar10=False, cutout=True,
                                                           download=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, pin_memory=True)

    if args.TET:
        print('==> using TET loss')
    if args.model == "resnet19" and args.method == "MPE-PSN":
        net = resnet19(n_input=[3, 32, 32], norm_layer=args.norm_layer, spiking_neuron=MPE_PSN, timestep=args.T, n_output=100)
    elif args.model == 'resnet18':
        net = resnet18()
    else:
        raise NotImplementedError

    loguru.logger.info("Creating model")
    n_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    loguru.logger.info(f"number of params: {n_parameters} = {n_parameters / 1e6:.2f} M")

    net = net.to(device)
    net.T = args.T

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,nesterov=True)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=5e-3, max_lr=args.lr
        )

    if args.amp:
        scaler = amp.GradScaler()
    else:
        scaler = None

    log_file_name = f'T{args.T}_opt_{args.optimizer}_model_{args.model}_method_{args.method}_lr{args.lr}_b_{args.batch_size}_lambda_mem_{args.lambda_mem}'
    if args.TET:
        log_file_name += '_TET'
    if args.mixup:
        log_file_name += "_mixup"
    log_file_name += f'_frame_sigmoid'

    start_epoch = 0
    best_acc = 0
    best_epoch = 0

    if args.mixup:
        out_dir = "./logs_frame_sigmoid_mixup/"
    else:
        out_dir = "./logs_frame_sigmoid/"
    out_dir = os.path.join(out_dir, log_file_name)

    if args.resume:
        print('load resume')
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']

    logger = get_logger(log_file_name + '.log')
    logger.info('start training!')

    writer = SummaryWriter(os.path.join(out_dir, 'logs'), purge_step=start_epoch)

    for epoch in range(start_epoch, args.epochs):
        loss, acc = train(net, device, train_loader, criterion, optimizer, args, scaler)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, args.epochs, loss, acc))
        writer.add_scalar('train_loss', loss, epoch)
        writer.add_scalar('train_acc', acc, epoch)
        scheduler.step()
        facc = test(net, test_loader, device)
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
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
