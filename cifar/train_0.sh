#python train_cifar10.py --model resnet19 --T 4 -b 64 --lr 0.01 --epochs 300 --device cuda:0
#python train_cifar10.py --model resnet18 --T 4 -b 64 --lr 0.01 -opt Adam --epochs 500 --device cuda:0 --mixup -TET -resume "/home/chrazqee/PycharmProjects/cifar10/logs_mixup/model_resnet18_T4_opt_Adam_MPE-PSN_lr0.01_b_64_device_cuda:0_TET_mixup/checkpoint_latest.pth"
#python train_cifar10.py --model resnet18 --T 4 -b 64 --lr 0.02 -opt Adam --epochs 300 --device cuda:0 --mixup -TET # --parallel
python train_cifar10.py --model resnet19 --T 4 -b 64 --lr 0.01 -opt Adam --epochs 300 --device cuda -TET --parallel
