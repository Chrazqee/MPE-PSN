#python train_vgg.py --device "cuda:0" -opt "SGD" --lr 0.05 -b 32 --epochs 230 -method "MPE-PSN" -TET -T 4 --lambda_mem 0.01 --mixup
#python train_vgg.py --device "cuda:0" -opt "SGD" --lr 0.05 -b 32 --epochs 230 -method "MPE-PSN" -TET -T 4 --lambda_mem 0.01
python train_vgg.py --device "cuda:0" -opt "SGD" --lr 0.05 -b 32 --epochs 230 -method "MPE-PSN" -TET -T 8 --lambda_mem 0.01 --mixup
python train_vgg.py --device "cuda:0" -opt "SGD" --lr 0.05 -b 32 --epochs 230 -method "MPE-PSN" -TET -T 8 --lambda_mem 0.01
