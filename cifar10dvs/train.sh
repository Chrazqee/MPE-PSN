#python train_vgg.py -b 32 --epochs 200 -method SSN -TET -T 4
#python train_vgg.py --device "cuda:1" -opt "SGD" --lr 0.1 -b 48 --epochs 200 -method SSN -TET -T 8 --lambda_mem 0.01
#python train_vgg.py --parallel -opt "SGD" --lr 0.1 -b 32 --epochs 200 -method SSN -TET -T 16 --lambda_mem 0.01 -resume "/home/chrazqee/Desktop/code/Spiking-Neuron/cifar10dvs/logs_frameV5/T16_opt_SGD_tau_0.25_method_SSN_lr0.1_b_32_lambda_mem_0.01_TET_2gpu_frameV2/checkpoint_latest.pth"
python train_vgg.py --device "cuda:1" -opt "SGD" --lr 0.1 -b 48 --epochs 200 -method SSN -TET -T 10 --lambda_mem 0.01 
