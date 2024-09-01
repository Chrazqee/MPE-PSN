#python train.py --lr 0.01 -TET -b 4 -T 10 --channels 256 --amp  # 96.18
#python train.py --lr 0.01 -TET -b 4 -T 10 --channels 256 --amp --use_tdbn  # 96.18
#python train.py --lr 0.01 -TET -b 6 -T 16 --channels 128 --amp  # 96.88
#python train.py --lr 0.01 -TET -b 6 -T 16 --channels 128 --amp --use_tdbn
#python train.py --lr 0.01 -TET -b 4 -T 16 --channels 256 --amp  # 97.22
#python train.py --lr 0.05 -TET -b 4 -T 16 --channels 256 --amp --resume "/home/chrazqee/Desktop/code/Parallel-Spiking-Neuron-main/DVSGustureV2/logs_frame/T16_opt__tau_0.25_method_MPE_PSN_lr0.05_b_4_channels256_TET_2gpu_frame/7957cb4a7f104882bc258a23a60602db/checkpoint_latest.pth"  # 97.57
# todo: 结果复现，获得曲线
#python train.py --lr 0.01 -TET -b 4 -T 16 --channels 256 --amp --resume "/home/chrazqee/Desktop/code/Parallel-Spiking-Neuron-main/DVSGusture/logs_frame_dist_no_memloss/T16_opt__tau_0.25_method_MPE_PSN_lr0.01_b_4_channels256_TET_no_memLoss/7f24b227d81544a88d275c84afafe8c5/checkpoint_latest.pth"

#python train.py --lr 0.01 -TET -b 4 -T 16 --channels 256 --amp
python train.py --lr 0.05 -TET -b 8 -T 16 --channels 256 --amp --resume "/home/chrazqee/Desktop/code/Parallel-Spiking-Neuron-main/DVSGusture/logs_frame_sigmoid/T16_opt__tau_0.25_method_MPE_PSN_lr0.05_b_8_channels256_TET_sigmoid/eaccd2199acd4394afae8729d5968d25/checkpoint_latest.pth" # 97.92
