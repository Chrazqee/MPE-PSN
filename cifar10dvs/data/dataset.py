import os.path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from .augmentation.cifar10dvs_augmentation import SNNAugmentWide
    from .utils.representations import StackedHistogram, IntegratedFixedFrameNumber
except ImportError:
    from augmentation.cifar10dvs_augmentation import SNNAugmentWide
    from utils.representations import StackedHistogram, IntegratedFixedFrameNumber

# import sys
#
# pprint(sys.path)

__LABEL_TO_NUM__ = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}

__NUM_TO_LABEL__ = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}


class Cifar10dvsBase(Dataset):
    def __init__(self, bins, size_scale=128):
        super().__init__()
        self.hist = StackedHistogram(bins, size_scale, size_scale, size_scale)
        self.frame = IntegratedFixedFrameNumber(size_scale, size_scale, bins)

    def __len__(self):
        ...

    def __getitem__(self, index):
        ...


class Cifar10dvs(Cifar10dvsBase):
    def __init__(self, bins: int = 3, data_path: str = '', data_type='training', split_ratio=None,
                 transform=False, transform_compose=None, repre="hist"):
        super().__init__(bins)
        """
        bins: number of bins for histogram representation
        data_path: e.g. /home/<user>/datasets/CIFAR10-DVS/events_np/[airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck/cifar10_xxx_x.pt]
        data_type: 'training', 'validation'
        split_ratio: [0.8, 0.2] for training, validation
        """
        assert bins > 0
        self.bins = bins
        assert isinstance(data_path, str)
        self.data_path = os.path.join(data_path)
        self.data_type = data_type
        if split_ratio is None:
            split_ratio = [0.9, 0.1]  # training, validation

        self.repre = repre

        # self.resize = transforms.Resize(size=(48, 48))  # 48 48
        # self.tensorx = transforms.ToTensor()
        # self.imgx = transforms.ToPILImage()

        self.transform = transform

        cls_files_path = []

        self.train_files_path = []
        self.val_files_path = []

        # 获取类别目录
        self.cls_list = os.listdir(
            self.data_path)  # ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.cls_list.sort()
        for cls in self.cls_list:
            cls_path = os.path.join(self.data_path,
                                    cls)  # /home/<user>/datasets/CIFAR10-DVS/events_np/[airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck]/]
            cls_files = os.listdir(cls_path)
            for file in cls_files:
                cls_file_path = os.path.join(cls_path, file)
                cls_files_path.append(cls_file_path)

            # 划分训练集和验证集；要保证训练集和验证集每次划分都不串一起
            cls_files_path.sort()  # 保证每次划分都不串一起
            self.train_files_path.extend(cls_files_path[:int(len(cls_files_path) * split_ratio[0])])
            self.val_files_path.extend(cls_files_path[int(len(cls_files_path) * split_ratio[0]):])
            cls_files_path.clear()

        # 将 self.train_files_path，self.val_files_path 在全局打乱；写一个 sampler，用 random，固定随机数种子
        self.train_files_path, self.val_files_path = self.sampler(self.train_files_path, self.val_files_path)
        # self.transform_compose = transforms.Compose([
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     SNNAugmentWide()
        # ])
        self.transform_compose = transform_compose

    def __getitem__(self, index):
        if self.data_type == "training":
            data_file_path = self.train_files_path
        elif self.data_type == "validation":
            data_file_path = self.val_files_path
        else:
            raise ValueError("data_type must be 'training' or 'validation'")

        # 获取数据文件路径
        file_path = data_file_path[index]
        # load 数据
        data = torch.load(file_path, weights_only=True)
        # 拿到 x, y, t, p
        x, y, t, p = data['x'], data['y'], data['t'], data['p']

        # 选择 数据 表示的方式
        if self.repre == "hist":
            hist_representation = self.generate_hist(x, y, t, p)
            data = hist_representation.permute(1, 0, 2, 3)  # [T, C, H, W]
        elif self.repre == "frame":
            frame_representation = self.generate_frame(x, y, t, p)
            data = frame_representation  # [frames_num, C, H, W]
        else:
            raise NotImplementedError

        # 根据 file_path 获取类别
        label = file_path.split('/')[-2]
        label = __LABEL_TO_NUM__[label]

        # 数据增强
        if self.transform:
            data = self.transform_compose(data)

        return data, label  # torch.Size([3, 2, 128, 128]) 0

    def __len__(self):
        if self.data_type == "training":
            return len(self.train_files_path)
        elif self.data_type == "validation":
            return len(self.val_files_path)
        else:
            raise ValueError("data_type must be 'training' or 'validation'")

    def generate_hist(self, x, y, t, p):
        img_ = self.hist.construct(x, y, p, t)
        return img_

    def generate_frame(self, x, y, t, p):
        img_ = self.frame.construct(x, y, p, t)
        return img_

    @staticmethod
    def sampler(train_files_path, val_files_path):
        # 打乱
        random.shuffle(train_files_path)
        random.shuffle(val_files_path)
        return train_files_path, val_files_path


def build_cifar10dvs(bins, data_path, data_type, split_ratio, transform=False, repre="hist"):
    assert data_type in ["training", "validation", "testing"]
    if data_type == "testing":
        return Cifar10dvs(bins, data_path, "validation", split_ratio, transform=transform, repre=repre)
    else:
        return Cifar10dvs(bins, data_path, data_type, split_ratio, transform=False, repre=repre)


if __name__ == "__main__":
    from visualization.event_to_img_viz import ev_repr_to_img

    dataset_ = Cifar10dvs(bins=5,
                          data_path='/home/chrazqee/datasets/CIFAR10-DVS/events_pt/',
                          data_type='training', transform=True)
    from torchvision.transforms import Resize, InterpolationMode
    # resize = Resize(512, interpolation=InterpolationMode.NEAREST)
    for data_, label_ in dataset_:
        print(data_.shape, label_)
        # data_ = resize(data_)
        img = ev_repr_to_img(data_.detach().reshape(-1, 256, 256))
        plt.title(__NUM_TO_LABEL__[label_])
        plt.imshow(img)
        plt.show()
