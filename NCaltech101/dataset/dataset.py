import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.utils.data
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from data.augmentation.ncaltech101_augmentation import FixedResolutionPad
from data.utils.representations import StackedHistogram
from data.visualization.bbox_viz import draw_bbox_on_img
from data.visualization.event_to_img_viz import ev_repr_to_img

__CLS_SKIP__ = "BACKGROUND_Google"  # 没有 annotation，跳过

__LABEL_TO_NUM__ = {
    'scorpion': 0,
    'crocodile': 1,
    'strawberry': 2,
    'ketch': 3,
    'ewer': 4,
    'hedgehog': 5,
    'flamingo_head': 6,
    'stegosaurus': 7,
    'Leopards': 8,
    'gerenuk': 9,
    'gramophone': 10,
    'dolphin': 11,
    'airplanes': 12,
    'starfish': 13,
    'ferry': 14,
    'pizza': 15,
    'elephant': 16,
    'minaret': 17,
    'crab': 18,
    'electric_guitar': 19,
    'crocodile_head': 20,
    'snoopy': 21,
    'dalmatian': 22,
    'lotus': 23,
    'ibis': 24,
    'hawksbill': 25,
    'garfield': 26,
    'umbrella': 27,
    'Motorbikes': 28,
    'mandolin': 29,
    'camera': 30,
    'buddha': 31,
    'platypus': 32,
    'beaver': 33,
    'lobster': 34,
    'wrench': 35,
    'dollar_bill': 36,
    'revolver': 37,
    'yin_yang': 38,
    'octopus': 39,
    'accordion': 40,
    'bonsai': 41,
    'schooner': 42,
    'kangaroo': 43,
    'grand_piano': 44,
    'inline_skate': 45,
    'cougar_face': 46,
    'scissors': 47,
    'emu': 48,
    'llama': 49,
    'menorah': 50,
    'tick': 51,
    'euphonium': 52,
    'anchor': 53,
    'sunflower': 54,
    'pagoda': 55,
    'binocular': 56,
    'flamingo': 57,
    'cougar_body': 58,
    'saxophone': 59,
    'butterfly': 60,
    'bass': 61,
    'cellphone': 62,
    'windsor_chair': 63,
    'brain': 64,
    'sea_horse': 65,
    'rhino': 66,
    'crayfish': 67,
    'dragonfly': 68,
    'ceiling_fan': 69,
    'wheelchair': 70,
    'okapi': 71,
    'car_side': 72,
    'pyramid': 73,
    'barrel': 74,
    'nautilus': 75,
    'stapler': 76,
    'headphone': 77,
    'watch': 78,
    'metronome': 79,
    'rooster': 80,
    'chandelier': 81,
    'laptop': 82,
    'cannon': 83,
    'ant': 84,
    'lamp': 85,
    'trilobite': 86,
    'helicopter': 87,
    'pigeon': 88,
    'Faces_easy': 89,
    'wild_cat': 90,
    'brontosaurus': 91,
    'water_lilly': 92,
    'cup': 93,
    'panda': 94,
    'joshua_tree': 95,
    'mayfly': 96,
    'soccer_ball': 97,
    'stop_sign': 98,
    'chair': 99
}
__NUM_TO_LABEL__ = {
    0: 'scorpion',
    1: 'crocodile',
    2: 'strawberry',
    3: 'ketch',
    4: 'ewer',
    5: 'hedgehog',
    6: 'flamingo_head',
    7: 'stegosaurus',
    8: 'Leopards',
    9: 'gerenuk',
    10: 'gramophone',
    11: 'dolphin',
    12: 'airplanes',
    13: 'starfish',
    14: 'ferry',
    15: 'pizza',
    16: 'elephant',
    17: 'minaret',
    18: 'crab',
    19: 'electric_guitar',
    20: 'crocodile_head',
    21: 'snoopy',
    22: 'dalmatian',
    23: 'lotus',
    24: 'ibis',
    25: 'hawksbill',
    26: 'garfield',
    27: 'umbrella',
    28: 'Motorbikes',
    29: 'mandolin',
    30: 'camera',
    31: 'buddha',
    32: 'platypus',
    33: 'beaver',
    34: 'lobster',
    35: 'wrench',
    36: 'dollar_bill',
    37: 'revolver',
    38: 'yin_yang',
    39: 'octopus',
    40: 'accordion',
    41: 'bonsai',
    42: 'schooner',
    43: 'kangaroo',
    44: 'grand_piano',
    45: 'inline_skate',
    46: 'cougar_face',
    47: 'scissors',
    48: 'emu',
    49: 'llama',
    50: 'menorah',
    51: 'tick',
    52: 'euphonium',
    53: 'anchor',
    54: 'sunflower',
    55: 'pagoda',
    56: 'binocular',
    57: 'flamingo',
    58: 'cougar_body',
    59: 'saxophone',
    60: 'butterfly',
    61: 'bass',
    62: 'cellphone',
    63: 'windsor_chair',
    64: 'brain',
    65: 'sea_horse',
    66: 'rhino',
    67: 'crayfish',
    68: 'dragonfly',
    69: 'ceiling_fan',
    70: 'wheelchair',
    71: 'okapi',
    72: 'car_side',
    73: 'pyramid',
    74: 'barrel',
    75: 'nautilus',
    76: 'stapler',
    77: 'headphone',
    78: 'watch',
    79: 'metronome',
    80: 'rooster',
    81: 'chandelier',
    82: 'laptop',
    83: 'cannon',
    84: 'ant',
    85: 'lamp',
    86: 'trilobite',
    87: 'helicopter',
    88: 'pigeon',
    89: 'Faces_easy',
    90: 'wild_cat',
    91: 'brontosaurus',
    92: 'water_lilly',
    93: 'cup',
    94: 'panda',
    95: 'joshua_tree',
    96: 'mayfly',
    97: 'soccer_ball',
    98: 'stop_sign',
    99: 'chair',
}


class NCaltech101Base(Dataset):
    def __init__(self, bins):
        super().__init__()
        self.test_ratio = 0.2
        self.resize_scale = 224  # 准备按照比例进行放缩变换
        self.hist = StackedHistogram(bins, self.resize_scale, self.resize_scale, self.resize_scale)

    def __getitem__(self, item):
        ...

    def __len__(self):
        ...

    def transform_bbox_by_resize_scale(self, x_max, y_max, target):
        (x1, y1), (x2, y2), (_, _), (x4, y4), (_, _) = target
        x, y = x1, y1
        w, h = x2 - x1, y4 - y1
        # 计算放缩比例
        scale_width = self.resize_scale / x_max
        scale_height = self.resize_scale / y_max

        scale = min(scale_height, scale_width)

        # x = torch.clamp(x * scale_width, 1, self.resize_scale)
        # y = torch.clamp(y * scale_height, 1, self.resize_scale)
        #
        # w = torch.clamp(w * scale_width, 1, self.resize_scale - x)
        # h = torch.clamp(h * scale_height, 1, self.resize_scale - y)

        x = torch.clamp(x * scale, 3, self.resize_scale)
        y = torch.clamp(y * scale, 3, self.resize_scale)
        w = torch.clamp(w * scale, 1, self.resize_scale - x - 3)
        h = torch.clamp(h * scale, 1, self.resize_scale - y - 3)

        return torch.tensor([x.int(), y.int(), w.int(), h.int()])

    def transform_event_data_by_resize_scale(self, x_max: th.Tensor, y_max: th.Tensor, x: th.Tensor, y: th.Tensor):
        # 计算放缩比例
        scale_width = self.resize_scale / x_max
        scale_height = self.resize_scale / y_max

        scale = min(scale_height, scale_width)

        # x = x * scale_width
        # y = y * scale_height

        x = x * scale
        y = y * scale

        return x.int(), y.int()

    def transform_image_data_by_resize_scale(self, _img: th.Tensor):
        """
        Args:
            _img: [H, W, C]

        Returns: [self.resize_scale, self.resize_scale, C]
        """
        _img = _img.permute(2, 0, 1)
        pad = FixedResolutionPad(self.resize_scale)
        _img, _ = pad(_img, None)
        _img = _img.permute(1, 2, 0)
        return _img


class NCaltech101(NCaltech101Base):
    def __init__(self, bins: int = 3, data_path: str = '', data_type='training', split_ratio=None,
                 transform=False):
        super().__init__(bins)
        """
        读取 event 数据，读取 label，target 数据，同时读取 image 数据
        """
        if split_ratio is None:
            split_ratio = [0.9, 0.1]
        self.train_ratio = split_ratio[0]
        self.val_ratio = split_ratio[1]
        self.file_path_dvs_img = os.path.join(data_path)
        self.file_path_dvs = os.path.join(data_path, "Caltech101_events")
        self.file_path_img = os.path.join(data_path, "Caltech101_images")
        self.file_path_annotation = os.path.join(data_path, "Caltech101_annotations")
        self.cls_list = os.listdir(
            self.file_path_dvs)  # 100 个类别  不可以 列出 self.file_path_dvs_img, 因为多了一项 BACKGROUND_Google
        self.cls_list.sort()

        self.dvs_filelist = []
        self.img_filelist = []
        self.label = self.cls_list  # label 就是 100 个类别
        self.targets = []
        self.resize = transforms.Resize(size=(self.resize_scale, self.resize_scale),
                                        interpolation=torchvision.transforms.InterpolationMode.NEAREST)

        for i, cls in enumerate(self.cls_list):
            if cls == __CLS_SKIP__:
                continue
            dvs_file_list = os.listdir(os.path.join(self.file_path_dvs, cls))

            num_files = len(dvs_file_list)
            cut_train = int(num_files * self.train_ratio)
            cut_val = int(num_files * self.val_ratio)

            train_file_list_dvs = dvs_file_list[:cut_train]
            val_file_list_dvs = dvs_file_list[cut_train:cut_train + cut_val]

            train_file_list_img = [(file.split('.'))[0] + ".jpg" for file in dvs_file_list[:cut_train]]
            val_file_list_img = [(file.split('.'))[0] + ".jpg" for file in dvs_file_list[cut_train:cut_train + cut_val]]

            train_file_list_annotation = ["annotation_" + file[6: 10] + "_box.npy" for file in
                                          dvs_file_list[:cut_train]]
            val_file_list_annotation = ["annotation_" + file[6: 10] + "_box.npy" for file in
                                        dvs_file_list[cut_train:cut_train + cut_val]]

            if data_type == 'training':
                for file_dvs, file_img, file_annotation in zip(train_file_list_dvs, train_file_list_img,
                                                               train_file_list_annotation):
                    self.dvs_filelist.append(os.path.join(self.file_path_dvs, cls, file_dvs))
                    self.img_filelist.append(os.path.join(self.file_path_img, cls, file_img))
                    self.targets.append(os.path.join(self.file_path_annotation, cls, file_annotation))
            elif data_type == 'validation':
                for file_dvs, file_img, file_annotation in zip(val_file_list_dvs, val_file_list_img,
                                                               val_file_list_annotation):
                    self.dvs_filelist.append(os.path.join(self.file_path_dvs, cls, file_dvs))
                    self.img_filelist.append(os.path.join(self.file_path_img, cls, file_img))
                    self.targets.append(os.path.join(self.file_path_annotation, cls, file_annotation))
            else:
                raise ValueError('Invalid data type')

        self.data_num = len(self.dvs_filelist)
        self.data_type = data_type
        # if data_type != 'train':
        #     counts = np.unique(np.array(self.targets), return_counts=True)[1]
        #     class_weights = counts.sum() / (counts * len(counts))
        #     self.class_weights = torch.Tensor(class_weights)
        self.classes = range(100)
        self.bins = bins
        self.transform = transform  # True or False
        self.shuffled_idx = self.sampler(self.data_num)

    def __getitem__(self, index):
        """
        return:
            for viz (origin scale):
            - data_origin: [torch.Tensor/np.ndarray]  content: (x, y, t, p)  shape: (4, N)
            - image: [torch.Tensor]  content/shape: (H, W, C)
            for train (after rescale):
            - data: [torch.Tensor]  content/shape: (C, T, H, W)  shape: (2, T, 224, 224); generate by HIST
            - target: [dict]  content: {"boxes": (x, y, w, h), }
            - label: [torch.Tensor]  content: (0, 1, ..., 99)
        """
        shuffled_idx = self.shuffled_idx[index]
        file_pth_dvs = self.dvs_filelist[shuffled_idx]
        file_path_img = self.img_filelist[shuffled_idx]
        bbox = self.targets[shuffled_idx]
        label_str = bbox.split('/')[-2]
        label_num = __LABEL_TO_NUM__[label_str]
        data_origin = np.load(file_pth_dvs)
        data = torch.from_numpy(data_origin.copy())
        image = torch.from_numpy(cv2.imread(file_path_img))

        image = self.transform_image_data_by_resize_scale(image)

        x, y, t, p = data.t()
        p //= 2
        x_max, y_max = max(x), max(y)
        bbox = torch.from_numpy(np.load(bbox))

        bbox = self.transform_bbox_by_resize_scale(x_max, y_max, bbox)
        x, y = self.transform_event_data_by_resize_scale(x_max, y_max, x, y)

        data = self.generate_hist(x, y, t, p)  # 此时的 data 的 shape = (2, bins, self.resize_scale, self.resize_scale)

        target = {"bbox": bbox, "label": label_num}

        return data_origin, data, image, target

    def __len__(self):
        return self.data_num

    def __iter__(self):
        for _idx in range(self.data_num):
            # 生成器，调用 __getitem__ 方法
            yield self.__getitem__(_idx)

    def generate_hist(self, x, y, t, p):
        img_ = self.hist.construct(x, y, p, t)
        return img_

    @staticmethod
    def sampler(data_num):
        # dvs img origin 需要同步修改，应该打乱索引，每次拿一个索引，根据索引取数据
        idx_list = list(range(data_num))
        random.shuffle(idx_list)
        return idx_list


def build_ncaltech(bins, data_path, data_type, split_ratio, transform=False):
    if data_type == "training":
        dataset = NCaltech101(bins, data_path, data_type, split_ratio, transform=transform)
    elif data_type == "validation":
        dataset = NCaltech101(bins, data_path, data_type, split_ratio, transform=False)
    else:
        raise NotImplementedError
    return dataset


if __name__ == "__main__":
    _data_path = "/home/chrazqee/datasets/NCaltech101_dvs_img"
    ncaltech101 = NCaltech101(data_path=_data_path, data_type='training')
    data_nums = ncaltech101.data_num
    for idx in range(data_nums):
        idx = ncaltech101.shuffled_idx[idx]
        file_path = ncaltech101.dvs_filelist[idx]
        file_name = file_path.split('/')[-1]
        data_origin_, data_, image_, target_all = ncaltech101[idx]
        label_ = __NUM_TO_LABEL__[target_all["label"]]
        print(file_path)
        print(data_.shape, image_.shape)
        target_ = target_all["bbox"]
        print(target_, label_)
        print('--------------------------------------------------------')
        # 将 得到的 data 可视化出来
        if file_name == "image_0011.npy":
            plt.title(label_)
            img = ev_repr_to_img(data_.detach().reshape(-1, 224, 224))
            img = draw_bbox_on_img(img, [target_[0]], [target_[1]], [target_[2]], [target_[3]], [0])
            plt.subplot(1, 2, 1)
            plt.imshow(img)

            image_ = draw_bbox_on_img(image_.numpy().copy(), [target_[0]], [target_[1]], [target_[2]], [target_[3]], [0])
            plt.subplot(1, 2, 2)
            plt.imshow(image_)
            plt.show()

    #  需求: 将 离散的 事件 hist 起来，然后让其经过一个 LIFNode 激活，再将其可视化出来，对比前后可视化的结果
    # _data_path = "/home/chrazqee/datasets/NCaltech101_dvs_img"
    # ncaltech101 = NCaltech101(data_path=_data_path, data_type="training")
    # # 新建一个 LIFNode
    # from spikingjelly.activation_based.neuron import LIFNode, ParametricLIFNode
    #
    # conv_node = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False)
    # bn_node = nn.BatchNorm2d(num_features=6)
    # tdbn_node = tdBatchNormBTDimFuse(channel=6)
    #
    # # 展开绘制 hist 直方图
    # plif_node = ParametricLIFNode(init_tau=0.5, detach_reset=True, backend='torch', step_mode='m', store_v_seq=False, v_reset=0.5)
    # lif_node = LIFNode(tau=2.0, detach_reset=True, backend='torch', step_mode='m', store_v_seq=False, v_reset=0.5)
    # data_nums = ncaltech101.data_num
    # for idx in range(data_nums):
    #     file_path = ncaltech101.dvs_filelist[idx]
    #     file_name = file_path.split('/')[-1]
    #     data_origin_, data_, image_, target_all = ncaltech101[idx]  # 此时的 data_ 的 shape = (2, bins, self.resize_scale, self.resize_scale)
    #     label_ = __NUM_TO_LABEL__[target_all["label"]]
    #     # 将 data_ 的 T 维度换到首位
    #     data_ = data_.permute(1, 0, 2, 3)
    #     # 将 data_ 用 LIFNode 激活
    #     data_lif = data_  # lif_node(data_)
    #     data_plif = data_
    #     data_conv = conv_node(data_lif.reshape(1, -1, 224, 224).float())
    #     data_tdbn = bn_node(data_conv).reshape(-1, 2, 224, 224)
    #     # data_tdbn = data_conv
    #
    #     vis_data = data_conv.reshape(-1)
    #     vis_data = vis_data.detach().numpy()
    #     print(vis_data.shape)
    #     plt.xlim(-3, 3)
    #     plt.hist(vis_data, bins=1000, density=False, alpha=1, color=None, rwidth=1, edgecolor="white")
    #     plt.show()

        # data_bn = bn_node(data_conv).reshape(-1, 2, 224, 224)
        # data_sum_bn = data_bn + data_tdbn
        # data_after_lif_activated = lif_node(data_sum_bn//2)
        # # 将 data_ 和 data_after_activated 可视化出来
        # img_data_ = ev_repr_to_img(data_.detach().reshape(-1, 224, 224))
        # img_data_after_activated = ev_repr_to_img(data_after_lif_activated.detach().reshape(-1, 224, 224))
        # plt.subplot(3, 1, 1)
        # plt.imshow(img_data_)
        # plt.subplot(3, 1, 2)
        # plt.imshow(img_data_after_activated)
        # plt.subplot(3, 1, 3)
        # plt.imshow(image_.numpy().copy())
        # plt.show()
