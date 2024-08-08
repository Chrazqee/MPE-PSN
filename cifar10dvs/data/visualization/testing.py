import os
import sys

import cv2
import numpy as np
import torch
import torch as th
from matplotlib import pyplot as plt

from data.visualization.bbox_viz import draw_bbox_on_img
from event_viz import draw_events_on_image

cur_path = os.getcwd()
cur_path_fa = cur_path.split("/")[:-1]
cur_path_fa_fa = cur_path_fa[:-1]
cur_path_fa = "/".join(cur_path_fa)
cur_path_fa_fa = "/".join(cur_path_fa_fa)
if cur_path not in sys.path:
    sys.path.append(cur_path)
if cur_path_fa not in sys.path:
    sys.path.append(cur_path_fa)
if cur_path_fa_fa not in sys.path:
    sys.path.append(cur_path_fa_fa)

from data.utils.representations import StackedHistogram
from event_to_img_viz import ev_repr_to_img


class TestEventUpToImageWithBbox:
    @staticmethod
    def testing(img_file_path, event_file_path, bbox_file_path):
        # 将 event 读取出来
        events = np.load(event_file_path)
        events = np.transpose(events)
        x, y, t, p = events

        # 读取图片
        img = cv2.imread(img_file_path)

        # 事件数据有一定程度的缩放，因此将其缩放回原尺寸
        H, W, C = img.shape
        X, Y = max(x), max(y)

        X_rate = W / X  # 宽度的缩放比例
        Y_rate = H / Y  # 高度的缩放比例

        img = draw_events_on_image(img, np.int16(x * X_rate), np.int16(y * Y_rate), p)

        # 读取 bbox
        bbox = np.load(bbox_file_path)
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        _, y3 = bbox[2]
        x4, y4 = bbox[3]
        w, h = np.array([x2 - x1]), np.array([y4 - y1])
        x, y = np.array([x1]), np.array([y1])
        w, h = th.from_numpy(w), th.from_numpy(h)
        x, y = th.from_numpy(x), th.from_numpy(y)

        img = draw_bbox_on_img(img, x * X_rate, y * Y_rate, w * X_rate, h * Y_rate, [1])

        plt.imshow(img)
        plt.show()


class TestEventToImage:
    def __init__(self):
        self.height = 224
        self.width = 224

    def testing(self, event_file_path, bbox_file_path):
        # 1. 读取 event
        events = np.load(event_file_path)
        events = np.transpose(events)
        x, y, t, p = events
        x, y, t, p = th.from_numpy(x), th.from_numpy(y), th.from_numpy(t), th.from_numpy(p) - 1

        # 2. 对 x, y 进行缩放
        X, Y = max(x), max(y)

        X_rate = self.width / X  # 宽度的缩放比例
        Y_rate = self.height / Y  # 高度的缩放比例

        # x, y = x * X_rate, y * Y_rate

        # 实例化 representation，也就是 HIST
        hist = StackedHistogram(3, self.height, self.width, 225, True)
        img = hist.construct(x.int(), y.int(), p, t)
        img = self.merge_channel_and_bins(img)

        # 调用 event_repr_to_img
        img = ev_repr_to_img(img.detach())

        # 读取 bbox
        bbox = np.load(bbox_file_path)
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]
        _, y3 = bbox[2]
        x4, y4 = bbox[3]
        w, h = np.array([x2 - x1]), np.array([y4 - y1])
        x, y = np.array([x1]), np.array([y1])
        w, h = th.from_numpy(w), th.from_numpy(h)
        x, y = th.from_numpy(x), th.from_numpy(y)

        img = draw_bbox_on_img(img, x, y, w, h, [0])

        plt.imshow(img)
        plt.show()

    def merge_channel_and_bins(self, representation: th.Tensor):
        assert representation.dim() == 4
        return th.reshape(representation, (-1, self.height, self.width))


if __name__ == "__main__":
    img_file_path_ = "/home/chrazqee/datasets/NCaltech101_dvs_img/Caltech101_images/Leopards/image_0104.jpg"
    event_file_path_ = "/home/chrazqee/datasets/NCaltech101_dvs_img/Caltech101_events/Leopards/image_0104.npy"
    bbox_file_path_ = "/home/chrazqee/datasets/NCaltech101_dvs_img/Caltech101_annotations/Leopards/annotation_0104_box.npy"
    TestEventUpToImageWithBbox.testing(img_file_path_, event_file_path_, bbox_file_path_)
    test = TestEventToImage()
    test.testing(event_file_path_, bbox_file_path_)
