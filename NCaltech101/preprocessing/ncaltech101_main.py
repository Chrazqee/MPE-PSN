import os

import numpy as np
from numpy import ndarray
from tqdm import tqdm

"""
./NCaltech101_dvs
    ├── Caltech101
    │        ├── accordion
    │        │    ├── image_0001.bin
    │        │    ├── image_0002.bin
    │        │    ├── image_0003.bin
    │        │    ├── image_0004.bin
    │        │    │   ...
    │        │    ├── image_0051.bin
    │        │    ├── image_0052.bin
    │        │    ├── image_0053.bin
    │        │    ├── image_0054.bin
    │        │    └── image_0055.bin
    │        ├── airplanes
    │        ├── anchor
    │        ├── ant
    │        ├── BACKGROUND_Google
    │            ...
    │        ├── wheelchair
    │        ├── wild_cat
    │        ├── windsor_chair
    │        ├── wrench
    │        └── yin_yang
    └── Caltech101_annotations
        ├── accordion
        ├── airplanes
        ├── anchor
        ├── ant
        ├── barrel
        │   ...
        ├── wheelchair
        ├── wild_cat
        ├── windsor_chair
        ├── wrench
        └── yin_yang
"""
__SKIP__ = "BACKGROUND_Google"  # THERE ARE NO 'BACKGROUND_Google' DIR IN CALTECH101_ANNOTATIONS DIR


def read_N_dataset(data_file_path: str) -> ndarray:
    with open(data_file_path, "rb") as f:
        data = np.fromfile(f, np.uint8)

    x_arr, y_arr, ts_arr, p_arr = [], [], [], []
    for i in range(0, len(data), 5):
        x = data[i] + 1
        y = data[i + 1] + 1
        p = ((data[i + 2] >> 7) & 0x01) + 1  # 右移7位再加1
        ts_h = (data[i + 2] & 0x7f) << 16  # 和 127 与，再右移 16 位
        ts_m = data[i + 3] << 8
        ts_l = data[i + 4]
        ts = ts_h | ts_m | ts_l

        x_arr.append(x)
        y_arr.append(y)
        ts_arr.append(ts)
        p_arr.append(p)
    res = np.array([np.array(x_arr), np.array(y_arr), np.array(ts_arr), np.array(p_arr)])
    res = np.transpose(res)
    return res


def read_annotation(annotation_file_path: str):
    with open(annotation_file_path, "rb") as f:
        # annotation = np.fromfile(f, dtype=np.int16)
        # if is_test: print(annotation, len(annotation))
        # 读取边界框的行数和列数
        rows_box = np.fromfile(f, dtype=np.int16, count=1)
        cols_box = np.fromfile(f, dtype=np.int16, count=1)

        # 读取边界框数据并重塑形状
        box_contour = np.fromfile(f, dtype=np.int16, count=rows_box[0] * cols_box[0])
        box_contour = box_contour.reshape((cols_box[0], rows_box[0]))

        # 读取对象轮廓的行数和列数
        rows_obj = np.fromfile(f, dtype=np.int16, count=1)
        cols_obj = np.fromfile(f, dtype=np.int16, count=1)

        # 读取对象轮廓数据并重塑形状
        obj_contour = np.fromfile(f, dtype=np.int16, count=rows_obj[0] * cols_obj[0])
        obj_contour = obj_contour.reshape((cols_obj[0], rows_obj[0]))

        return box_contour, obj_contour


def main(dataset_path: str, is_test: bool):
    dataset_path = dataset_path
    dataset_data_path = dataset_path + '/' + "Caltech101"
    dataset_annotation_path = dataset_path + '/' + "Caltech101_annotations"
    dataset_classes = os.listdir(dataset_data_path)

    for cls in dataset_classes:
        if cls == __SKIP__: continue
        cls_data_path = dataset_data_path + '/' + cls
        cls_annotation_path = dataset_annotation_path + '/' + cls

        # 获取每个类别下的二进制文件名称
        cls_data_file_name = sorted(os.listdir(cls_data_path))
        cls_annotation_name = sorted(os.listdir(cls_annotation_path))
        # pprint(cls_data_file_name)
        # pprint(cls_annotation_name)
        for data, annotation in tqdm(zip(cls_data_file_name, cls_annotation_name)):
            data_file_path = cls_data_path + '/' + data
            annotation_file_path = cls_annotation_path + '/' + annotation
            if is_test: print(data_file_path)
            if is_test: print(annotation_file_path)

            # 读取 data
            xytsp = read_N_dataset(data_file_path)
            # 将 xytsp 数据写入到文件中
            dataset_path_split = dataset_path.split('/')
            dataset_path_fa = '/'.join(dataset_path_split[:-1])
            save_dataset_data_path = dataset_path_fa + '/' + "NCaltech101_dvs_npy" + '/' + "Caltech101" + '/' + cls + '/' + data.split('.')[0] + '.npy'
            if os.path.exists(save_dataset_data_path): continue
            os.makedirs('/'.join(save_dataset_data_path.split('/')[:-1]), exist_ok=True)
            np.save(save_dataset_data_path, xytsp)

            # 读取 annotation
            # box_contour: [5, 2] -> 5个点，每个点的坐标 x, y = box_contour[i] for i in range(5), box_contour[0] == box_contour[-1]
            # obj_contour: [n, 2] -> n 个点围成的区域就是分割图
            box_contour, obj_contour = read_annotation(annotation_file_path)
            save_dataset_annotation_box_contour_path = dataset_path_fa + '/' + "NCaltech101_dvs_npy" + '/' + "Caltech101_annotations" + '/' + cls + '/' + annotation.split('.')[0] + '_box' + '.npy'
            save_dataset_annotation_obj_contour_path = dataset_path_fa + '/' + "NCaltech101_dvs_npy" + '/' + "Caltech101_annotations" + '/' + cls + '/' + annotation.split('.')[0] + '_seg' + '.npy'
            os.makedirs('/'.join(save_dataset_annotation_box_contour_path.split('/')[:-1]), exist_ok=True)
            os.makedirs('/'.join(save_dataset_annotation_obj_contour_path.split('/')[:-1]), exist_ok=True)
            np.save(save_dataset_annotation_box_contour_path, box_contour)
            np.save(save_dataset_annotation_obj_contour_path, obj_contour)

if __name__ == "__main__":
    main("/home/chrazqee/datasets/NCaltech101_dvs", False)
