import torch
import torch.nn as nn
import random
import os
import numpy as np
import logging


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_logger(filename, verbosity=1, name=None):
    # 创建文件夹存放日志信息
    log_dir = os.path.join("log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    # 定义日志消息格式
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    # 创建logger，name指定名称
    logger = logging.getLogger(name)
    # 设置Log等级
    logger.setLevel(level_dict[verbosity])
    # 创建一个文件处理器对象，用于将日志消息写入到指定的文件中。
    filename = os.path.join(log_dir, filename)
    fh = logging.FileHandler(filename, "w")
    # 为文件处理器设置格式化器。
    fh.setFormatter(formatter)
    # 将日志添加到日志记录器中。通过 logger 对象记录的日志消息就会被发送到指定的文件中。
    logger.addHandler(fh)
    #将日志输出到控制台
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def TET_loss(outputs, labels, criterion, means, lamb):
    r"""
        Forked from https://github.com/Gus-Lab/temporal_efficient_training
        lambda = 0.05 for CIFAR
        lambda = 0.001 for imagenet
        """
    T = outputs.size(1)
    Loss_es = 0
    for t in range(T):
        Loss_es += criterion(outputs[:, t, ...], labels)
    Loss_es = Loss_es / T  # L_TET
    if lamb != 0:
        MMDLoss = torch.nn.MSELoss()
        y = torch.zeros_like(outputs).fill_(means)
        Loss_mmd = MMDLoss(outputs, y)  # L_mse
    else:
        Loss_mmd = 0
    return (1 - lamb) * Loss_es + lamb * Loss_mmd  # L_Total


