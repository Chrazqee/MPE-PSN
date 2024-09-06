import torch
from torch import nn


class Seq2ANN(nn.Module):
    r"""
    Forked from spikingjelly
    功能：B, T 维度合并，前向过程 module
    """
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())  # 合并 BT
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)  # 还原 B，T，...

class tdBatchNormBTDimFuse(nn.BatchNorm2d):
    def __init__(self, channel):
        super(tdBatchNormBTDimFuse, self).__init__(channel)
        # according to tdBN paper, the initialized weight is changed to alpha*Vth
        self.weight.data.mul_(0.5)

    def forward(self, x):
        B, T, *spatial_dims = x.shape
        out = super().forward(x.reshape(B * T, *spatial_dims))
        BT, *spatial_dims = out.shape
        out = out.view(B, T, *spatial_dims).contiguous()
        return out

class tdBatchNorm(nn.BatchNorm2d):
    def __init__(self, channel):
        super(tdBatchNorm, self).__init__(channel)
        # according to tdBN paper, the initialized weight is changed to alpha*Vth
        self.weight.data.mul_(2)

    def forward(self, x):
        out = super().forward(x)
        return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


if __name__ == "__main__":
    input_ = torch.randn(8, 3, 2, 128, 128)
    model = Seq2ANN(tdBatchNorm(2))
    print(model)
    output_ = model(input_)
    print(output_.shape)
