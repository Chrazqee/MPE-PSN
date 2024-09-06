import torch
import torch.nn as nn

from utils import Seq2ANN, conv1x1, conv3x3, tdBatchNorm


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, neuron=None, downsample=None, is_tdbn=False, **kwargs):
        super().__init__()
        self.stride = stride
        self.seq1 = Seq2ANN(conv1x1(in_planes=in_channel, out_planes=out_channel),
                            nn.BatchNorm2d(out_channel))
        self.lif1 = neuron(**kwargs)

        self.seq2 = Seq2ANN(conv3x3(in_planes=out_channel, out_planes=out_channel, stride=stride),
                            nn.BatchNorm2d(out_channel) if not is_tdbn else tdBatchNorm(out_channel))
        self.lif2 = neuron(**kwargs)

        self.seq3 = Seq2ANN(conv1x1(in_planes=out_channel, out_planes=out_channel * self.expansion),
                            nn.BatchNorm2d(out_channel * self.expansion) if not is_tdbn else tdBatchNorm(out_channel * self.expansion))

        self.downsample = downsample
        # if `v_reset = None`, soft_reset will be trigger!!!
        self.lif3 = neuron(**kwargs)

    def forward(self, x):
        identity = x
        y = self.seq1(x)
        y = self.lif1(y.permute(1, 0, 2, 3, 4)).permute(1, 0, 2, 3, 4)

        y = self.seq2(y)
        y = self.lif2(y.permute(1, 0, 2, 3, 4)).permute(1, 0, 2, 3, 4)

        y = self.seq3(y)

        if self.downsample is not None:
            identity = self.downsample(identity)

        y += identity
        y = self.lif3(y.permute(1, 0, 2, 3, 4)).permute(1, 0, 2, 3, 4)
        return identity + y  # To avoid the sparsity of features.


if __name__ == "__main__":
    input_ = torch.randn(8, 3, 2, 128, 128)
    downsample_ = Seq2ANN()
    model = BottleNeck(2, 64, stride=1, downsample=None, is_tdbn=True)
    print(model)
    output_ = model(input_)
    print(output_.shape)
