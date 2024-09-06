import torch.nn as nn

try:
    from utils import Seq2ANN, conv3x3
except ImportError:
    from .utils import Seq2ANN, conv3x3


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, neuron, stride=1, downsample=None, T=None):
        super().__init__()
        self.stride = stride
        self.seq1 = Seq2ANN(conv3x3(in_planes=in_channel, out_planes=out_channel, stride=stride),
                            nn.BatchNorm2d(out_channel))
        self.lif1 = neuron(T=T)
        self.seq2 = Seq2ANN(conv3x3(in_planes=out_channel, out_planes=out_channel),
                            nn.BatchNorm2d(out_channel))
        self.downsample = downsample
        self.lif2 = neuron(T=T)

    def forward(self, x):
        identity = x  # B, T, C, H, W
        y = self.seq1(x)
        y = self.lif1(y)

        y = self.seq2(y)

        if self.downsample is not None:
            identity = self.downsample(identity)  # B, T, C, H, W

        # validation code will trigger an error, unless downsample module has been defined!
        y += identity
        y = self.lif2(y)
        return y  # To avoid the sparsity of features.
