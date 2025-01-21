import torch
import torch.nn as nn



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 归一化操作
class TensorNormalization(nn.Module):
    def __init__(self,mean, std):
        super(TensorNormalization, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.mean = mean
        self.std = std
    def forward(self,X):
        return normalizex(X,self.mean,self.std)
def normalizex(tensor, mean, std):
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    if mean.device != tensor.device:
        mean = mean.to(tensor.device)
        std = std.to(tensor.device)
    return tensor.sub(mean).div(std)

class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class tdLayer(nn.Module):
    def __init__(self, layer, bn=None):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
        self.bn = bn

    def forward(self, x):
        x_ = self.layer(x)
        if self.bn is not None:
            x_ = self.bn(x_)
        return x_

class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y

def add_dimention(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x


class TEBN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(TEBN, self).__init__()
        self.bn = nn.BatchNorm3d(num_features)
        self.p = nn.Parameter(torch.ones(10, 1, 1, 1, 1, device=device))

    def forward(self, input):
        y = input.transpose(1, 2).contiguous()  # N T C H W ,  N C T H W
        y = self.bn(y)
        y = y.contiguous().transpose(1, 2)
        y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        y = y * self.p
        y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
        return y


class TEBNLayer(nn.Module):
    def __init__(self, in_plane, out_plane, kernel_size, stride=1, padding=1):
        super(TEBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TEBN(out_plane)

    def forward(self, input):
        y = self.fwd(input)
        y = self.bn(y)
        return y


class BNLayer(nn.Module):
    def __init__(self,out_plane):
        super(BNLayer,self).__init__()
        self.bn = nn.BatchNorm3d(out_plane)

    def forward(self, input):
        y = input.transpose(1, 2).contiguous()  # N T C H W ,  N C T H W
        y = self.bn(y)
        y = y.contiguous().transpose(1, 2)
        return y
