import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly
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


class Layer(nn.Module):  # baseline
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(Layer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
            nn.BatchNorm2d(out_plane)
        )
        # self.act = LIFSpike()

    def forward(self, x):
        x = self.fwd(x)
        # x = self.act(x)
        return x


class TEBN(nn.Module):
    def __init__(self, out_plane, eps=1e-5, momentum=0.1):
        super(TEBN, self).__init__()
        self.bn = SeqToANNContainer(nn.BatchNorm2d(out_plane))
        self.p = nn.Parameter(torch.ones(10, 1, 1, 1, 1, device=device))

    def forward(self, input):
        y = self.bn(input)
        y = y.transpose(0, 1).contiguous()  # NTCHW  TNCHW
        y = y * self.p
        y = y.contiguous().transpose(0, 1)  # TNCHW  NTCHW
        return y


class TEBNLayer(nn.Module):  # baseline+TN
    def __init__(self, in_plane, out_plane, kernel_size, stride, padding):
        super(TEBNLayer, self).__init__()
        self.fwd = SeqToANNContainer(
            nn.Conv2d(in_plane, out_plane, kernel_size, stride, padding),
        )
        self.bn = TEBN(out_plane)
        # self.act = LIFSpike()

    def forward(self, x):
        y = self.fwd(x)
        y = self.bn(y)
        # x = self.act(x)
        return y


# class ZIF(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, gama=1.0):
#         out = (input > 0).float()
#         L = torch.tensor([gama])
#         ctx.save_for_backward(input, out, L)
#         return out
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         (input, out, others) = ctx.saved_tensors
#         gama = others[0].item()
#         grad_input = grad_output.clone()
#         tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
#         grad_input = grad_input * tmp
#         return grad_input, None

# ======================================================================================================================
def heaviside(x: torch.Tensor):
    return (x >= 0.0).to(x)

class SigmoidSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, input, alpha, threshold
    ):
        ctx.save_for_backward(input)
        ctx.alpha = alpha
        return heaviside(input - threshold)

    @staticmethod
    def backward(ctx, grad_output):
        grad_ = None
        if ctx.needs_input_grad[0]:
            sgax = (ctx.saved_tensors[0] * ctx.alpha).sigmoid_()
            grad_x = grad_output * (1.0 - sgax) * sgax * ctx.alpha
        return grad_x, None, None


class AtanSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.alpha = alpha
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = ctx.alpha / 2 / (
                    1 + math.pi / 2 * ctx.alpha * ctx.saved_tensors[0].pow_(2)
            ) * grad_output
        return grad_x, None


class RectangularSurrogate(torch.autograd.Function):
    r"""
    default alpha=0.8
    """

    @staticmethod
    def forward(ctx, input, threshold=0.5, alpha=0.8):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        ctx.alpha = alpha  # surrogate gradient function hyper-parameter
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (2 * abs(input - ctx.threshold) < ctx.alpha) * 1. / ctx.alpha
        return grad_input * temp, None, None


class TriangularSurrogate(torch.autograd.Function):
    r"""
    default alpha=1.0
    """

    @staticmethod
    def forward(ctx, input, alpha=1.0):
        ctx.save_for_backward(input)
        ctx.alpha = alpha  # surrogate gradient function hyper-parameter
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (1 / ctx.alpha) * (1 / ctx.alpha) * (
            (ctx.alpha - input.abs()).clamp(min=0)
        )
        return grad_input * temp, None


class ERFSurrogate(torch.autograd.Function):
    r"""
    """

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.save_for_backward(input)
        ctx.alpha = alpha  # surrogate gradient function hyper-parameter
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = (- (input * ctx.alpha).pow_(2)).exp_() * (
                ctx.alpha / math.sqrt(math.pi)
        )
        return grad_output * temp, None


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama=1.0):
        out = (input > 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output.clone()
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class MPE_PSN(nn.Module):
    def __init__(self,
                 T: int = None,
                 tau: float = 0.25,
                 threshold: float = None,
                 surrogate_function=TriangularSurrogate):
        super().__init__()
        assert T is not None, "T must be not None!!!"
        self.T = T
        self.tau = tau

        self.threshold = nn.Parameter(torch.as_tensor(1.)) if threshold is None else threshold

        self.surrogate_function = surrogate_function.apply

        self.coef = nn.Parameter(torch.tensor([1 / (T - 1) for _ in range(T - 1)]).view(1, T - 1, 1, 1, 1))  # N, T, C, H, W; 均匀分布
        # self.coef = nn.Parameter(torch.linspace(1., 0.01, T - 1).view(1, T - 1, 1, 1, 1), requires_grad=True)
        self.p = nn.Parameter(torch.as_tensor(0.2), requires_grad=True)
        # self.soft_max = nn.Softmax2d()
        self.mem_loss = 0
        self.dist = 0

    def forward(self, x: torch.Tensor):
        # x: [N, T, ...]
        device = x.device
        # N, T, C, H, W = x.shape
        # 对初始化的膜电位 进行一次 投影+激活
        # soft_max_x = self.soft_max(x.view(-1, C, H, W)).view(N, T, C, H, W)
        soft_max_x = F.softmax(x, dim=1)
        mem_hat = (1 - torch.bernoulli(soft_max_x)) * x
        v_0 = torch.zeros(x.shape[0], 1, *list(x.shape[2:]), device=device)
        mem_hat = torch.cat((v_0, mem_hat[:, 1:, ...]), dim=1)
        mem_memo = mem_hat * self.tau + x
        o_t = self.surrogate_function(mem_memo - self.threshold)
        mem_real = (1 - o_t) * mem_memo  # 用于计算 loss
        self.mem_loss = (torch.nn.functional.mse_loss(mem_hat[:, 1:, ...], mem_real[:, :-1, ...], reduction='none') * self.coef).mean().to(device)
        self.calMemDist(x, mem_real)
        return o_t

    @torch.no_grad()
    def calMemDist(self, x, mem_real):
        # 计算膜电位之间的距离，体现 mem loss 的作用
        dist = 0  # 记录膜电位与 mem_real 之间的距离变化；使用 l_2 norm 度量
        mem = 0  # 第 0 个时间步的膜电位
        for i in range(self.T):
            mem = mem * self.tau + x[:, i, ...]
            o = (mem - self.threshold > 0).float()
            mem = (1 - o) * mem
            dist += torch.linalg.norm(mem - mem_real[:, i, ...]).item()
        self.dist = dist


class tdBatchNorm(nn.BatchNorm2d):
    """
    Implementation of tdBN in 'Going Deeper With Directly-Trained Larger Spiking Neural Networks '
    """
    def __init__(self, num_features, eps=1e-05, momentum=0.1, alpha=1, affine=True, track_running_stats=True, Vth=0.5):
        super(tdBatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha
        self.register_buffer('Vth', torch.tensor(Vth, dtype=torch.float))

    def forward(self, input):
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])   #T*Batch, channel, height, width
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        input = self.alpha * self.Vth * (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]
        return input


class DVSGestureNet(nn.Module):
    def __init__(self, channels=256, spiking_neuron: callable = MPE_PSN, use_tdbn=False, **kwargs):
        super().__init__()

        conv = []
        for i in range(5):
            if conv.__len__() == 0:
                in_channels = 2
            else:
                in_channels = channels

            conv.append(SeqToANNContainer(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
                                          tdBatchNorm(channels) if use_tdbn else nn.BatchNorm2d(channels)
                                          ))
            conv.append(spiking_neuron(**deepcopy(kwargs)))
            conv.append(SeqToANNContainer(nn.MaxPool2d(2, 2)))


        self.conv_fc = nn.Sequential(
            *conv,
            SeqToANNContainer(nn.Flatten(),
                              nn.Dropout(0.5),
                              nn.Linear(channels * 4 * 4, 1024)
                              ),
            spiking_neuron(**deepcopy(kwargs)),
            SeqToANNContainer(nn.Dropout(0.5),
                              nn.Linear(1024, 110),
                              ),
            SeqToANNContainer(nn.AvgPool1d(10, 10))
        )

    def forward(self, x: torch.Tensor):
        return self.conv_fc(x)


if __name__ == "__main__":
    a = torch.randn(16, 3, 2, 128, 128)
    model = DVSGestureNet(128, T=3)
    model.train()
    out = model(a)

    for m in model.modules():
        pass
    print(out.shape)
