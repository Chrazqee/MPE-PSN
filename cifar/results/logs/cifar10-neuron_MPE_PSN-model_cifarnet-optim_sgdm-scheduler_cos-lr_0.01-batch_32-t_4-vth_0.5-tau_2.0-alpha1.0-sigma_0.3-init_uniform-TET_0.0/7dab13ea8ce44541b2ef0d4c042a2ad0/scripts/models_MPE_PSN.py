'''
Author: ----
Date: 2022-04-29 09:43:08
LastEditors: GhMa
LastEditTime: 2022-09-20 14:35:41
'''
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


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

surrogate_fn = TriangularSurrogate.apply

class Neuron(nn.Module):
    def __init__(self,
                 T: int = None,
                 tau: float = 0.25,
                 threshold: float = None,
                 decay_input = False,
                 alpha = None,
                 sigma = None):
        super().__init__()
        assert T is not None, "T must be not None!!!"
        self.T = T
        self.tau = tau

        # self.threshold = nn.Parameter(torch.as_tensor(1.)) if threshold is None else threshold
        self.threshold = 0.5
        self.surrogate_function = surrogate_fn
        self.coef = nn.Parameter(torch.tensor([1 / (T - 1) for _ in range(T - 1)]).view(1, T - 1, 1, 1, 1))  # N, T, C, H, W; 均匀分布
        # self.coef = nn.Parameter(torch.linspace(1., 0.01, T - 1).view(1, T - 1, 1, 1, 1), requires_grad=True)
        # self.p = nn.Parameter(torch.as_tensor(0.2), requires_grad=True)
        # self.soft_max = nn.Softmax2d()
        self.mem_loss = 0

    def forward(self, x: torch.Tensor):
        # x: [N, T, ...]
        device = x.device
        # N, T, C, H, W = x.shape
        # 对初始化的膜电位 进行一次 投影+激活
        # soft_max_x = self.soft_max(x.view(-1, C, H, W)).view(N, T, C, H, W)
        soft_max_x = F.softmax(x, dim=1)
        mem_hat = (1 - torch.bernoulli(soft_max_x)) * x
        # v_0 = torch.zeros(N, 1, C, H, W, device=device)
        v_0 = torch.zeros([x.shape[0], 1, *list(x.shape[2:])], device=device)
        mem_hat = torch.cat((v_0, mem_hat[:, 1:, ...]), dim=1)
        mem_memo = mem_hat * self.tau + x
        o_t = self.surrogate_function(mem_memo - self.threshold)
        mem_real = (1 - o_t) * mem_memo  # 用于计算 loss
        self.mem_loss = (torch.nn.functional.mse_loss(mem_hat[:, 1:, ...], mem_real[:, :-1, ...], reduction='none') * self.coef).mean().to(device)
        # self.mem_loss = (torch.nn.functional.mse_loss(mem_hat[:, 1:, ...], mem_real[:, :-1, ...])).to(device)
        return o_t
