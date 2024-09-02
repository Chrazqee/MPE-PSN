import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate
import torch.distributed as dist

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


class LIFSpike(nn.Module):
    def __init__(self, thresh=1.0, tau=0.25, gamma=1.0):
        super(LIFSpike, self).__init__()
        self.heaviside = ZIF.apply
        self.v_th = thresh
        self.tau = tau
        self.gamma = gamma
        self.pre_spike_mem = []

    def forward(self, x):
        mem_v = []
        _mem = []
        mem = 0
        T = x.shape[1]
        for t in range(T):
            mem = self.tau * mem + x[:, t, ...]
            _mem.append(mem.detach().cpu().clone())
            spike = self.heaviside(mem - self.v_th, self.gamma)
            mem = mem * (1 - spike)
            mem_v.append(spike)
        self.pre_spike_mem = torch.stack(_mem)
        return torch.stack(mem_v, dim=1)


class MaskedSlidingPSN(nn.Module):

    def gen_gemm_weight(self, T: int):
        weight = torch.zeros([T, T], device=self.weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.order)
            length = min(end - start, self.order)
            weight[i][start: end] = self.weight[self.order - length: self.order]

        return weight

    def __init__(self, order: int = 2, surrogate_function=surrogate.ATan(), exp_init: bool = True):
        super().__init__()

        self.order = order
        if exp_init:
            weight = torch.ones([order])
            for i in range(order - 2, -1, -1):
                weight[i] = weight[i + 1] / 2.

            self.weight = nn.Parameter(weight)
        else:
            self.weight = torch.ones([1, order])
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.weight = nn.Parameter(self.weight[0])
        self.threshold = nn.Parameter(torch.as_tensor(-1.))
        self.surrogate_function = surrogate_function

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [N, T, *]
        weight = self.gen_gemm_weight(x_seq.shape[1])
        h_seq = F.linear(x_seq.transpose(1, -1), weight, self.threshold)  # bias = threshold
        h_seq = h_seq.transpose(1, -1)

        return self.surrogate_function(h_seq)


class VGGPSN(nn.Module):
    def __init__(self, tau=0.5):
        super(VGGPSN, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            MaskedSlidingPSN(),  # surrogate.ATan() 做线性层的初始化权重，过线性层，再激活
            Layer(64, 128, 3, 1, 1),
            MaskedSlidingPSN(),
            pool,
            Layer(128, 256, 3, 1, 1),
            MaskedSlidingPSN(),
            Layer(256, 256, 3, 1, 1),
            MaskedSlidingPSN(),
            pool,
            Layer(256, 512, 3, 1, 1),
            MaskedSlidingPSN(),
            Layer(512, 512, 3, 1, 1),
            MaskedSlidingPSN(),
            pool,
            Layer(512, 512, 3, 1, 1),
            MaskedSlidingPSN(),
            Layer(512, 512, 3, 1, 1),
            MaskedSlidingPSN(),
            pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 10
        self.classifier = nn.Sequential(SeqToANNContainer(nn.Dropout(0.25), nn.Linear(512 * W * W, 10)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # print(input.shape)
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGSNN(nn.Module):
    def __init__(self, tau=0.5):
        super(VGGSNN, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            TEBNLayer(2, 64, 3, 1, 1),
            LIFSpike(tau=self.tau),
            TEBNLayer(64, 128, 3, 1, 1),
            LIFSpike(tau=self.tau),
            pool,
            TEBNLayer(128, 256, 3, 1, 1),
            LIFSpike(tau=self.tau),
            TEBNLayer(256, 256, 3, 1, 1),
            LIFSpike(tau=self.tau),
            pool,
            TEBNLayer(256, 512, 3, 1, 1),
            LIFSpike(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1),
            LIFSpike(tau=self.tau),
            pool,
            TEBNLayer(512, 512, 3, 1, 1),
            LIFSpike(tau=self.tau),
            TEBNLayer(512, 512, 3, 1, 1),
            LIFSpike(tau=self.tau),
            pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 10
        self.classifier = nn.Sequential(nn.Dropout(0.25), SeqToANNContainer(nn.Linear(512 * W * W, 10)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x

# ======================================================================================================================
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

        self.coef = nn.Parameter(torch.tensor([1 / T for _ in range(T)]).view(1, T, 1, 1, 1))  # N, T, C, H, W; 均匀分布
        # self.coef = nn.Parameter(torch.linspace(1., 0.01, T - 1).view(1, T - 1, 1, 1, 1), requires_grad=True)
        # self.p = nn.Parameter(torch.as_tensor(0.2), requires_grad=True)
        # self.soft_max = nn.Softmax2d()
        self.mem_loss = 0
        self.dist = 0

    def forward(self, x: torch.Tensor):
        # x: [N, T, ...]
        device = x.device
        # N, T, C, H, W = x.shape
        # 对初始化的膜电位 进行一次 投影+激活
        # soft_max_x = self.soft_max(x.view(-1, C, H, W)).view(N, T, C, H, W)
        # soft_max_x = F.softmax(x, dim=1)
        # mem_hat = (1 - torch.bernoulli(soft_max_x)) * x
        sig_x = torch.sigmoid(x)
        mem_hat = (1 - torch.bernoulli(sig_x)) * x
        v_0 = torch.zeros(x.shape[0], 1, *list(x.shape[2:]), device=device)
        # mem_hat = torch.cat((v_0, mem_hat[:, 1:, ...]), dim=1)
        mem_hat = torch.cat((v_0, mem_hat), dim=1)
        mem_memo = mem_hat[:, :-1, ...] * self.tau + x
        o_t = self.surrogate_function(mem_memo - self.threshold)
        mem_real = (1 - o_t) * mem_memo  # 用于计算 loss
        self.mem_loss = (F.mse_loss(mem_hat[:, 1:, ...], mem_real, reduction='none') * self.coef).mean().to(device)
        return o_t


class MPE_PSN_(nn.Module):
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
        # self.p = nn.Parameter(torch.as_tensor(0.2), requires_grad=True)
        # self.soft_max = nn.Softmax2d()
        self.mem_loss = 0

    def forward(self, x: torch.Tensor):
        # x: [N, T, ...]
        device = x.device
        # N, T, C, H, W = x.shape
        # 对初始化的膜电位 进行一次 投影+激活
        # soft_max_x = self.soft_max(x.view(-1, C, H, W)).view(N, T, C, H, W)
        # soft_max_x = F.softmax(x, dim=1)
        # mem_hat = (1 - torch.bernoulli(soft_max_x)) * x
        sig_x = torch.sigmoid(x)
        mem_hat = (1 - torch.bernoulli(sig_x)) * x
        v_0 = torch.zeros(x.shape[0], 1, *list(x.shape[2:]), device=device)
        mem_hat = torch.cat((v_0, mem_hat[:, 1:, ...]), dim=1)
        mem_memo = mem_hat * self.tau + x
        o_t = self.surrogate_function(mem_memo - self.threshold)
        mem_real = (1 - o_t) * mem_memo  # 用于计算 loss
        self.mem_loss = (torch.nn.functional.mse_loss(mem_hat[:, 1:, ...], mem_real[:, :-1, ...], reduction='none') * self.coef).mean().to(device)
        return o_t


class VGGMPE(nn.Module):
    def __init__(self, tau=0.25, T=None):
        super(VGGMPE, self).__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            MPE_PSN(T=T),  # surrogate.ATan() 做线性层的初始化权重，过线性层，再激活
            Layer(64, 128, 3, 1, 1),
            MPE_PSN(T=T),
            pool,
            Layer(128, 256, 3, 1, 1),
            MPE_PSN(T=T),
            Layer(256, 256, 3, 1, 1),
            MPE_PSN(T=T),
            pool,
            Layer(256, 512, 3, 1, 1),
            MPE_PSN(T=T),
            Layer(512, 512, 3, 1, 1),
            MPE_PSN(T=T),
            pool,
            Layer(512, 512, 3, 1, 1),
            MPE_PSN(T=T),
            Layer(512, 512, 3, 1, 1),
            MPE_PSN(T=T),
            pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 10
        self.classifier = nn.Sequential(SeqToANNContainer(nn.Dropout(0.25), nn.Linear(512 * W * W, 10)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # print(input.shape)
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class VGGPLIF(nn.Module):
    def __init__(self, tau=0.5):
        super().__init__()
        self.tau = tau
        pool = SeqToANNContainer(nn.AvgPool2d(2))
        # pool = APLayer(2)
        self.features = nn.Sequential(
            Layer(2, 64, 3, 1, 1),
            PLIF(),  # surrogate.ATan() 做线性层的初始化权重，过线性层，再激活
            Layer(64, 128, 3, 1, 1),
            PLIF(),
            pool,
            Layer(128, 256, 3, 1, 1),
            PLIF(),
            Layer(256, 256, 3, 1, 1),
            PLIF(),
            pool,
            Layer(256, 512, 3, 1, 1),
            PLIF(),
            Layer(512, 512, 3, 1, 1),
            PLIF(),
            pool,
            Layer(512, 512, 3, 1, 1),
            PLIF(),
            Layer(512, 512, 3, 1, 1),
            PLIF(),
            pool,
        )
        W = int(48 / 2 / 2 / 2 / 2)
        # self.T = 10
        self.classifier = nn.Sequential(SeqToANNContainer(nn.Dropout(0.25), nn.Linear(512 * W * W, 10)))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, input):
        # print(input.shape)
        # input = add_dimention(input, self.T)
        x = self.features(input)
        x = torch.flatten(x, 2)
        x = self.classifier(x)
        return x


class PLIF(nn.Module):
    def __init__(self, init_tau=2.0, v_threshold=1.0, surrogate_function=surrogate.ATan()):
        super().__init__()
        self.surrogate_function = surrogate_function
        self.v_th = v_threshold
        self.pre_spike_mem = []
        init_w = - math.log(init_tau - 1)
        self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        mem_v = []
        _mem = []
        mem = 0
        T = x.shape[1]
        for t in range(T):
            mem = self.w * mem + x[:, t, ...]
            _mem.append(mem.detach().cpu().clone())
            spike = self.surrogate_function(mem - self.v_th)
            mem = mem * (1 - spike)
            mem_v.append(spike)
        self.pre_spike_mem = torch.stack(_mem)
        return torch.stack(mem_v, dim=1)

    def tau(self):
        return 1 / self.w.data.sigmoid().item()

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={0}, tau={self.tau()}'


# class PLIFNode(nn.Module):
#     def __init__(self, init_tau=2.0, v_threshold=1.0, v_reset=0.0, surrogate_function=surrogate.ATan()):
#         super().__init__()
#         self.v_threshold = v_threshold
#         self.v_reset = v_reset
#         self.surrogate_function = surrogate_function
#         init_w = - math.log(init_tau - 1)
#         self.w = nn.Parameter(torch.tensor(init_w, dtype=torch.float))
#
#     def forward(self, dv: torch.Tensor):
#         if self.v_reset is None:
#             # self.v += dv - self.v * self.w.sigmoid()
#             self.v += (dv - self.v) * self.w.sigmoid()
#         else:
#             # self.v += dv - (self.v - self.v_reset) * self.w.sigmoid()
#             self.v += (dv - (self.v - self.v_reset)) * self.w.sigmoid()
#         return self.spiking()
#
#     def tau(self):
#         return 1 / self.w.data.sigmoid().item()
#
#     def extra_repr(self):
#         return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, tau={self.tau()}'

if __name__ == "__main__":
    # a = torch.randn(16, 3, 2, 48, 48)
    # model = VGGSSN(0.25, 3)
    # model.train()
    # out = model(a)
    #
    # for m in model.modules():
    #     pass
    # print(out.shape)
    pass
