import random

import torch

from models.layer import *
from models.MPE_PSN import *

# ------------------- #
#     Neuron          #
# ------------------- #

class TriangularSurrogate(torch.autograd.Function):
    r"""
    default alpha=1.0
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
        temp = (1 / ctx.alpha) * (1 / ctx.alpha) * (
            (ctx.alpha - input.abs()).clamp(min=0)
        )
        return grad_input * temp, None

act_fun = TriangularSurrogate.apply

# ------------------- #
#   ResNet Example    #
# ------------------- #

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes, planes,
            stride=1, downsample=None, groups=1,
            base_width=64, dilation=1, norm_layer=None,
            spiking_neuron=None,timestep=10
    ):
        super(BasicBlock, self).__init__()
        if norm_layer == 'tdbn':
            norm_layer = tdBatchNorm
        elif norm_layer == 'tebn':
            norm_layer = TEBNLayer
        elif norm_layer == 'bn':
            norm_layer = BNLayer

        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.sn1 = spiking_neuron(T=timestep)
        self.sn2 = spiking_neuron(T=timestep)

        # if norm_layer is not None:
        if norm_layer == tdBatchNorm:
            self.conv1 = tdLayer(
                conv3x3(inplanes, planes, stride),
                norm_layer((planes))
            )
            self.conv2 = tdLayer(
                conv3x3(planes, planes),
                norm_layer(planes)
            )
        elif norm_layer == TEBNLayer:
            self.conv1 = TEBNLayer(inplanes, planes,3, stride,1)
            self.conv2 = TEBNLayer(planes, planes, 3, 1,1)

        else:
            self.conv1 = tdLayer(conv3x3(inplanes, planes, stride),bn=BNLayer(planes))
            self.conv2 = tdLayer(conv3x3(planes, planes),bn=BNLayer(planes))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.sn1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.sn2(out)
        return out


# Networks ################################################################
class SpikingResNet(nn.Module):
    r"""
    128 width S-ResNet
    """

    def __init__(
            self,
            block, layers,
            num_classes=1000,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            spiking_neuron=None,
            # n_input=[3, 32, 32],
            n_input=[2, 48, 48],
            timestep=10,
            **kwargs
    ):

        super(SpikingResNet, self).__init__()
        self.spiking_neuron = spiking_neuron
        if norm_layer == 'tdbn':
            norm_layer = tdBatchNorm
        elif norm_layer == 'tebn':
            norm_layer = TEBNLayer
        self._norm_layer = norm_layer
        self.in_dim = n_input
        self.inplanes = 64
        self.dilation = 1
        self.T = timestep
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        # 此处改为生成多个snn ，因为使用可学习的snn
        self.sn1 = spiking_neuron(T=self.T)
        self.sn2 = spiking_neuron(T=self.T)

        # if norm_layer is not None:
        if norm_layer == tdBatchNorm:
            self.conv1 = tdLayer(
                nn.Conv2d(
                    n_input[0], self.inplanes,
                    kernel_size=3, stride=1, padding=1, bias=False
                ),
                norm_layer(self.inplanes)
            )
        elif norm_layer == TEBNLayer:

            # self.conv1 = TEBNLayer(
            #         n_input[0], self.inplanes,
            #         kernel_size=3, stride=1, padding=1, bias=False
            #     )
            self.conv1 = TEBNLayer(
                n_input[0], self.inplanes,
                kernel_size=3, stride=1, padding=1,
            )
        else:
            self.conv1 = tdLayer(
                nn.Conv2d(
                    n_input[0], self.inplanes,
                    kernel_size=3, stride=1, padding=1, bias=False
                 ),
                bn=BNLayer(self.inplanes)
            )

        self.layer1 = self._make_layer(
            block, 128, layers[0],
            # spiking_neuron=spiking_neuron,
            **kwargs
        )
        self.layer2 = self._make_layer(
            block, 256, layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            # spiking_neuron=spiking_neuron,
            **kwargs
        )
        self.layer3 = self._make_layer(
            block, 512, layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            # spiking_neuron=spiking_neuron,
            **kwargs
        )
        self.avgpool = tdLayer(
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc1 = tdLayer(
            nn.Linear(512 * block.expansion, 256)
        )
        # self.fc1 = tdLayer(nn.Dropout(0.25),
        #     nn.Linear(512 * block.expansion, 256)
        # )
        self.fc2 = tdLayer(
            nn.Linear(256, num_classes)
        )
        # self.fc2 = tdLayer(nn.Dropout(0.25),
        #     nn.Linear(256, num_classes)
        # )

        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu'
                    )

    def _make_layer(
            self,
            block, planes, blocks,
            stride=1, dilate=False,
            # spiking_neuron= spiking_neuron,
            **kwargs
    ):
        spiking_neuron = self.spiking_neuron
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # if norm_layer is not None:
            if norm_layer == tdBatchNorm:
                downsample = tdLayer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion)
                )
            elif norm_layer == TEBNLayer:
                downsample = TEBNLayer(self.inplanes, planes * block.expansion, 1, stride, 0,)
            else:
                downsample = tdLayer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),bn=BNLayer(planes * block.expansion)
                )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer,
                spiking_neuron,timestep=self.T,
                **kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation,
                    norm_layer=norm_layer,
                    spiking_neuron=spiking_neuron,timestep=self.T,
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)

        x = self.sn1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        x = self.sn2(x)
        x = self.fc2(x)
        # mod 0408 15.24
        # x = self.sn(x, out_u=True)
        return x

    def forward(self, x):
        """ if x.dim() == 2:
            # [N, C*H*W]
            bsz = x.size(0)
            x = x.view(bsz, self.in_dim[0], self.in_dim[1], self.in_dim[2]) """
        if len(x.size()) != 5:
            x = add_dimention(x, self.T)

        return self._forward_impl(x)


class SpikingResNet_type2(nn.Module):
    r"""
    64-width S-ResNet
    """

    def __init__(
            self,
            block, layers,
            num_classes=10,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer='bn',
            spiking_neuron: callable = None,
            # n_input=[3, 32, 32],
            n_input=[2, 48, 48],
            **kwargs
    ):
        super(SpikingResNet_type2, self).__init__()
        if norm_layer == 'tdbn':
            norm_layer = tdBatchNorm
        elif norm_layer == 'tebn':
            norm_layer = TEBNLayer
        self._norm_layer = norm_layer
        self.in_dim = n_input
        self.inplanes = 64
        self.dilation = 1
        self.T = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(
                    replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        self.sn = spiking_neuron()

        if n_input[1] == 32:
            conv1_ks = 3
            conv1_stride = 1
            conv1_padding = 1
        else:
            conv1_ks = 7
            conv1_stride = 2
            conv1_padding = 3

        if norm_layer == tdBatchNorm:
            self.conv1 = tdLayer(
                nn.Conv2d(
                    n_input[0], self.inplanes,
                    kernel_size=conv1_ks,
                    stride=conv1_stride,
                    padding=conv1_padding,
                    bias=False
                ),
                norm_layer(self.inplanes)
            )
        elif norm_layer == TEBNLayer:
            self.conv1 = TEBNLayer(n_input[0], self.inplanes,kernel_size=conv1_ks,stride=conv1_stride,padding=conv1_padding)
            # self.conv1 = TEBNLayer(n_input[0], self.inplanes,kernel_size=conv1_ks,stride=conv1_stride,padding=conv1_padding,bias=False)
        else:
            self.conv1 = tdLayer(
                nn.Conv2d(
                    n_input[0], self.inplanes,
                    kernel_size=conv1_ks,
                    stride=conv1_stride,
                    padding=conv1_padding,
                    bias=False
                ),
            )

        self.maxpool = tdLayer(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(
            block, 64, layers[0],
            # spiking_neuron=spiking_neuron,
            **kwargs
        )
        self.layer2 = self._make_layer(
            block, 128, layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            spiking_neuron=spiking_neuron,
            **kwargs
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            spiking_neuron=spiking_neuron,
            **kwargs
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            spiking_neuron=spiking_neuron,
            **kwargs
        )

        self.avgpool = tdLayer(
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc1 = tdLayer(
            nn.Linear(512 * block.expansion, num_classes)
        )

        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
                elif isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu'
                    )

    def _make_layer(
            self,
            block, planes, blocks,
            stride=1, dilate=False,
            # spiking_neuron= SpikeLIF,
            **kwargs
    ):
        spiking_neuron = self.spiking_neuron
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if norm_layer == tdBatchNorm:
                downsample = tdLayer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion)
                )
            elif norm_layer == TEBNLayer:
                downsample = TEBNLayer(self.inplanes, planes * block.expansion,1, stride,0)
            else:
                downsample = tdLayer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups,
                self.base_width, previous_dilation, norm_layer,
                spiking_neuron,
                **kwargs
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation,
                    norm_layer=norm_layer,
                    spiking_neuron=spiking_neuron,
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        # only sn use SpikeLIF
        x = self.sn(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)
        x = self.fc1(x)
        return x

    def forward(self, x):
        if len (x.size()) != 5:
            x = add_dimention(x, self.T)
        return self._forward_impl(x)


def _spiking_resnet(
        block, layers, pretrained, progress,
        spiking_neuron,
        n_input, n_output,
        **kwargs
):
    model = SpikingResNet(
        block, layers,
        spiking_neuron=spiking_neuron,
        n_input=n_input,
        num_classes=n_output,
        **kwargs
    )

    return model


def _spiking_resnet_type2(
        block, layers, pretrained, progress,
        spiking_neuron,
        n_input, n_output,
        **kwargs
):
    model = SpikingResNet_type2(
        block, layers,
        spiking_neuron=spiking_neuron,
        n_input=n_input,
        num_classes=n_output,
        **kwargs
    )

    return model


def resnet19(
        pretrained=False,
        progress=True,
        spiking_neuron: callable = None,
        # n_input=[3, 32, 32],
        n_input=[2, 48, 48],
        n_output=10,
        norm_layer='tebn',
        **kwargs
):
    r"""
    A spiking version of ResNet-19 model
    """

    return _spiking_resnet(
        BasicBlock,
        [3, 3, 2],
        pretrained,
        progress,
        spiking_neuron,
        n_input,
        n_output,
        width_per_group=64,
        norm_layer=norm_layer,
        **kwargs
    )


def resnet18(
        pretrained=False,
        progress=True,
        spiking_neuron: callable = None,
        n_input=[2, 48, 48],
        # n_input=[3, 32, 32],
        n_output=10,
        norm_layer='bn',
        **kwargs
):
    r"""
    A spiking version of ResNet-18 model
    """

    return _spiking_resnet_type2(
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        spiking_neuron,
        n_input,
        n_output,
        width_per_group=64,
        norm_layer=norm_layer,
        **kwargs
    )



if __name__ == '__main__':
    # model = resnet19(n_output=10, norm_layer="bn",spiking_neuron=SpikeLIF_3,timestep=5)
    # model = resnet19(n_output=10, norm_layer="tebn").to("cuda")
    # # model = BasicBlock(3,32,norm_layer='tebn',spiking_neuron=SpikeLIF)
    # # model.T = 3
    # x = torch.rand(1,10,2,32,32).to("cuda") #NTCHW
    # y = model(x)
    # print(x)
    # print(y, y.shape)
    # LIF  = SpikeLIF_3()
    # x= torch.rand([10,2,2,2])
    # print(x)
    # print(LIF(x))
    # print(model)
    ...