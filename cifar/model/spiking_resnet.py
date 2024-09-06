import torch
import torch.nn as nn

try:
    from layers.basic_block import BasicBlock
    from layers.utils import Seq2ANN, tdBatchNorm
except ImportError:
    from .layers.basic_block import BasicBlock
    from .layers.utils import Seq2ANN, tdBatchNorm


class SpikingResNet(nn.Module):
    def __init__(self, block, layers, neuron, num_classes=10, zero_init_residual=False, T=None):
        super().__init__()
        self.inplanes = 64
        self.neuron = neuron

        # self.init_seq = Seq2ANN(
        #     nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.BatchNorm2d(64))

        self.init_seq = Seq2ANN(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            # nn.AvgPool2d(2)
        )
        self.init_lif = neuron(T=T)
        # self.init_max_pooling = Seq2ANN(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.init_max_pooling = Seq2ANN(nn.AvgPool2d(2))

        self.conv2_x = self._make_layer(block, 64, layers[0], stride=1, T=T)
        self.conv3_x = self._make_layer(block, 128, layers[1], stride=2, T=T)
        self.conv4_x = self._make_layer(block, 256, layers[2], stride=2, T=T)
        self.conv5_x = self._make_layer(block, 512, layers[3], stride=2, T=T)

        # 输出维度
        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = Seq2ANN(nn.AvgPool2d(2), nn.Dropout(0.25))
        self.fc1 = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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

    def _make_layer(self, block, planes, blocks, stride=1, T=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Seq2ANN(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = list()
        layers.append(block(self.inplanes, planes, self.neuron, stride, downsample, T))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):  # 几个 block
            layers.append(block(self.inplanes, planes, self.neuron, T=T))

        return nn.Sequential(*layers)

    def forward(self, x):  # x \in B, C, H, W
        x = self.init_seq(x)  # B, T, C, H, W
        x = self.init_lif(x)
        x = self.init_max_pooling(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x).flatten(2)
        x = self.fc1(x)
        return x


class SpikingResNet19(nn.Module):
    def __init__(self, block, layers, neuron, num_classes=10, zero_init_residual=False, T=None):
        super().__init__()
        self.inplanes = 64
        self.neuron = neuron

        self.init_seq = Seq2ANN(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            # nn.AvgPool2d(2)
        )
        self.init_lif = neuron(T=T)
        # self.init_max_pooling = Seq2ANN(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        # self.init_max_pooling = Seq2ANN(nn.AvgPool2d(2))

        self.conv2_x = self._make_layer(block, 128, layers[0], stride=1, T=T)
        self.conv3_x = self._make_layer(block, 256, layers[1], stride=2, T=T)
        self.conv4_x = self._make_layer(block, 512, layers[2], stride=2, T=T)

        # 输出维度
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512*block.expansion, 256)
        self.fc_lif = neuron(T=T)
        self.fc2 = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

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

    def _make_layer(self, block, planes, blocks, stride=1, T=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Seq2ANN(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = list()
        layers.append(block(self.inplanes, planes, self.neuron, stride, downsample, T))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):  # 几个 block
            layers.append(block(self.inplanes, planes, self.neuron, T=T))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_seq(x)  # B, T, C, H, W
        x = self.init_lif(x)
        # x = self.init_max_pooling(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        # x = self.conv5_x(x)
        x = self.avg_pool(x).flatten(2)
        x = self.fc1(x)
        x = self.fc_lif(x)
        x = self.fc2(x)
        return x



def spiking_resnet18(num_classes=10, neuron=None, T=None):
    return SpikingResNet(BasicBlock,
                         [2, 2, 2, 2],
                         neuron=neuron,
                         num_classes=num_classes,
                         T=T)

def spiking_resnet19(num_classes=10, neuron=None, T=None):
    return SpikingResNet19(BasicBlock,
                         [3, 3, 2],
                         neuron=neuron,
                         num_classes=num_classes,
                         T=T)


if __name__ == "__main__":
    ...
