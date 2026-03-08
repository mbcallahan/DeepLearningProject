import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.celu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.celu(out)
        return out

class SmallStemResNet18(nn.Module):
    """
    ResNet-18 with a 3x3 stride-1 stem and no maxpool, suitable for 28x28 inputs. Skeleton form copilot. I replaced the relu with smoothed version celu hoping this will speed up training. 
    """
    def __init__(self, num_classes=10, in_channels=1, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64

        # Small-image stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        
        # maxpool removed

        # Residual layers: (2,2,2,2) blocks with channel sizes (64,128,256,512)
        self.layer1 = self._make_layer(64,  blocks=2, stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(128, blocks=2, stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(256, blocks=2, stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(512, blocks=2, stride=2, norm_layer=norm_layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        # He init like torchvision
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, planes, blocks, stride=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=norm_layer))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   # [B, 64, 28, 28]
        x = self.bn1(x)
        x = F.celu(x)
        # no maxpool

        x = self.layer1(x)  # -> [B, 64, 28, 28]
        x = self.layer2(x)  # -> [B, 128, 14, 14]
        x = self.layer3(x)  # -> [B, 256, 7, 7]
        x = self.layer4(x)  # -> [B, 512, 4, 4] (since 7/2 -> 4 via floor)

        x = self.avgpool(x) # -> [B, 512, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)      # -> [B, num_classes]
        return x


if __name__ == "__main__":
    model = SmallStemResNet18(num_classes=10, in_channels=1)
    x = torch.randn(16, 1, 28, 28)
    y = model(x)
    print(y.shape)  # -> [16, 10]
