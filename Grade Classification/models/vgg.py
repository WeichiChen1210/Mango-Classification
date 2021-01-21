"""
VGG for ImageNet(224 x 224 images) or CIFAR
"""
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .shallownet import DownSampleBottleneck, AttentionModule

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

cfgs = {
    'A': [1, 1, 2, 2, 2],
    'B': [2, 2, 2, 2, 2],
    'D': [2, 2, 3, 3, 3],
    'E': [2, 2, 4, 4, 4],
}
"""
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
"""
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

class ShallowNet(nn.Module):
    def __init__(self, channels=None, expansion=None, num_classes=3):
        super(ShallowNet, self).__init__()
        if expansion is None:
            expansion = 1
        layers = []
        for idx in range(1, len(channels)):
            layers += [DownSampleBottleneck(channels[idx-1] * expansion, channels[idx] * expansion)]
        self.bottleneck = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * expansion, num_classes)
    
    def forward(self, x, get_outputs=True):
        x = self.bottleneck(x)
        x = self.avgpool(x)
        feature = torch.flatten(x, 1)
        
        if get_outputs:
            out = self.fc(feature)
            return out, feature
        else:
            return feature

class ShallowAttentionNet(ShallowNet):
    def __init__(self, channels=None, expansion=None, num_classes=3):
        super(ShallowAttentionNet, self).__init__(channels, expansion, num_classes)
        if expansion is None:
            expansion = 1
        self.attention = AttentionModule(channels[0] * expansion, channels[0] * expansion)
    
    def forward(self, x, get_outputs=True):
        x = self.attention(x)
        return super().forward(x, get_outputs)

class VGG(nn.Module):
    """
    VGG backbone
    """
    def __init__(self, num_layers=None, batch_norm=True, init_weights=True, num_classes=100):
        super(VGG, self).__init__()
        self.conv1 = self._make_block(3, 64, num_layer=num_layers[0], batch_norm=batch_norm)
        self.conv2 = self._make_block(64, 128, num_layer=num_layers[1], batch_norm=batch_norm)
        self.conv3 = self._make_block(128, 256, num_layer=num_layers[2], batch_norm=batch_norm)
        self.conv4 = self._make_block(256, 512, num_layer=num_layers[3], batch_norm=batch_norm)
        self.conv5 = self._make_block(512, 512, num_layer=num_layers[4], batch_norm=batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            # nn.Linear(512 * 1 * 1, 256),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(256, 128),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(128, num_classes)
            nn.Linear(1024, num_classes)
        )
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        feature_list = []
        feature1 = self.conv1(x)
        feature_list.append(feature1)

        feature2 = self.conv2(feature1)
        feature_list.append(feature2)

        feature3 = self.conv3(feature2)
        feature_list.append(feature3)

        feature4 = self.conv4(feature3)
        feature_list.append(feature4)

        feature5 = self.conv5(feature4)

        feature5 = self.avgpool(feature5)
        feature5 = torch.flatten(feature5, 1)
        feature_list.append(feature5)

        out = self.classifier(feature5)

        return out, feature_list

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_block(self, in_channels, out_channels, num_layer, batch_norm=False):
        block = []
        for _ in range(num_layer):
            block += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
            
            if batch_norm:
                block += [nn.BatchNorm2d(out_channels)]
            
            block += [nn.ReLU(inplace=True)]
            in_channels = out_channels
        
        block += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*block)

class DistillNet(nn.Module):
    def __init__(self, backbone=None, num_classes=100, use_attention=False, **kwargs):
        super(DistillNet, self).__init__()
        self.backbone = backbone
        
        self.classifier1, self.classifier2, self.classifier3, self.classifier4 = None, None, None, None
        if use_attention:
            self.classifier1 = ShallowAttentionNet(channels=[64, 128, 256, 512, 512], num_classes=num_classes)
            self.classifier2 = ShallowAttentionNet(channels=[128, 256, 512, 512], num_classes=num_classes)
            self.classifier3 = ShallowAttentionNet(channels=[256, 512, 512], num_classes=num_classes)
            self.classifier4 = ShallowAttentionNet(channels=[512, 512], num_classes=num_classes)
        else:
            self.classifier1 = ShallowNet(channels=[64, 128, 256, 512, 512], num_classes=num_classes)
            self.classifier2 = ShallowNet(channels=[128, 256, 512, 512], num_classes=num_classes)
            self.classifier3 = ShallowNet(channels=[256, 512, 512], num_classes=num_classes)
            self.classifier4 = ShallowNet(channels=[512, 512], num_classes=num_classes)

    def forward(self, x, get_outputs=True):
        out5, feature_list = self.backbone(x)
        feature5 = feature_list[-1]

        if get_outputs:
            out1, feature1 = self.classifier1(feature_list[0], True)
            out2, feature2 = self.classifier2(feature_list[1], True)
            out3, feature3 = self.classifier3(feature_list[2], True)
            out4, feature4 = self.classifier4(feature_list[3], True)

            return [out5, out4, out3, out2, out1], [feature5, feature4, feature3, feature2, feature1]
        else:
            feature1 = self.classifier1(feature_list[0], False)
            feature2 = self.classifier2(feature_list[1], False)
            feature3 = self.classifier3(feature_list[2], False)
            feature4 = self.classifier4(feature_list[3], False)

            return [feature5, feature4, feature3, feature2, feature1]

def _vgg(arch, cfg, batch_norm, pretrained, progress, use_attention=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    
    backbone = VGG(num_layers=cfgs[cfg], batch_norm=batch_norm, **kwargs)
    
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        backbone.load_state_dict(state_dict)
    
    model = DistillNet(backbone=backbone, use_attention=use_attention, **kwargs)
    return model

def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13', 'B', False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg13_bn', 'B', True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16', 'D', False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg16_bn', 'D', True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19_bn', 'E', True, pretrained, progress, **kwargs)