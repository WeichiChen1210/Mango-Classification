import torch
import torch.nn as nn

def downsample(channel_in, channel_out, stride=2):
    return nn.Sequential(
        nn.Conv2d(channel_in, 128, kernel_size=1, stride=1), 
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, channel_out, kernel_size=1, stride=1),
        nn.BatchNorm2d(channel_out),
        nn.ReLU())

class DownSampleBottleneck(nn.Module):
    """Bottleneck to do downsample"""
    def __init__(self, channel_in, channel_out, stride=2):
        super(DownSampleBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, 128, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, channel_out, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(channel_out)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

class AttentionModule(nn.Module):
    """Attention module"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, affine=True):
        super(AttentionModule, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(in_channels, affine=affine)
        self.relu = nn.ReLU(False)

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=in_channels, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels, affine=affine)
        
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm2(x)
        x = self.relu(x)

        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.upsample(x)
        x = self.sigmoid(x)

        return x

class ShallowNet(nn.Module):
    """The shallow classifier containing a bottleneck block to make the feature map size fit the last classifier"""
    def __init__(self, channels=None, expansion=None, num_classes=3):
        super(ShallowNet, self).__init__()
        layers = []
        for idx in range(1, len(channels)):
            layers += [DownSampleBottleneck(channels[idx-1] * expansion, channels[idx] * expansion)]
        self.bottleneck = nn.Sequential(*layers)
        self.avgpool = nn.AvgPool2d(4, 4)
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
        """The shallow classifier containing an attention module and a bottleneck block to make the feature map size fit the last classifier"""
        super(ShallowAttentionNet, self).__init__(channels, expansion, num_classes)
        self.attention = AttentionModule(channels[0] * expansion, channels[0] * expansion)
    
    def forward(self, x, get_outputs=True):
        attention = self.attention(x)
        x = attention * x
        return super().forward(x, get_outputs)