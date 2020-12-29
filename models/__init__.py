from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from .vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from .ensemble import EnsembleFC, EnsembleFV

networks = {'vgg11_bn': vgg11_bn, 'vgg16_bn': vgg16_bn, 
            'resnext50': resnext50_32x4d, 'resnext101': resnext101_32x8d, 
            'resnet18': resnet18, 'resnet34': resnet34, 'resnet50': resnet50, 'resnet101': resnet101, 'resnet152': resnet152,
            'wideresnet50': wide_resnet50_2, 'wideresnet101': wide_resnet101_2,
            'ensemble_fc': EnsembleFC, 'ensemble_fv': EnsembleFV}