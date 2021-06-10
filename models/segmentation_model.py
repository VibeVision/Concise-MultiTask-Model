
'''
Project         : Global-Reasoned Multi-Task Surgical Scene Understanding
Lab             : MMLAB, National University of Singapore
contributors    : Lalithkumar Seenivasan, Sai Mitheran, Mobarakol Islam, Hongliang Ren
Note            : Code adopted and modified from Visual-Semantic Graph Attention Networks and Dual attention network for scene segmentation

                        @inproceedings{fu2019dual,
                        title={Dual attention network for scene segmentation},
                        author={Fu, Jun and Liu, Jing and Tian, Haijie and Li, Yong and Bao, Yongjun and Fang, Zhiwei and Lu, Hanqing},
                        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
                        pages={3146--3154},
                        year={2019}
                        }
'''


import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.functional import interpolate
from typing import Type, Any, Callable, Union, List, Optional

# Setting the kwargs for upsample configuration
up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class Namespace:
    """
    Namespace class for custom args to be parsed 
    Inputs: **kwargs

    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def get_backbone(name, **kwargs):
    """
    Function to get backbone feature extractor 
    Inputs: name of backbone, **kwargs

    """
    models = {
        'resnet18_model': resnet18_model,
    }
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (str(name), '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """
    3x3 convolution with padding
    Inputs: in_planes, out_planes, stride, groups, dilation

    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    1x1 convolution
    Inputs: in_planes, out_planes, stride

    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

        
class BasicBlock(nn.Module):
    """
    Basic block for ResNet18 backbone 
    init    : 
        inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer
        
    forward : x

    """
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")

        self.planes = planes

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck block for ResNet18 
    init    : 
        inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer
        
    forward : x

    """
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Downsampling of the input variable (x)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    ResNet base class for different variants
    init    : 
        block, layers, num_classes (ImageNet), zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer
        
    forward : x
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # Each element in the tuple indicates whether we should replace the 2x2 stride with a dilated convolution 
            replace_stride_with_dilation = [False, False, False]

        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)


    def _forward_impl(self, x) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4
 

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class BaseNet(nn.Module):
    """
    BaseNet class for Multi-scale global reasoned segmentation module

    init    : 
        block, layers, num_classes (ImageNet), zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer
        
    forward : x

    """
    def __init__(self, nclass, backbone, pretrained, dilated=True, norm_layer=None,
                root='~/.encoding/models', *args, **kwargs):
        super(BaseNet, self).__init__()
        self.nclass = nclass

        # Copying modules from pretrained models
        self.backbone = backbone
        self.pretrained = get_backbone(backbone, pretrained=pretrained, dilated=dilated,
                                       norm_layer=norm_layer, root=root,
                                       *args, **kwargs)
        self.pretrained.fc = None
        self._up_kwargs = up_kwargs

    def base_forward(self, x):

        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        c = self.pretrained.layer1(x)
        c = self.pretrained.layer2(c)
        c = self.pretrained.layer3(c)
        c = self.pretrained.layer4(c)

        return None, None, None, c

    def evaluate(self, x, target=None):
        pred = self.forward(x)
        if isinstance(pred, (tuple, list)):
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = batch_pix_accuracy(pred.data, target.data)
        inter, union = batch_intersection_union(
            pred.data, target.data, self.nclass)
        return correct, labeled, inter, union


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
    ) -> ResNet:

    """
    ResNet model function to load pre-trained model: Class call
    init    : 
        arch, block, layers, pretrained, progress, **kwargs
        
    forward : x
    """

    model = ResNet(block, layers, **kwargs)
    if pretrained:
        print("Loading pre-trained ImageNet weights")
        state_dict = torch.load('models/r18/resnet18-f37072fd.pth')
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = True, progress: bool = True, **kwargs: Any) -> ResNet:
    """
    ResNet18 model call function
    Inputs: pretrained, progress, **kwargs

    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

class Resnet18_main(nn.Module):
    """
    ResNet main function for feature extractor
    init    : pretrained, num_classes
    forward : x
    """
    def __init__(self, pretrained, num_classes=1000):

        super(Resnet18_main, self).__init__()
        resnet18_block = resnet18(
            pretrained=pretrained)

        resnet18_block.fc = nn.Conv2d(resnet18_block.inplanes, num_classes, 1)

        self.resnet18_block = resnet18_block
        self._normal_initialization(self.resnet18_block.fc)

        self.in_planes = 64
        self.kernel_size = 3


    def _normal_initialization(self, layer):

        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x):      
        c1, c2, c3, c4 = self.resnet18_block(x)
 
        return c1, c2, c3, c4


class GCN(nn.Module):
    """
    Graph Convolution network for Global interaction space 
    init    : 
        num_state, num_node, bias=False
        
    forward : x, scene_feat = None, model_type = None

    """
    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, padding=0,
                               stride=1, groups=1, bias=bias)
        self.x_avg_pool = nn.AvgPool1d(128,1)

    def forward(self, x, scene_feat = None, model_type = None):
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)

        if (model_type == 'amtl-t1' or model_type == 'mtl-t1') and scene_feat is not None:    # (x+h+(avg(x)*f))
            x_p = torch.matmul(self.x_avg_pool(x.permute(0, 2, 1).contiguous()), scene_feat)
            h = h + x + x_p.permute(0, 2, 1).contiguous()
        else:
            h = h + x
        
        h = self.relu(h)
        h = self.conv2(h)

        return h


class GloRe_Unit(nn.Module):
    """
    Global Reasoning Unit (GR/GloRe)
    init    : 
        num_in, num_mid, stride=(1, 1), kernel=1
        
    forward : x, scene_feat = None, model_type = None
    AMTL - Sequential MTL Optimisation
    MTL - Naive MTL Optimisation

    """    
    def __init__(self, num_in, num_mid, stride=(1, 1), kernel=1):