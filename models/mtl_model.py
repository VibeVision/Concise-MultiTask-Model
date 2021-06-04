
'''
Project         : Global-Reasoned Multi-Task Surgical Scene Understanding
Lab             : MMLAB, National University of Singapore
contributors    : Lalithkumar Seenivasan, Sai Mitheran, Mobarakol Islam, Hongliang Ren
Note            : Code adopted and modified from Visual-Semantic Graph Attention Networks and Dual attention network for scene segmentation
'''

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision
import torch.nn as nn

class mtl_model(nn.Module):
    '''
    Multi-task model : Graph Scene Understanding and segmentation
    Forward uses features from feature_extractor
    '''