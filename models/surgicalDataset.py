'''
Project         : Global-Reasoned Multi-Task Surgical Scene Understanding
Lab             : MMLAB, National University of Singapore
contributors    : Lalithkumar Seenivasan, Sai Mitheran, Mobarakol Islam, Hongliang Ren
Note            : Code adopted and modified from Visual-Semantic Graph Attention Networks and Dual attention network for scene segmentation

'''


import os
import sys
import random

import h5py
import numpy as np
from glob import glob
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class SurgicalSceneConstants():
    '''
    Set the instrument classes and action classes, with path to