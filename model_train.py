'''
Project         : Global-Reasoned Multi-Task Surgical Scene Understanding
Lab             : MMLAB, National University of Singapore
contributors    : Lalithkumar Seenivasan, Sai Mitheran, Mobarakol Islam, Hongliang Ren
'''

import os
import time

import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.mtl_model import *
from models.scene_graph import *
from models.surgicalDataset import *
from models.segmentation_model import get_gcnet 

from util