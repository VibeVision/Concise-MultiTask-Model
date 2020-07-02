#from functools import lru_cache
import os
import time
import json

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
from models.segmentation_model import get_gcnet  # for the get_gcnet function

from utils.scene_graph_eval_matrix import *
from utils.segmentation_eval_matrix import *  # SegmentationLoss and Eval code

from tabulate import tabulate

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
warnings.filterwarnings('ignore')

def label_to_index(lbl):
    '''
    Label to index mapping
    Input: class label
    Output: class index
    '''
    return torch.tensor(map_dict.index(lbl))


def index_to_label(index):
    '''
    Index to label mapping
    Input: class index
    Output: class label
    '''
    return map_dict[index]



def seed_everything(seed=27):
    '''
    Set random seed for reproducibl