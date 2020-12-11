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

from utils.scene_graph_eval_matrix import *
from utils.segmentation_eval_matrix import *  


import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def seed_everything(seed=27):
    '''
    Set random seed for reproducible experiments
    Inputs: seed number 
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seg_eval_batch(seg_output, target):
    '''
    Calculate segmentation loss, pixel acc and IoU
    Inputs: predicted segmentation mask, GT segmentation mask 
    '''
    seg_criterion = SegmentationLosses(se_loss=False, aux=False, nclass=8, se_weight=0.2, aux_weight=0.2)
    loss = seg_criterion(seg_output, target)
    correct, labeled = batch_pix_accuracy(seg_output.data, target)
    inter, union = batch_intersection_union(seg_output.data, target, 8)  # 8 is num classes
    return correct, labeled, inter, union, loss

def get_checkpoint_loc(model_type, seg_mode = None):
    loc = None
    if model_type == 'amtl-t0' or model_type == 'amtl-t3':
        if seg_mode is None:
            loc = 'checkpoints/stl_s/stl_s/epoch_train/checkpoint_D153_epoch.pth'
        elif seg_mode == 'v1':
            loc = 'checkp