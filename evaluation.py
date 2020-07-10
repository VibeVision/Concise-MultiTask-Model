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
    '''
    seg_criterion = SegmentationLosses(se_loss=False, aux=False, nclass=8, se_weight=0.2, aux_weight=0.2)
    loss = seg_criterion(seg_output, target)
    correct, labeled = batch_pix_accuracy(seg_output.data, target)
    inter, union = batch_intersection_union(seg_output.data, target, 8)  # 8 is num classes
    return correct, labeled, inter, union, loss


def build_model(args):
    '''
    Build MTL model
    1) Scene Graph Understanding Model
    2) Segmentation Model : Encoder, Reasoning unit, Decoder

    Inputs: args
    '''

    '''==== Graph model ===='''
    # graph model
    scene_graph = AGRNN(bias=True, bn=False, dropout=0.3, multi_attn=False, layer=1, diff_edge=False, global_feat=args.global_feat)

    # segmentation model
    seg_model = get_gcnet(backbone='resnet18_model', pretrained=False)
    model = mtl_model(seg_model.pretrained, scene_graph, seg_model.gr_interaction, seg_model.gr_decoder, seg_mode = args.seg_mode)
    model.to(torch.device('cpu'))
    return model



def model_eval(model, validation_dataloader, nclass=8):
    '''
    Evaluate MTL
    '''

    model.eval()

    class_values = np.zeros(nclass)

    # graph
    scene_graph_criterion = nn.MultiLabelSoftMarginLoss()
    scene_graph_edge_count = 0
    scene_graph_total_acc = 0.0
    scene_graph_total_loss = 0.0
    scene_graph_logits_list = []
    scene_graph_labels_list = []

    test_seg_los