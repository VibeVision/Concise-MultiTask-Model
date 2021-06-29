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
    Set the instrument classes and action classes, with path to XML and Word2Vec Features (if applicable)
    '''
    def __init__(self):
        self.instrument_classes = ('kidney', 'bipolar_forceps', 'prograsp_forceps', 'large_needle_driver',
                                   'monopolar_curved_scissors', 'ultrasound_probe', 'suction', 'clip_applier',
                                   'stapler', 'maryland_dissector', 'spatulated_monopolar_cautery')

        self.action_classes = ('Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                               'Tool_Manipulation', 'Cutting', 'Cauterization',
                               'Suction', 'Looping', 'Suturing', 'Clipping', 'Staple',
                               'Ultrasound_Sensing')

        self.xml_data_dir = 'dataset/instruments18/seq_'
        self.word2vec_loc = 'dataset/surgicalscene_word2vec.hdf5'


class SurgicalSceneDataset(Dataset):
    '''
    Dataset class for the MTL Model
    Inputs: sequence set, data directory (root), image directory, mask directory, augmentation flag (istrain), dataset (dset), feature extractor chosen
    '''
    def __init__(self, seq_set, data_dir, img_dir, mask_dir, istrain, dset, dataconst, feature_extractor, reduce_size=False):

        self.data_size = 143
        self.dataconst = dataconst
        