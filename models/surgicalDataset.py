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
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.is_train = istrain
        self.feature_extractor = feature_extractor
        self.reduce_size = reduce_size

        # Images and masks are resized to (320, 400)
        self.resizer = transforms.Compose([transforms.Resize((320, 400))]) 

        self.xml_dir_list = []
        self.dset = []

        for domain in range(len(seq_set)):
            domain_dir_list = []
            for i in seq_set[domain]:
                xml_dir_temp = data_dir[domain] + str(i) + '/xml/'
                domain_dir_list = domain_dir_list + glob(xml_dir_temp + '/*.xml')
            if self.reduce_size:
                indices = np.random.permutation(len(domain_dir_list))
                domain_dir_list = [domain_dir_list[j] for j in indices[0:self.data_size]]
            for file in domain_dir_list:
                self.xml_dir_list.append(file)
                self.dset.append(dset[domain])
        self.word2vec = h5py.File('dataset/surgicalscene_word2vec.hdf5', 'r')

    # Word2Vec function
    def _get_word2vec(self, node_ids, sgh=0):
        word2vec = np.empty((0, 300))
        for node_id in node_ids:
            if sgh == 1 and node_id == 0:
                vec = self.word2vec['tissue']
            else:
                vec = self.word2vec[self.dataconst.instrument_classes[node_id]]
            word2vec = np.vstack((word2vec, vec))
        return word2vec

    # Dataset length
    def __len__(self):
        return len(self.xml_dir_list)

    # Function to get images and masks 
    def __getitem__(self, idx):

        file_name = os.path.splitext(os.path.basename(self.xml_dir_list[idx]))[0]
        file_root = os.path.dirname(os.path.dirname(self.xml_dir_list[idx]))
        if len(self.img_dir) == 1:
            _img_loc = os.path.join(file_root+self.img_dir[0] + file_name + '.png')
            _mask_loc = os.path.join(file_root+self.mask_dir[0] + file_name + '.png')

        else:
            _img_loc = os.path.join( file_root+self.img_dir[self.dset[idx]] + file_name + '.png')
            _mask_loc = os.path.join( file_root+self.mask_dir[self.dset[idx]] + file_name + '.png')


        _img = Image.open(_img_loc).convert('RGB')
        _target = Image.open(_mask_loc)

        if self.is_train:
            isAugment = random.random() < 0.5
            if isAugment:
                isHflip = random.random() < 0.5
                if isHflip:
                    _img = _img.transpose(Image.FLIP_LEFT_RIGHT)
                    _target = _target.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    _img = _img.transpose(Image.FLIP_TOP_BOTTOM)
                    _target = _target.transpose(Image.FLIP_TOP_BOTTOM)

        _img = np.asarray(_img, np.float32) * 1.0 / 255
        _img = torch.from_numpy(np.array(_img).transpose(2, 0, 1)).float()
        _target = torch.from_numpy(np.array(_target)).long()

        frame_data = h5py.File(os.path.join( file_root+'/vsgat/'+self.feature_extractor+'/'+ file_name + '_features.hdf5'), 'r')
        
        data = {}

        data['img_name'] = frame_data['img_na