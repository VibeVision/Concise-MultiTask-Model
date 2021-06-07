
'''
Project         : Global-Reasoned Multi-Task Surgical Scene Understanding
Lab             : MMLAB, National University of Singapore
contributors    : Lalithkumar Seenivasan, Sai Mitheran, Mobarakol Islam, Hongliang Ren
Note            : Code adopted and modified from Visual-Semantic Graph Attention Networks and Dual attention network for scene segmentation
                        Visual-Semantic Graph Network:
                        @article{liang2020visual,
                          title={Visual-Semantic Graph Attention Networks for Human-Object Interaction Detection},
                          author={Liang, Zhijun and Rojas, Juan and Liu, Junfa and Guan, Yisheng},
                          journal={arXiv preprint arXiv:2001.02302},
                          year={2020}
                        }
'''


import dgl
import math
import numpy as np

import torch
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

'''
Configurations of the network
    
    readout: G_ER_L_S = [1024+300+16+300+1024,  1024, 117]

    node_func: G_N_L_S = [1024+1024, 1024]
    node_lang_func: G_N_L_S2 = [300+300+300]
    
    edge_func : G_E_L_S = [1024*2+16, 1024]
    edge_lang_func: [300*2, 1024]
    
    attn: [1024, 1]
    attn_lang: [1024, 1]
'''
class CONFIGURATION(object):
    '''
    Configuration arguments: feature type, layer, bias, batch normalization, dropout, multi-attn
    
    readout           : fc_size, activation, bias, bn, droupout
    gnn_node          : fc_size, activation, bias, bn, droupout
    gnn_node_for_lang : fc_size, activation, bias, bn, droupout
    gnn_edge          : fc_size, activation, bias, bn, droupout
    gnn_edge_for_lang : fc_size, activation, bias, bn, droupout
    gnn_attn          : fc_size, activation, bias, bn, droupout
    gnn_attn_for_lang : fc_size, activation, bias, bn, droupout
    '''
    def __init__(self, layer=1, bias=True, bn=False, dropout=0.2, multi_attn=False, global_feat = 0):
        
        # if multi_attn:
        if True:
            if layer==1:
                feature_size = 512
                additional_sf = global_feat
                # readout
                self.G_ER_L_S = [feature_size+300+16+additional_sf+300+feature_size, feature_size, 13]
                self.G_ER_A   = ['ReLU', 'Identity']
                self.G_ER_B   = bias    #true
                self.G_ER_BN  = bn      #false
                self.G_ER_D   = dropout #0.3
                # self.G_ER_GRU = feature_size

                # # gnn node function
                self.G_N_L_S = [feature_size+feature_size, feature_size]
                self.G_N_A   = ['ReLU']
                self.G_N_B   = bias #true
                self.G_N_BN  = bn      #false
                self.G_N_D   = dropout #0.3
                # self.G_N_GRU = feature_size

                # # gnn node function for language
                self.G_N_L_S2 = [300+300, 300]
                self.G_N_A2   = ['ReLU']
                self.G_N_B2   = bias    #true
                self.G_N_BN2  = bn      #false
                self.G_N_D2   = dropout #0.3
                # self.G_N_GRU2 = feature_size

                # gnn edge function1
                self.G_E_L_S           = [feature_size*2+16+additional_sf, feature_size]
                self.G_E_A             = ['ReLU']
                self.G_E_B             = bias     # true
                self.G_E_BN            = bn       # false
                self.G_E_D             = dropout  # 0.3
                # self.G_E_c_kernel_size = 3


                # gnn edge function2 for language
                self.G_E_L_S2 = [300*2, feature_size]
                self.G_E_A2   = ['ReLU']
                self.G_E_B2   = bias     #true
                self.G_E_BN2  = bn       #false
                self.G_E_D2   = dropout  #0.3

                # gnn attention mechanism
                self.G_A_L_S = [feature_size, 1]
                self.G_A_A   = ['LeakyReLU']
                self.G_A_B   = bias     #true
                self.G_A_BN  = bn       #false
                self.G_A_D   = dropout  #0.3

                # gnn attention mechanism2 for language
                self.G_A_L_S2 = [feature_size, 1]
                self.G_A_A2   = ['LeakyReLU']
                self.G_A_B2   = bias    #true
                self.G_A_BN2  = bn      #false
                self.G_A_D2   = dropout #0.3
                    
    def save_config(self):
        model_config = {'graph_head':{}, 'graph_node':{}, 'graph_edge':{}, 'graph_attn':{}}
        CONFIG=self.__dict__
        for k, v in CONFIG.items():
            if 'G_H' in k:
                model_config['graph_head'][k]=v
            elif 'G_N' in k:
                model_config['graph_node'][k]=v
            elif 'G_E' in k:
                model_config['graph_edge'][k]=v
            elif 'G_A' in k:
                model_config['graph_attn'][k]=v
            else:
                model_config[k]=v
        
        return model_config


class Identity(nn.Module):
    '''
    Identity class activation layer
    f(x) = x
    '''
    def __init__(self):
        super(Identity,self).__init__()

    def forward(self, x):
        return x

def get_activation(name):
    '''
    get_activation function
    argument: Activation name (eg. ReLU, Identity, Tanh, Sigmoid, LeakyReLU)
    '''
    if name=='ReLU': return nn.ReLU(inplace=True)
    elif name=='Identity': return Identity()
    elif name=='Tanh': return nn.Tanh()
    elif name=='Sigmoid': return nn.Sigmoid()
    elif name=='LeakyReLU': return nn.LeakyReLU(0.2,inplace=True)
    else: assert(False), 'Not Implemented'


class MLP(nn.Module):
    '''
    Args:
        layer_sizes: a list, [1024,1024,...]
        activation: a list, ['ReLU', 'Tanh',...]
        bias : bool
        use_bn: bool
        drop_prob: default is None, use drop out layer or not
    '''
    def __init__(self, layer_sizes, activation, bias=True, use_bn=False, drop_prob=None):
        super(MLP, self).__init__()
        self.bn = use_bn
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1], bias=bias)
            activate = get_activation(activation[i])
            block = nn.Sequential(OrderedDict([(f'L{i}', layer), ]))
            
            # !NOTE:# Actually, it is inappropriate to use batch-normalization here
            if use_bn:                                  
                bn = nn.BatchNorm1d(layer_sizes[i+1])
                block.add_module(f'B{i}', bn)
            
            # batch normalization is put before activation function 
            block.add_module(f'A{i}', activate)

            # dropout probablility
            if drop_prob:
                block.add_module(f'D{i}', nn.Dropout(drop_prob))
            
            self.layers.append(block)
    
    def forward(self, x):
        for layer in self.layers:
            # !NOTE: Sometime the shape of x will be [1,N], and we cannot use batch-normalization in that situation
            if self.bn and x.shape[0]==1:
                x = layer[0](x)
                x = layer[:-1](x)
            else:
                x = layer(x)
        return x


class H_H_EdgeApplyModule(nn.Module): #Human to Human edge
    '''
        init    : config, multi_attn 
        forward : edge
    '''
    def __init__(self, CONFIG, multi_attn=False):
        super(H_H_EdgeApplyModule, self).__init__()        
        self.edge_fc = MLP(CONFIG.G_E_L_S, CONFIG.G_E_A, CONFIG.G_E_B, CONFIG.G_E_BN, CONFIG.G_E_D)
        self.edge_fc_lang = MLP(CONFIG.G_E_L_S2, CONFIG.G_E_A2, CONFIG.G_E_B2, CONFIG.G_E_BN2, CONFIG.G_E_D2)
    
    def forward(self, edge):
        feat = torch.cat([edge.src['n_f'], edge.data['s_f'], edge.dst['n_f']], dim=1)
        feat_lang = torch.cat([edge.src['word2vec'], edge.dst['word2vec']], dim=1)
        e_feat = self.edge_fc(feat)
        e_feat_lang = self.edge_fc_lang(feat_lang)
  
        return {'e_f': e_feat, 'e_f_lang': e_feat_lang}



class H_NodeApplyModule(nn.Module): #human node
    '''
        init    : config
        forward : node
    '''
    def __init__(self, CONFIG):
        super(H_NodeApplyModule, self).__init__()
        self.node_fc = MLP(CONFIG.G_N_L_S, CONFIG.G_N_A, CONFIG.G_N_B, CONFIG.G_N_BN, CONFIG.G_N_D)
        self.node_fc_lang = MLP(CONFIG.G_N_L_S2, CONFIG.G_N_A2, CONFIG.G_N_B2, CONFIG.G_N_BN2, CONFIG.G_N_D2)
    
    def forward(self, node):
        feat = torch.cat([node.data['n_f'], node.data['z_f']], dim=1)
        feat_lang = torch.cat([node.data['word2vec'], node.data['z_f_lang']], dim=1)
        n_feat = self.node_fc(feat)
        n_feat_lang = self.node_fc_lang(feat_lang)

        return {'new_n_f': n_feat, 'new_n_f_lang': n_feat_lang}


class E_AttentionModule1(nn.Module): #edge attention
    '''
        init    : config
        forward : edge
    '''
    def __init__(self, CONFIG):
        super(E_AttentionModule1, self).__init__()
        self.attn_fc = MLP(CONFIG.G_A_L_S, CONFIG.G_A_A, CONFIG.G_A_B, CONFIG.G_A_BN, CONFIG.G_A_D)
        self.attn_fc_lang = MLP(CONFIG.G_A_L_S2, CONFIG.G_A_A2, CONFIG.G_A_B2, CONFIG.G_A_BN2, CONFIG.G_A_D2)

    def forward(self, edge):
        a_feat = self.attn_fc(edge.data['e_f'])
        a_feat_lang = self.attn_fc_lang(edge.data['e_f_lang'])
        return {'a_feat': a_feat, 'a_feat_lang': a_feat_lang}