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

    test_seg_loss = 0.0
    total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
    
    
    for data in tqdm(validation_dataloader):
        seg_img = data['img']
        seg_masks = data['mask']
        img_loc = data['img_loc']
        node_num = data['node_num']
        roi_labels = data['roi_labels']
        det_boxes = data['det_boxes']
        edge_labels = data['edge_labels']
        spatial_feat = data['spatial_feat']
        word2vec = data['word2vec']
        
        spatial_feat, word2vec, edge_labels = spatial_feat.cuda(non_blocking=True), word2vec.cuda(non_blocking=True), edge_labels.cuda(non_blocking=True)
        seg_img, seg_masks = seg_img.cuda(non_blocking=True), seg_masks.cuda(non_blocking=True)

        with torch.no_grad():
            interaction, seg_outputs, _ = model(seg_img, img_loc, det_boxes, node_num, spatial_feat, word2vec, roi_labels, validation=True)

        scene_graph_logits_list.append(interaction)
        scene_graph_labels_list.append(edge_labels)

        # loss and accuracy
        scene_graph_loss = scene_graph_criterion(interaction, edge_labels.float())
        scene_graph_acc = np.sum(np.equal(np.argmax(interaction.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))
        correct, labeled, inter, union, t_loss = seg_eval_batch(seg_outputs, seg_masks)

        # accumulate scene graph loss and acc
        scene_graph_total_loss += scene_graph_loss.item() * edge_labels.shape[0]
        scene_graph_total_acc += scene_graph_acc
        scene_graph_edge_count += edge_labels.shape[0]

        total_correct += correct
        total_label += labeled
        total_inter += inter
        total_union += union
        test_seg_loss += t_loss.item()

    # graph evaluation
    scene_graph_total_acc = scene_graph_total_acc / scene_graph_edge_count
    scene_graph_total_loss = scene_graph_total_loss / len(validation_dataloader)
    scene_graph_logits_all = torch.cat(scene_graph_logits_list).cuda()
    scene_graph_labels_all = torch.cat(scene_graph_labels_list).cuda()
    scene_graph_logits_all = F.softmax(scene_graph_logits_all, dim=1)
    scene_graph_map_value, scene_graph_recall = calibration_metrics(scene_graph_logits_all, scene_graph_labels_all)

    # segmentation evaluation
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    class_values += IoU
    mIoU = IoU.mean()

    print('\n================= Evaluation ====================')
    print('Graph        :  acc: %0.4f  map: %0.4f recall: %0.4f  loss: %0.4f}' % (scene_graph_total_acc, scene_graph_map_value, scene_graph_recall, scene_graph_total_loss))
    print('Segmentation : Pacc: %0.4f mIoU: %0.4f   loss: %0.4f}' % (pixAcc, mIoU, test_seg_loss/len(validation_dataloader)))

    print('\n================= Class-wise IoU ====================')
    class_wise_IoU = []
    m_vals = []
    for idx, value in enumerate(class_values):
        class_name = index_to_label(idx)
        pair = [class_name, value]
        m_vals.append(value)
        class_wise_IoU.append(pair)

    print("Mean Value: ", np.mean(np.array(m_vals)), "\n")

    print(tabulate(class_wise_IoU,
          headers=['Class', 'IoU'], tablefmt='orgtbl'))

    return(scene_graph_total_acc, scene_graph_map_value, mIoU)


if __name__ == "__main__":

    '''
    Main function to set arguments
    '''

    '''
    To reproduce the results, set the model_type, ver, seg_mode and checkpoint_dir based on the table below
    TBR = To be released
    ============================================================================================================
    Paper_name              |   model_type     |        ver           |    seg_mode    |   checkpoint_dir
    ============================================================================================================
                                                        STL
    ------------------------|-----------------------------------------------------------------------------------
    VS-GAT                  |    'stl-sg'      |    'stl_sg'          |      None      |    'stl_sg'
    SEG                     |    'stl-s'       |    'stl_s_ng'        |      TBR       |    'stl_s_ng'
    SEG-GR                  |    'stl-s'       |    'stl_s'           |      None      |    'stl_s'
    SEG-MSGR                |    'stl-s'       |    'stl_s_v2gc'      |     'v2gc'     |    'stl_s_v2gc'
    SEG-MSLRGR              |    'stl-s'       |    'stl_s_v1'        |     'v1'       |    'stl_s_v1'
    ------------------------------------------------------------------------------------------------------------
                                                       SMTL
    ------------------------------------------------------------------------------------------------------------
    GR                      |    'amtl-t0'     |    'amtl_t0_s'       |      None      |    'amtl_t0_s'
    MSGR                    |    'amtl-t0'     |    'amtl_t0_sv2gc'   |      'v2gc'    |    'amtl_t0_sv2gc'
    MSLRGR                  |    'amtl-t0'     |    'amtl_t0_sv1'     |      'v1'      |    'amtl_t0_sv1'
    MSLRGR-GISFSG           |    'amtl-t3'     |    'amtl_t3pn_sv1'   |      'v1'      |    'amtl_t3pn_sv1'          
    ------------------------------------------