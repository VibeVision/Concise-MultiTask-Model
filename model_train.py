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
            loc = 'checkpoints/stl_s_v1/stl_s_v1/epoch_train/checkpoint_D168_epoch.pth'
        elif seg_mode == 'v2_gc':
            loc = 'checkpoints/stl_s_v2_gc/stl_s_v2_gc/epoch_train/checkpoint_D168_epoch.pth'
    elif model_type == 'amtl-t1':
        loc = 'checkpoints/stl_s/stl_s/epoch_train/checkpoint_D168_epoch.pth'
    elif model_type == 'amtl-t2':
        loc = 'checkpoints/stl_sg_wfe/stl_sg_wfe/epoch_train/checkpoint_D110_epoch.pth'
    return loc

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
    seg_model = get_gcnet(backbone='resnet18_model', pretrained=True)
    model = mtl_model(seg_model.pretrained, scene_graph, seg_model.gr_interaction, seg_model.gr_decoder, seg_mode = args.seg_mode)
    model.to(torch.device('cpu'))
    return model


def model_eval(args, model, validation_dataloader):
    '''
    Evaluate function for the MTL model (Segmentation and Scene Graph Performance)
    Inputs: args, model, val-dataloader

    '''

    model.eval()

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

        # Loss and accuracy
        scene_graph_loss = scene_graph_criterion(interaction, edge_labels.float())
        scene_graph_acc = np.sum(np.equal(np.argmax(interaction.cpu().data.numpy(), axis=-1), np.argmax(edge_labels.cpu().data.numpy(), axis=-1)))
        correct, labeled, inter, union, t_loss = seg_eval_batch(seg_outputs, seg_masks)

        # Accumulate scene graph loss and acc
        scene_graph_total_loss += scene_graph_loss.item() * edge_labels.shape[0]
        scene_graph_total_acc += scene_graph_acc
        scene_graph_edge_count += edge_labels.shape[0]

        total_correct += correct
        total_label += labeled
        total_inter += inter
        total_union += union
        test_seg_loss += t_loss.item()

    # Graph evaluation
    scene_graph_total_acc = scene_graph_total_acc / scene_graph_edge_count
    scene_graph_total_loss = scene_graph_total_loss / len(validation_dataloader)
    scene_graph_logits_all = torch.cat(scene_graph_logits_list).cuda()
    scene_graph_labels_all = torch.cat(scene_graph_labels_list).cuda()
    scene_graph_logits_all = F.softmax(scene_graph_logits_all, dim=1)
    scene_graph_map_value, scene_graph_recall = calibration_metrics(scene_graph_logits_all, scene_graph_labels_all)

    # Segmentation evaluation
    pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    mIoU = IoU.mean()

    print('================= Evaluation ====================')
    print('Graph        :  acc: %0.4f  map: %0.4f recall: %0.4f  loss: %0.4f}' % (scene_graph_total_acc, scene_graph_map_value, scene_graph_recall, scene_graph_total_loss))
    print('Segmentation : Pacc: %0.4f mIoU: %0.4f   loss: %0.4f}' % (pixAcc, mIoU, test_seg_loss/len(validation_dataloader)))
    return(scene_graph_total_acc, scene_graph_map_value, mIoU)


def train_model(gpu, args):
    '''
    Train function for the MTL model
    Inputs:  number of gpus per node, args

    '''
    # Store best value and epoch number
    best_value = [0.0, 0.0, 0.0]
    best_epoch = [0, 0, 0]

    # Decaying learning rate
    decay_lr = args.lr

    # This is placed above the dist.init process, because of the feature_extraction model.
    model = build_model(args)

    # Load pre-trained weights
    if args.model == 'amtl-t0' or args.model == 'amtl-t3' or args.model == 'amtl-t0-ft' or args.model == 'amtl-t1' or args.model == 'amtl-t2':
        print('Loading pre-trained weights for Sequential Optimisation')
        pretrained_model = torch.load(get_checkpoint_loc(args.model, args.seg_mode))
        pretrained_dict = pretrained_model['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        model_dict.update(pretrained_dict) 
        model.load_state_dict(model_dict)

    # Set training flag for submodules based on train model.
    model.set_train_test(args.model)


    if args.KD:
        teacher_model = build_model(args, load_pretrained=False)
        # Load pre-trained stl_mtl_model
        print('Preparing teacher model')
        pretrained_model = torch.load('/media/mobarak/data/lalith/mtl_scene_understanding_and_segmentation/checkpoints/stl_s_v1/stl_s_v1/epoch_train/checkpoint_D168_epoch.pth')
        pretrained_dict = pretrained_model['state_dict']
        model_dict = teacher_model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in model_dict) and (model_dict[k].shape == pretrained_dict[k].shape)}
        model_dict.update(pretrained_dict) 
        teacher_model.load_state_dict(model_dict)
        if args.model == 'mtl-t3':
            teacher_model.set_train_test('mtl-t3')
            teacher_model.model_type3_insert()
            teacher_model.cuda()
        else:
            teacher_model.set_train_test('stl-s')
        teacher_model.cuda()
        teacher_model.eval()

    # Insert nn layers based on type.
    if args.model == 'amtl-t1' or args.model == 'mtl-t1':
        model.model_type1_insert()
    elif args.model == 'amtl-t2' or args.model == 'mtl-t2':
        model.model_type2_insert()
    elif args.model == 'amtl-t3' or args.model == 'mtl-t3':
        model.model_type3_insert()

    # Priority rank given to node 0 -> current pc, if more nodes -> multiple PCs
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port #8892
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)

    # Set cuda
    torch.cuda.set_device(gpu)

    # Wrap the model with ddp
    model.cuda()
    model = DDP(model, device_ids=[gpu], find_unused_parameters=True)#, find_unused_parameters=True)

    # Define loss function (criterion) and optimizer
    seg_criterion = SegmentationLosses(se_loss=False, aux=False, nclass=8, se_weight=0.2, aux_weight=0.2).cuda(gpu)
    graph_scene_criterion = nn.MultiLabelSoftMarginLoss().cuda(gpu)
    
    # train and test dataloader
    train_seq = [[2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]]
    val_seq = [[1, 5, 16]]
    data_dir = ['datasets/instruments18/seq_']
    img_dir = ['/left_frames/']
    mask_dir = ['/annotations/']
    dset = [0]
    data_const = SurgicalSceneConstants()

    seq = {'train_seq': train_seq, 'val_seq': val_seq, 'data_dir': data_dir, 'img_dir': img_dir, 'dset': dset, 'mask_dir': mask_dir}

    # Val_dataset only set in 1 GPU
    val_dataset = SurgicalSceneDataset(seq_set=seq['val_seq'], dset=seq['dset'], data_dir=seq['data_dir'], \
                                       img_dir=seq['img_dir'], mask_dir=seq['mask_dir'], istrain=False, dataconst=data_const, \
                                       feature_extractor=args.feature_extractor, reduce_size=False)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # Train_dataset distributed to 2 GPU
    train_dataset = SurgicalSceneDataset(seq_set=seq['train_seq'], data_dir=seq['data_dir'],
                                         img_dir=seq['img_dir'], mask_dir=seq['mask_dir'], dset=seq['dset'], istrain=True, dataconst=data_const,
                                         feature_extractor=args.feature_extractor, reduce_size=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, pin_memory=True, sampler=train_sampler)

    # Evaluate the model before start of training
    if gpu == 0:
        if args.KD:
            print("=================== Teacher Model=========================")
            eval_sc_acc, eval_sc_map, eval_seg_miou = model_eval(args, teacher_model, val_dataloader)
            print("=================== Student Model=========================")
        eval_sc_acc, eval_sc_map, eval_seg_miou = model_eval(args, model, val_dataloader)
        print("PT SC ACC: [value: {:0.4f}] PT SC mAP: [value: {:0.4f}] PT Seg mIoU: [value: {:0.4f}]".format(eval_sc_acc, eval_sc_map, eval_seg_miou))

    for epoch_count in range(args.epoch):

        start_time = time.time()

        # Set model / submodules in train mode
        model.train()
        if args.model == 'stl-sg' or args.model == 'amtl-t0' or args.model == 'amtl-t3':
            model.module.feature_encoder.eval()
            model.module.gcn_unit.eval()
            model.module.seg_decoder.eval()
        elif args.model == 'stl-sg-wfe':
            model.module.gcn_unit.eval()
            model.module.seg_decoder.eval()
        elif args.model == 'stl-s':
            model.module.scene_graph.eval()

        train_seg_loss = 0.0
