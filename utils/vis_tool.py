import time

import random
import numpy as np
import matplotlib
import torch as t

matplotlib.use('Agg')
from matplotlib import pyplot as plot
from PIL import Image, ImageDraw, ImageFont


def vis_img(img, node_classes, bboxs,  det_action, data_const, score_thresh = 0.7):
    
    Drawer = ImageDraw.Draw(img)
    line_width = 3
    outline = '#FF0000'
    font = ImageFont.truetype(font='/usr/share/fonts/truetype/freefont/FreeMono.ttf', size=25)
    
    im_w,im_h = img.size
    node_num = len(node_classes)
    edge_num = len(det_action)
    tissue_num = len(np.where(node_classes == 1)[0])
    
    for node in range(node_num):
        
        r_color = random.choice(np.arange(256))
        g_color = random.choice(np.arange(256))
        b_color = random.choice(np.arange(256))
        
        text = data_const.instrument_classes[node_classes[node]]
        h, w = font.getsize(text)
        Drawer.rectangle(list(bboxs[node]), outline=outline, width=line_width)
        Drawer.text(xy=(bboxs[node][0], bboxs[node][1]-w-1), text=text, font=font, fill=(r_col