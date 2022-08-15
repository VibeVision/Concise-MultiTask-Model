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
    font = ImageFont.truetype(font='/usr/share/fonts/truetype/f