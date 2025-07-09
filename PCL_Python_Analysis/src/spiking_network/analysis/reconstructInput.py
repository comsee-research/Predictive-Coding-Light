#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:48:33 2024

@author: comsee
"""

import numpy as np
from PIL import Image

def getReconstructedInput(bins,size,start,zone,overlap,rfs,weights,path):
    actbin = size[0] * size[1] * size[2]
    map_ = []
    mapz = []
    startx = start[1]
    for k in range(0, size[1]): # x
        starty = start[0]
        for j in range(0, size[0]): # y
            for l in range(0, size[2]): # z
                # map_.append(k * size[0] * size[2] + j * size[2] + l)
                map_.append([starty, startx])
                mapz.append(l)
            starty += zone - overlap
        startx += zone - overlap
        
    imgs = []
    for num, i in enumerate(range(actbin, actbin * 6)):
        img = np.full((260, 346, 3), 0, np.uint8) 
        for pos, b in enumerate(bins[i - actbin: i]):
            if(b!=0):
                img[map_[pos][0]:map_[pos][0]+zone,map_[pos][1]:map_[pos][1]+zone,1] += weights[l,:,1]
                img[map_[pos][0]:map_[pos][0]+zone,map_[pos][1]:map_[pos][1]+zone,0] += weights[l,:,0]
            # max_weight = np.max(max_weight, np.max(weights[l]))
        img = np.array(255 * ( (img) / np.max(img)), dtype=np.uint8)
        Image.fromarray(img).save(path + str(num) + ".png")
        imgs.append(img)
    return imgs
        