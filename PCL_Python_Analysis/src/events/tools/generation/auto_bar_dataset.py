#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 11:17:09 2022

@author: comsee
"""

import os
os.chdir("/home/comsee/Internship_Antony/neuvisys/neuvisys-analysis")

%load_ext autoreload
%autoreload 2

import shutil

import numpy as np
from scipy.stats import laplace
import matplotlib.pyplot as plt

from src.events.Events import (
    Events,
)
from src.events.tools.generation.pix2nvs import Pix2Eve
from src.events.tools.generation.stimuli_gen import (
    moving_lines,
    disparity_bars,
    disparity_bars_2
)
from src.events.tools.modification.event_modification import (
    rectify_events,
    concatenate_npz,
)
from src.frames.frame_analysis import (
    load_frames,
    rectify_frames,
    write_frames,
    stereo_matching,
)
from src.spiking_network.network.network_params import (
    reinforcement_learning,
    inhibition_orientation,
    inhibition_disparity,
)
from src.spiking_network.network.neuvisys import (
    SpikingNetwork,
    delete_files,
    clean_network,
    shuffle_weights,
)
from src.spiking_network.planning.network_planner import (
    create_networks,
    random_params,
    launch_neuvisys_multi_pass,
    change_param,
    divide_visual_field,
)

os.chdir("/home/comsee/Internship_Antony/neuvisys/neuvisys-analysis/src")
home = "/home/comsee/"

# network_path = home + "neuvisys-dv/configuration/network_ref_vh/"
network_path = home + "Internship_Antony/neuvisys/neuvisys-analysis/configuration/network_dir/"

rotations = np.array([0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338])
disparities = np.array([-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])

framerate = 1000
time_gap = 1e6 * 1 / framerate
pix2eve = Pix2Eve(
    time_gap=time_gap,
    log_threshold=0,
    map_threshold=0.4,
    n_max=5,
    adapt_thresh_coef_shift=0.05,
    timestamp_noise=50
)

folder_pics = home + "Internship_Antony/neuvisys/Events/new_bars/pics/"
folder_npz = home + "Internship_Antony/neuvisys/Events/new_bars/npz/"
folder_ev = home + "Internship_Antony/neuvisys/Events/new_bars/npz/events/"
os.chdir(folder_pics)

x_val=[]
speeds_val=[]
y_val=[]
z_val=[]
heights=[]
rg_speeds=[25,500]
rg_y_start = [0,99]
rg_y_height = [1,105]
max_number_of_bars = 150

i = 100

for j in range(i):
    os.mkdir(str(j))
    if(j%2==0):
        value = np.random.randint(max_number_of_bars)+1
        for(k in range(value)):
            heights.append(np.random.randint(rg_y_height[0],rg_y_height[1]))
            y_val.append(np.random.randint(rg_y_start[0],rg_y_start[1]))
            while(y_val[-1]+height > rg_y_height[1]-1):
                y_val[-1]=np.random.randint(rg_y_start[0],rg_y_start[1])
            z_val.append(y_val[-1]+height)
            speeds_val.append(np.random.randint(rg_speeds[0],rg_speeds[1]))
            x_val.append(0)
        disparity_bars(folder_pics+str(j)+"/",framerate=framerate,speeds=speeds_val,disparities=x,y=y_val,z=z_val)
    else:
        disparity_bars_2(folder_pics+str(j)+"/",framerate=framerate,speeds=-speeds_val,disparities=x,y=y_val,z=z_val)
        x_val=[]
        speeds_val=[]
        y_val=[]
        z_val=[]
        heights=[]
    #Get npz event
    ev=pix2eve.run(folder_pics+str(j)+"/")
    dir_ = folder_npz+str(j)+".npz"
    ts=np.int64(ev[0:len(ev),0])
    x=np.int16(ev[0:len(ev),1])
    y=np.int16(ev[0:len(ev),2])
    p=np.int8(ev[0:len(ev),3])
    c=np.full(len(ev), False)
    np.savez(dir_,ts, x, y, p,c)
    #Final event file
    events = Events(dir_)    
    events.sort_events()
    events.save_as_file(folder_ev+str(j)+".npz")