#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 13:56:18 2020

@author: thomas
"""

import cv2 as cv
import numpy as np
import array
from PIL import Image
#from psychopy import visual
from scipy import ndimage
import math
from natsort import natsorted
import glob
import sys
import os
from src.events.Events import (
    Events,
)
from src.events.tools.generation.pix2nvs import Pix2Eve
import shutil


def counterphase_grating(win, frequency=1 / 346, orientation=0, phase=0, contrast=1):
    grat_stim = visual.GratingStim(
        win=win, tex="sqr", units="pix", pos=(0.0, 0.0), size=500
    )
    grat_stim.sf = frequency
    grat_stim.ori = orientation
    grat_stim.phase = phase
    grat_stim.contrast = contrast
    grat_stim.draw()


def grating_generation(folder, display=False, time=0.2, framerate=1000, flash_period=0.1):
    """
    time # s
    framerate # fps
    flash_period # s
    """

    x = np.sin(np.linspace(-np.pi / 2, np.pi / 2, int(flash_period * framerate) // 2))
    flash = (np.hstack((x, x[::-1])) + 1) / 2
    phases = [0, 0.5]

    win = visual.Window(
        [346, 260],
        screen=0,
        monitor="testMonitor",
        fullscr=False,
        color=[0, 0, 0],
        units="pix",
    )
    switch = 0
    for i in range(int(time * framerate)):
        if i % int(flash_period * framerate) == 0:
            switch = not switch
        index = i % int(flash_period * framerate)
        contrast = flash[index]
        phase = phases[switch]

        counterphase_grating(win, 58 / 346, 0, phase, contrast)

        win.getMovieFrame(buffer="back")
        if display:
            win.flip()

    win.saveMovieFrames(fileName=folder)
    win.close()


def falling_leaves(time=10, framerate=1000, nb_circle_frame=4):
    img = np.full((260, 346, 3), 127, np.uint8)

    cnt = 0
    for frame in range(int(time * framerate)):
        for i in range(nb_circle_frame):
            center_x = np.random.randint(0, 346)
            center_y = np.random.randint(0, 260)
            intensity = np.random.randint(0, 255)
            size = np.random.randint(10, 40)
            cv.circle(
                img, (center_x, center_y), size, (intensity, intensity, intensity), 2
            )

        image = Image.fromarray(img)
        image.save("/home/alphat/Desktop/circles/img" + str(cnt) + ".png")
        cnt += 1


def moving_lines(folder, time=10, framerate=1000, speed=200, rotation=0, disparity=0, frame_start=0):
    cnt = frame_start
    positions = np.linspace(0, 550, 15, dtype=np.uint16)

    for frame in range(int(time * framerate)):
        img = np.full((460, 550, 3), 0, np.uint8)

        shift = int(frame * (speed / framerate))
        for i in positions:
            pos = (i + shift) % 550
            cv.line(img, (pos + disparity, 0), (pos + disparity, 460), (255, 255, 255), 4)

        img = ndimage.rotate(img, rotation, reshape=False, order=0)

        image = Image.fromarray(img[100:360, 100:446])
        image.save(folder + "img" + str(cnt) + ".png")
        cnt += 1


def moving_bars(folder, framerate=1000, speeds=None):
    if speeds is None:
        speeds = [400, 200, 100, 50]
    frame = 0
    y = np.linspace(0, 260, len(speeds) + 1, dtype=np.uint16)
    shift = 0

    while shift < 350:
        img = np.full((260, 346, 3), 0, np.uint8)

        for i, speed in enumerate(speeds):
            shift = int(frame * (speed / framerate))
            cv.line(img, (0 + shift, y[i]), (0 + shift, y[i + 1]), (255, 255, 255), 4)

        # img = ndimage.rotate(img, rotation, reshape=False, order=0)

        image = Image.fromarray(img)
        image.save(folder + "img" + str(frame) + ".png")
        frame += 1

def disparity_bars_2(folder, framerate=1000, speeds=None, disparities=None, rotation=None, y=None, z=None, thness = 4,color=True, frame_ref = 0):
    if disparities is None:
        disparities = [334, 340, 342, 344]
    if y is None:
        y = np.array([0,85,195,130])
    if z is None: 
        z = np.array([65,100,130,230])
    if speeds is None:
        speeds = [-400, -200, -100, -50]
        
    frame = 0
    x = np.array(disparities, dtype=np.uint16)
    #y = np.linspace(0, 260, len(disparities) + 1, dtype=np.uint16)
    shift = 0
    speeds = np.array(speeds)
    lim = np.where(speeds<0)
    ok = lim[0][speeds[lim].argmax()]
    speed_ok = frame * (speeds[ok]/ framerate)
    while x[ok] + speed_ok > -(231+5):#-(346+5):
        img = np.full((260, 346, 3), 128, np.uint8)        
        for i, speed in enumerate(speeds):
            shift = int(frame * (speed / framerate))
            if(type(thness)==list):
                if(color[i]):
                    cv.line(img, (x[i]+shift, y[i]), (x[i]+shift, z[i]), (255, 255, 255), thness[i])    
                else:
                    cv.line(img, (x[i]+shift, y[i]), (x[i]+shift, z[i]), (0, 0, 0), thness[i])    
            else:
                if(color):
                    cv.line(img, (x[i]+shift, y[i]), (x[i]+shift, z[i]), (255, 255, 255), thness) 
                else:
                    cv.line(img, (x[i]+shift, y[i]), (x[i]+shift, z[i]), (0, 0, 0), thness) 
        speed_ok = frame * (speeds[ok]/ framerate)
        if rotation is not None:
            img = ndimage.rotate(img, rotation, reshape=False, order=0)
        cv.rectangle(img, (0,128), (345,138), (128,128,128), -1)
        image = Image.fromarray(img)
        if(frame_ref!=0):
            image.save(folder + "img" + str(frame_ref) + ".png")
            frame_ref+=1
        else:
            image.save(folder + "img" + str(frame) + ".png")
        frame += 1
    return frame
        
def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return [float('inf'), float('inf')]
    return [np.int(np.round(x/z)), np.int(np.round(y/z))]

def from_pic_to_ev2(folder, angles):
    framerate = 1000
    time_gap = 1e6 * 1 / framerate
    pix2eve = Pix2Eve(
        time_gap=time_gap,
        log_threshold=0,
        map_threshold=0.4,
        min_threshold = 0.2, #0.2 # 0.02
        n_max=5,
        adapt_thresh_coef_shift=0.05,
        timestamp_noise=50
        )
    folders = folder + "/vanhateren_iml/"
    
    files_train_x = natsorted(glob.glob(os.path.join(folders, '*.iml')))
    folder_img = folder + "/events_ver/pics8/"
    os.chdir(folder_img)
    speed = 100
    directions = 2
    counter = 0
    max_files = 2000
    for j, file in enumerate(files_train_x):
        if(j+1>max_files):
            break
        #angle = angles[np.random.randint(0,len(angles))]
        for angle in angles:
            angle = angles[np.random.randint(0,len(angles))]
            # angle=0
            angle = np.radians(angle)
            for direction in range(directions):
                direction = np.random.randint(0,directions)
                # direction = 0
    #800 , frame number = 150
    #100 , frame number = 600
        #direction = np.random.randint(0,directions)
                if(direction==0):
                    speed = -abs(speed)
                else:
                    speed = abs(speed)
                frame=0
                os.chdir(folder_img)
                folder_file = folder_img + str(counter) + "/"
                if(not(os.path.isdir(folder_file))):
                    os.mkdir(str(counter))
                os.chdir(folder_file)
                with open(file, 'rb') as handle:
                    s = handle.read()
                    arr = array.array('H', s)
                    arr.byteswap()
                    img = np.array(arr, dtype='uint16').reshape(1024, 1536)
                img = cv.resize(img, (400, 267), interpolation = cv.INTER_AREA)
                img = img[3:263, 27:373]
                img = cv.resize(img, (512, 512), interpolation = cv.INTER_AREA)
                img = img / np.max(img)
                img = img * 255
                while(frame < 800):
                    shift = int(frame * (speed / framerate))
                    M = np.float32([[1,0,shift*np.cos(angle)],[0,1,shift*np.sin(angle)]])
                    # image = cv.warpAffine(img, M,(346,260))
                    image = cv.warpAffine(img, M,(512,512))
                    image = Image.fromarray(image.astype(np.uint16))
                    image = image.convert("RGB")
                    image.save(folder_file + "/img" + str(frame) + ".png")
                    frame+=1
                folder_npz = folder + "/events_ver/npz8/"
                print(folder_file)
                try:
                    ev=pix2eve.run(folder_file+"/")
                    dir_ = folder_npz + str(counter)+".npz"
                    ts=np.int64(ev[0:len(ev),0])
                    x=np.int16(ev[0:len(ev),1])
                    y=np.int16(ev[0:len(ev),2])
                    p=np.int8(ev[0:len(ev),3])
                    c=np.full(len(ev), False)
                    np.savez(dir_,ts, x, y, p,c)
                    
                    #Final event file
                    events = Events(dir_)    
                    events.sort_events()
                    events.translate(-128, -128)
                    folder_ev = folder + "/events_ver/events5_newsize/"
                    os.chdir(folder_ev)
                    dir_ = folder_ev + str(counter)+".npz"
                    events.save_as_file(dir_)
                    counter+=1
                except:
                    print("Skipped because no events")
                shutil.rmtree(folder_file)
                break
            break
    #shutil.rmtree(folder_npz)
        
def surround_moving_bars_ag(folder, start_x, start_y, end_x, end_y, index_cell, n_cells, rotation_main, rotation_border, jitter, cst = 25, color = True, cell_dimension_x = 10, cell_dimension_y = 10, overlap = 3, framerate=1000, thness_main=1, thness_borders=1, frame_ref=0, space = 3, sped = 6000, surr = 10):
    frame = 0
    center_x = 183 #173 #183
    center_y = 133 #129 #133
    #rotation_main = np.radians(rotation_main)
    #rotation_border = np.radians(rotation_border)
    x = start_x
    y = start_y
    x2 = start_x
    y2 = start_y
    z = 1
    #getting the coordinates of the main cell with the correct rotation
    for i in range(1,index_cell+1):
        if(i%9==0 and i!=0):
            x = start_x
            y += cell_dimension_y - overlap
        else:
            x += cell_dimension_x - overlap
    for i in range(1,40+1):
        if(i%9==0 and i!=0):
            x2 = start_x
            y2 += cell_dimension_y - overlap
        else:
            x2 += cell_dimension_x - overlap
    #getting the coordinates of the borders of the main cell    
    x_border_left = x - 1
    x_border_right = x + cell_dimension_x 
    y_border_up = y - 1
    y_border_bottom = y + cell_dimension_y
    jitter = 1
    print(x)
    print(y)
    mem_x = x - surr #start_x
    mem_y = y - surr #start_y
    mem_end_x = x + cell_dimension_x + surr #end_x
    mem_end_y = y + cell_dimension_x + surr #end_y
    start_x = 0
    start_y = 0
    end_x = 345
    end_y = 259
    space = 0
    cst = 0
    rotation_main2 = rotation_main
    #print(rotation_main2)
    rotation_border2 = rotation_border
    #print(rotation_border2)
    rotation_main = 0
    rotation_border = 0
    rot_matrix_main = np.array([ [np.cos(rotation_main), np.sin(rotation_main), 
                         (1-np.cos(rotation_main))*center_x - np.sin(rotation_main)*center_y], 
                       [-np.sin(rotation_main), np.cos(rotation_main), 
                        np.sin(rotation_main)*center_x+(1-np.cos(rotation_main))*center_y] ])
    #rotation_border2 = rotation_border
    rot_matrix_border = np.array([ [np.cos(rotation_border), np.sin(rotation_border), 
                         (1-np.cos(rotation_border))*center_x - np.sin(rotation_border)*center_y], 
                       [-np.sin(rotation_border), np.cos(rotation_border), 
                        np.sin(rotation_border)*center_x+(1-np.cos(rotation_border))*center_y] ])
    #print(rot_matrix_border)

    coords_x_main = []
    coords_y_main = []
    
    jittered_x_main = []
    jittered_y_main = []
    
    jittered_x_main_neg = []
    jittered_y_main_neg = []
    #if(rotation_main != rotation_border):
    for i in range(-500, 500):
        not_rotated_left = np.array([start_x - cst + i*(thness_main+space), start_y - cst, z])
        not_rotated_right = np.array([start_x - cst + i*(thness_main+space), end_y + cst, z])
        
        rotated_left = np.matmul(rot_matrix_main, not_rotated_left)
        rotated_right = np.matmul(rot_matrix_main, not_rotated_right)
        
        a1 = get_intersect(rotated_left, rotated_right, (x-cell_dimension_x*(n_cells-1), y-cell_dimension_y*(n_cells-1)), (x+cell_dimension_x*n_cells,y-cell_dimension_y*(n_cells-1)))
        a2 = get_intersect(rotated_left, rotated_right, (x-cell_dimension_x*(n_cells-1), y-cell_dimension_y*(n_cells-1)), (x-cell_dimension_x*(n_cells-1),y+cell_dimension_y*n_cells))
        a3 = get_intersect(rotated_left, rotated_right, (x-cell_dimension_x*(n_cells-1), y+cell_dimension_y*n_cells), (x+cell_dimension_x*n_cells, y+cell_dimension_y*n_cells))
        a4 = get_intersect(rotated_left, rotated_right, (x+cell_dimension_x*n_cells, y-cell_dimension_y*(n_cells-1)), (x+cell_dimension_x*n_cells,y+cell_dimension_y*n_cells))
        
        tab = [a1, a2, a3, a4]
        keep = []
        for i in range(len(tab)):
            if(not math.isinf(tab[i][0])):
                if(tab[i][0]>=x+cell_dimension_x*n_cells and tab[i][0]<=x + cell_dimension_x*n_cells and
                   tab[i][1]>=y-cell_dimension_y*(n_cells-1) and tab[i][1]<=y+cell_dimension_y*n_cells):
                    keep.append(tab[i])
        """rotated_left_jitter1 = np.matmul(rot_matrix_main, not_rotated_left_jitter1)
        rotated_right_jitter1 = np.matmul(rot_matrix_main, not_rotated_right_jitter1)"""
        to_add = np.array([1, 0])
        
        if(len(keep)==2): 
            coords_x_main.append(int(keep[0][0]))
            coords_x_main.append(int(keep[1][0]))
            coords_y_main.append(int(keep[0][1]))
            coords_y_main.append(int(keep[1][1]))
            
                
    coords_x_border = []
    coords_y_border = []
    
    for i in range(-500,500):
        not_rotated_left = np.array([start_x - cst + i*(thness_borders+space), start_y - cst, z])
        not_rotated_right = np.array([start_x - cst + i*(thness_borders+space), end_y + cst, z])
        
        rotated_left = np.matmul(rot_matrix_border, not_rotated_left)
        rotated_right = np.matmul(rot_matrix_border, not_rotated_right)
        a1 = get_intersect(rotated_left, rotated_right, (start_x, start_y), (end_x, start_y))
        a2 = get_intersect(rotated_left, rotated_right, (start_x, start_y), (start_x, end_y))
        a3 = get_intersect(rotated_left, rotated_right, (start_x, end_y), (end_x, end_y))
        a4 = get_intersect(rotated_left, rotated_right, (end_x, start_y), (end_x, end_y))
        
        tab = [a1, a2, a3, a4]
        keep = []
        for i in range(len(tab)):
            if(not math.isinf(tab[i][0])):
                #if(tab[i][0]>=start_x and tab[i][0]<=end_x and
                #   tab[i][1]>=start_y and tab[i][1]<=end_y):
                keep.append(tab[i])
        
        if(len(keep)==2):
            
            coords_x_border.append(int(np.floor(keep[0][0])))
            coords_y_border.append(int(np.floor(keep[0][1])))
            coords_x_border.append(int(np.floor(keep[1][0])))
            coords_y_border.append(int(np.floor(keep[1][1])))
    #print(rotation_main2)
    #print(rotation_border2)
    init = np.full((260, 346, 3), 128, np.uint8) 
    speed = sped
    while(frame < 2000):
        #intensity = 128 + abs(127 * np.sin(2*np.pi*fqcy*frame))
        #print(intensity)
        img = np.full((260, 346, 3), 128, np.uint8) 
        img2 = np.full((260, 346, 3), 128, np.uint8)
        length_2 = len(coords_x_border)
        for i in range(0, length_2, 2):
            shift = int(frame * (speed / framerate))
            if(i%4==0):
                cv.line(img, (coords_x_border[i]+shift, coords_y_border[i]), (coords_x_border[i+1]+shift, coords_y_border[i+1]), (255, 255, 255), thness_borders, cv.LINE_8)
                cv.line(img2, (coords_x_border[i]+shift, coords_y_border[i]), (coords_x_border[i+1]+shift, coords_y_border[i+1]), (255, 255, 255), thness_borders, cv.LINE_8)
            else:
                cv.line(img, (coords_x_border[i]+shift, coords_y_border[i]), (coords_x_border[i+1]+shift, coords_y_border[i+1]), (0, 0, 0), thness_borders, cv.LINE_8)
                cv.line(img2, (coords_x_border[i]+shift, coords_y_border[i]), (coords_x_border[i+1]+shift, coords_y_border[i+1]), (0, 0, 0), thness_borders, cv.LINE_8)
        
        img = ndimage.rotate(img, rotation_border2, reshape=False, order=0, cval=128)
        img2 = ndimage.rotate(img2, rotation_main2, reshape=False, order=0)
        img3 = img
        #print(img.shape)
        #print(img[y:y+cell_dimension_y,x:x+cell_dimension_x].shape)
        #print(img[y:y+cell_dimension_y,x+cell_dimension_x].shape)
        #print(img2[y:y+cell_dimension_y, x:x+cell_dimension_x].shape)
        
        #img[y+overlap:y+cell_dimension_y-overlap,x+overlap:x+cell_dimension_x-overlap] = img2[y+overlap:y+cell_dimension_y-overlap, x+overlap:x+cell_dimension_x-overlap]
        img[y-cell_dimension_y*(n_cells-1):y+cell_dimension_y*n_cells,x-cell_dimension_x*(n_cells-1):x+cell_dimension_x*n_cells] = img2[y-cell_dimension_y*(n_cells-1):y+cell_dimension_y*n_cells, x-cell_dimension_x*(n_cells-1):x+cell_dimension_x*n_cells]
        img[y2-cell_dimension_y*(n_cells-1):y2+cell_dimension_y*n_cells,x2-cell_dimension_x*(n_cells-1):x2+cell_dimension_x*n_cells] = img3[y2-cell_dimension_y*(n_cells-1):y2+cell_dimension_y*n_cells, x2-cell_dimension_x*(n_cells-1):x2+cell_dimension_x*n_cells]

        """if(rotation_main!=rotation_border):
            cv.rectangle(img,(x,y),(x+cell_dimension_x,y+cell_dimension_y),(128, 128, 128),-1)
            length = len(coords_x_main)
            for i in range(0,length,2):
                if(i%4==0):
                    cv.line(img, (coords_x_main[i], coords_y_main[i]), (coords_x_main[i+1], coords_y_main[i+1]), (255, 255, 255), thness_main, cv.LINE_8)
                else:
                    cv.line(img, (coords_x_main[i], coords_y_main[i]), (coords_x_main[i+1], coords_y_main[i+1]), (0, 0, 0), thness_main, cv.LINE_8)"""
        img[:,start_x:mem_x] = init[:,start_x:mem_x]
        img[:,mem_end_x:end_x+1] = init[:,mem_end_x:end_x+1]
        img[start_y:mem_y, :] = init[start_y:mem_y, :]
        img[mem_end_y:end_y+1, :] = init[mem_end_y:end_y+1, :]

        #img[mem_end_y:end_y+1,start_x:mem_x] = init[mem_end_y:end_y+1,start_x:mem_x]
        #img[mem_end_x:end_x+1] = init[mem_end_x:end_x+1]
        #img[mem_end_y:end_y+1] = init[mem_end_y:end_y+1]

        #ctrast = abs(np.sin(2*np.pi*fqcy*frame))
        #img = (img * ctrast + ( 1 - ctrast )*128)
        image = Image.fromarray(img.astype(np.uint8))
        #image= image.convert("RGB")
        #print(img.shape)
        image.save(folder + "img" + str(frame) + ".png")
        frame+=1
        
def sinusoidalGrating(folder, K, theta, phi, omega, A = 255, start_x = 150, end_x = 215, start_y = 100, end_y = 165, num = 66, dur = 2001):
    A = 127.5 #1 #255
    frame = 0
    while(frame < dur):
        img = np.full((260, 346, 3), 128, np.uint8) 
        # arrx = np.linspace(150, 215, 66, dtype=np.int16)
        # arry = np.linspace(100, 165, 66, dtype=np.int16)
        
        # arrx = np.linspace(178, 187, 10, dtype=np.int16)
        # arry = np.linspace(128, 137, 10, dtype=np.int16)
        
        arrx = np.linspace(start_x, end_x, num, dtype=np.int16)
        arry = np.linspace(start_y, end_y, num, dtype=np.int16)
        # print(frame / 1e3)
        for x in arrx:
            for y in arry:
                v = np.floor(A * np.cos(K * x * np.cos(theta) + K * abs(y-259) * np.sin(theta) - phi) * np.cos(omega * frame / 1e3) + 127.5)
                # print(K * x * np.cos(theta) + K * y * np.sin(theta) - phi)
                # if(v < 0):
                #     img[y][x] = 0
                # elif(v>0):
                #     img[y][x] = 255
                # else:
                #     img[y][x] = 128
                # print(v)
                # if(v==128):
                #     cos_ = np.cos(K * x * np.cos(theta) + K * abs(y-259) * np.sin(theta) - phi)
                #     other_cos = np.cos(omega * frame / 1e3)
                #     print("x = {}, y = {}, inside cos = {}, other cos = {}".format(x,y,cos_, other_cos))
                
                img[y][x] = v # A * np.cos(K * x * np.cos(theta) + K * y * np.sin(theta) - phi) * np.cos(omega * frame / 1e3) 
                # print(A * np.cos(K * x * np.cos(theta) + K * y * np.sin(theta) - phi) * np.cos(omega * frame / 1e3))
            once = False
        image = Image.fromarray(img.astype(np.uint8))
        if(frame>=125):
            image.save(folder + "img" + str(frame-125) + ".png")
        frame +=1