#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 11:45:10 2025

@author: comsee
"""

import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir("/home/comsee/PhD_Antony/PCL_Python/neuvisys_analysis/src/events/")
from Events import (
    Events
)

path_save_basic = "/home/comsee/PhD_Antony/data_basic_PCL_NatComms/test/"
npzTemp = "/home/comsee/PhD_Antony/data_basic_PCL_NatComms/temp.npz"
n_ = 35

for i in range(n_):
    ts = []
    x = []
    y =[]
    p = []
    c = []
    
    ts1 = []
    ts2 = []
    ts3 = []
    
    x1 = []
    x2 = []
    x3 = []
    
    y1 = []
    y2 = []
    y3 = []
    
    t = 0
    t1 = 0
    t2 = 0
    t3 = 0
    
    rate = 20  # spikes per second
    T = 1    # total duration in seconds
    while t < T:
        interval = -np.log(np.random.rand()) / rate
        interval1 = -np.log(np.random.rand()) / rate
        interval2 = -np.log(np.random.rand()) / rate
        interval3 = -np.log(np.random.rand()) / rate
        t += interval
        t1 += interval1
        t2 += interval2
        t3 += interval3
        if t < T:
            ts.append(np.int32(t * 1e6))
            x.append(0)
            y.append(0)
            p.append(0)
            c.append(False)
        if t1 < T:
            ts1.append(np.int32(t1 * 1e6))
        if t2 < T:
            ts2.append(np.int32(t2 * 1e6))
        if t3 < T:
            ts3.append(np.int32(t3 * 1e6))
         
    # neuron 1
    while(len(ts1) > np.int32(10 * len(ts)/100 )):
        rd_id = np.random.randint(len(ts1))
        ts1.pop(rd_id)
        
    while(len(ts1) != len(ts)):
        rd_id = np.random.randint(len(ts))
        lag = 1000 #np.random.randint(500, 2500)
        if(len(np.where(np.array(ts1)==ts[rd_id])[0]+lag)==0):
            ts1.append(ts[rd_id]+lag)
        
    while(len(ts2) > np.int32(50 * len(ts)/100 )):
        rd_id = np.random.randint(len(ts2))
        ts2.pop(rd_id)
        
    while(len(ts2) != len(ts)):
        rd_id = np.random.randint(len(ts))
        lag = 1000 #np.random.randint(500, 2500)
        if(len(np.where(np.array(ts2)==ts[rd_id])[0]+lag)==0):
            ts2.append(ts[rd_id]+lag)
        
    while(len(ts3) > np.int32(90 * len(ts)/100 )):
        rd_id = np.random.randint(len(ts3))
        ts3.pop(rd_id)
        
    while(len(ts3) != len(ts)):
        rd_id = np.random.randint(len(ts))
        lag = 1000 #np.random.randint(500, 2500)
        if(len(np.where(np.array(ts3)==ts[rd_id])[0]+lag)==0):
            ts3.append(ts[rd_id]+lag)
            
    np.savez(npzTemp, ts, x, y, p, c)
    events_ = Events(npzTemp)    
    events_.sort_events()
    events_.save_as_file(path_save_basic+'t0.npz')
    
    np.savez(npzTemp, ts1, np.full_like(x, 1), y, p, c)
    events_ = Events(npzTemp)    
    events_.sort_events()
    events_.save_as_file(path_save_basic+'t1.npz')
    
    np.savez(npzTemp, ts2, np.full_like(x, 2), y, p, c)
    events_ = Events(npzTemp)    
    events_.sort_events()
    events_.save_as_file(path_save_basic+'t2.npz')
    
    np.savez(npzTemp, ts3, np.full_like(x, 3), y, p, c)
    events_ = Events(npzTemp)    
    events_.sort_events()
    events_.save_as_file(path_save_basic+'t3.npz')
    
    events_.add_events(path_save_basic + 't0.npz')
    events_.add_events(path_save_basic + 't1.npz')
    events_.add_events(path_save_basic + 't2.npz')
    
    events_.sort_events()
    events_.save_as_file(path_save_basic+str(i)+'.npz')



