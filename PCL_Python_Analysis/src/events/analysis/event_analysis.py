#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:19:28 2021

@author: thomas
"""

from dv import AedatFile
import numpy as np
# from numba import jit
import matplotlib.pyplot as plt
import os

receptive_field = (346, 260)  # px
file_name = "/home/thomas/Vidéos/DVS_Recordings/test/hand_slow.aedat4"
folder = "/home/thomas/neuvisys-report/data/"

def quantity_variation(events, binsize):
    nb_events = []
    t = np.arange(events["timestamp"][0], events["timestamp"][-1], binsize)
    for i in range(t.size -1):
        nb_events.append(events[(events["timestamp"] >= t[i]) & (events["timestamp"] < t[i+1])].size)
    return np.array(nb_events)

def spatial_correlation(x, y, polarity, timestamp, spat_corr, timestamps, tau, l, rf_size):
    if x >= l and x <= rf_size[0] - l - 1 and y >= l and y <= rf_size[1] - l - 1:
        if polarity:
            spat_corr[0] += 1 * (
                timestamp - timestamps[1, y - l : y + l + 1, x - l : x + l + 1] <= tau
            )  # P(On | On)
            spat_corr[1] += 1 * (
                timestamp - timestamps[0, y - l : y + l + 1, x - l : x + l + 1] <= tau
            )  # P(Off | On)
        else:
            spat_corr[2] += 1 * (
                timestamp - timestamps[1, y - l : y + l + 1, x - l : x + l + 1] <= tau
            )  # P(On | Off)
            spat_corr[3] += 1 * (
                timestamp - timestamps[0, y - l : y + l + 1, x - l : x + l + 1] <= tau
            )  # P(Off | Off)
    timestamps[1 * polarity, y, x] = timestamp

# @jit(nopython=True)
def temporal_correlation(x, y, polarity, timestamp, temp_corr, timestamps, tau, taus):
    if polarity:
        for i, t in enumerate(taus):
            temp_corr[0, i] += (
                1 * timestamps[0, y, x] >= timestamp - t
                and timestamps[0, y, x] < timestamp - t + tau
            )  # P(On | On)
            temp_corr[1, i] += (
                1 * timestamps[1, y, x] >= timestamp - t
                and timestamps[1, y, x] < timestamp - t + tau
            )  # P(Off | On)
    else:
        for i, t in enumerate(taus):
            temp_corr[2, i] += (
                1 * timestamps[0, y, x] >= timestamp - t
                and timestamps[0, y, x] < timestamp - t + tau
            )  # P(On | Off)
            temp_corr[3, i] += (
                1 * timestamps[1, y, x] >= timestamp - t
                and timestamps[1, y, x] < timestamp - t + tau
            )  # P(Off | Off)
    timestamps[1 * polarity, y, x] = timestamp

def compute_spatial_correlation():
    bins = [100]  # np.geomspace(100, 1000000, 9) # us
    rf_size = (346, 260)
    l = 40
    
    for tau in bins:
        with AedatFile(file_name) as f:
            print("Spatial Correlation computation with tau = " + str(tau))
            timestamps = np.zeros((rf_size[1], rf_size[0]))
            spat_corr = np.zeros((2 * l + 1, 2 * l + 1))
    
            count = 0
            for e in f["events"]:
                spatial_correlation(
                    e.x, e.y, e.polarity, e.timestamp, spat_corr, timestamps, tau, l, rf_size
                )
                count += 1
    
        spat_corr /= count
        # np.save(folder + "spat_corr_" + str(tau) + "_" + str(l), spat_corr)

def compute_temporal_correaltion():
    bins = [100, 500, 1000, 5000, 10000]  # us
    rf_size = (346, 260)
    tau_max = 500000
    
    for tau in bins:
        taus = np.arange(0, tau_max + 1, tau)
        with AedatFile(file_name) as f:
            print("Temporal Correlation computation with tau = " + str(tau))
            timestamps = np.zeros((2, rf_size[1], rf_size[0]))
            temp_corr = np.zeros((4, taus.size))
    
            count = 0
            for e in f["events"]:
                temporal_correlation(
                    e.x, e.y, e.polarity, e.timestamp, temp_corr, timestamps, tau, taus
                )
                count += 1
    
        temp_corr /= count
        np.save(folder + "temp_corr_" + str(tau), temp_corr)

def compute_cross_correlation():
    bins = [500]  # µs
    buf = 1000000
    
    for tau in bins:
        with AedatFile(file_name) as f:
            print("Cross Correlation computation with tau = " + str(tau))
            xon = np.zeros(buf)
            xoff = np.zeros(buf)
            count = 0
    
            for e in f["events"]:
                first = e.timestamp
                break
    
            for e in f["events"]:
                if (e.timestamp - first) // tau >= 1000000:
                    break
                if e.polarity:
                    xon[(e.timestamp - first) // tau] += 1
                else:
                    xoff[(e.timestamp - first) // tau] += 1
    
        xon = np.trim_zeros(xon, "b")
        xoff = xoff[: xon.size]
        np.save(folder + "xon_" + str(tau), xon)
        np.save(folder + "xoff_" + str(tau), xoff)
    
        cross_corr = []
        xon = (xon - np.mean(xon)) / np.std(xon)
        xoff = (xoff - np.mean(xoff)) / np.std(xoff)
        cross_corr.append(np.correlate(xon, xon, "full"))
        cross_corr.append(np.correlate(xon, xoff, "full"))
        cross_corr.append(np.correlate(xoff, xon, "full"))
        cross_corr.append(np.correlate(xoff, xoff, "full"))
        cross_corr = np.array(cross_corr)[:, cross_corr[0].size // 2 :]
        np.save(folder + "cross_corr_" + str(tau), cross_corr)

def compute_entripy_metric():
    directory = "/home/thomas/neuvisys-analysis/results/metric/"
    
    for i in range(5):
        sizes = [
            os.path.getsize(file) for file in os.scandir(directory + "weights_" + str(i))
        ]
        plt.hist(sizes, 20, alpha=0.5, label=str(i) + ": " + str(np.mean(sizes))[0:5])
        plt.legend()
    plt.show()
