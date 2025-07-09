#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:39:25 2020

@author: thomas
"""

import json
import os
from itertools import combinations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from natsort import natsorted
from scipy import spatial

import copy

from src.spiking_network.network.neuvisys import SpikingNetwork
import matplotlib

from src.spiking_network.analysis.modif_inhib_visual import (
    visualize_total_inhibition_evolution2,
    visualize_total_tdinhibition_evolution2,
    data_analysis_inhibition
)

def top_down_inhibition_weight_sum(spinet, responses: np.ndarray, stimulus: []):
    weight_sums = []
    for complex_cell in spinet.neurons[1][::spinet.l_shape[1, 2]]:
        simple_cells = complex_cell.params["in_connections"]

        weight = []
        for simple_cell in simple_cells:
            weight.append(spinet.neurons[0][simple_cell].weights_tdi)
        weight_sums.append(np.sum(np.array(weight), axis=0)
                           * np.std(responses[:, complex_cell.id]))
    weight_sums = np.array(weight_sums).flatten()

    index_pref_stimulus, change_stimulus, change_stimulus_indices = preffered_stimulus(
        responses, stimulus)

    plt.figure()
    x = np.arange(len(weight_sums))
    y = weight_sums
    last_index = 0

    full_sum = []
    for index in change_stimulus_indices:
        plt.bar(x[last_index:index], y[last_index:index])
        full_sum.append(np.sum(y[last_index:index]))
        last_index = index

    plt.xticks(change_stimulus_indices, change_stimulus, rotation=45)
    plt.title(
        "Sum of inhibition weights sorted by preferred complex cell orientation")
    plt.ylabel("Sum of inhibition weights")
    plt.xlabel("Complex cell preferred orientation in degrees (°)")
    plt.show()

    plt.figure()
    plt.bar(np.arange(len(full_sum)), full_sum)
    plt.show()


def lateral_inhibition_weight_sum(spinet: SpikingNetwork, responses: np.ndarray, stimulus: []):
    total = np.zeros(8)
    for simple_cell in spinet.neurons[0]:
        responses_subset = responses[:,
                                     simple_cell.params["lateral_dynamic_inhibition"]]
        index_pref_stimulus, change_stimulus, change_stimulus_indices = preffered_stimulus(
            responses_subset, stimulus)

        plt.figure()
        x = np.arange(len(simple_cell.weights_li))
        y = simple_cell.weights_li  # / np.max(simple_cell.weights_li)
        last_index = 0

        full_sum = []
        for index in change_stimulus_indices:
            plt.bar(x[last_index:index], y[last_index:index])
            full_sum.append(np.sum(y[last_index:index]))
            last_index = index
        total += np.array(full_sum)

        plt.xticks(change_stimulus_indices, change_stimulus, rotation=45)
        plt.title(
            "Normalized sum of inhibition weights sorted by preferred orientation")
        plt.ylabel("Normalized sum of inhibition weights")
        plt.xlabel("Complex cell preferred orientation in degrees (°)")
        plt.show()

        plt.figure()
        plt.bar(np.arange(len(full_sum)), full_sum)
        plt.show()
        break

    plt.figure()
    plt.bar(np.arange(len(total)), total)
    plt.show()

def visualize_potentials(spinet: SpikingNetwork, layer_id, neuron_z, visualize=True):
    number_of_displays = len(spinet.stats)
    x = [[]]
    y = [[]]
    step = 0.1
    val_max = 500
    excit_x = [[]]
    excit_y = [[]]
    x_rest = [[]]
    y_rest = [[]]
    x_spike = [[]]
    y_spike = [[]]
    thresh = [[]]
    #print(number_of_displays)
    for i in range(number_of_displays):
        for count, potential_time in enumerate(spinet.stats[i][str(i)][layer_id][str(layer_id)][neuron_z][1]["potential_train"]):
            x[i].append(potential_time[1])
            y[i].append(potential_time[0])
            if(layer_id == 0):
                thresh[i].append(30)
            else:
                thresh[i].append(3)
            if(y[i][-1] > thresh[i][-1]):
                x_rest[i].append(potential_time[1])
                y_rest[i].append(-20)
                x_spike[i].append(x[i][-1])
                y_spike[i].append(y[i][-1])
            # if(count+1 < len(spinet.stats[i][str(i)][layer_id][str(layer_id)][neuron_z][1]["potential_train"])):
            #     if(x[i][-1] < spinet.stats[i][str(i)][layer_id][str(layer_id)][neuron_z][1]["potential_train"][count+1][1]):
            #         number = int((spinet.stats[i][str(i)][layer_id][str(
            #             layer_id)][neuron_z][1]["potential_train"][count+1][1] - x[i][-1]-step) / step)
            #         if(number > val_max):
            #             number = val_max
            #         temp = np.linspace(x[i][-1]+step, spinet.stats[i][str(i)][layer_id][str(
            #             layer_id)][neuron_z][1]["potential_train"][count+1][1], number)
            #         potential = y[i][-1]
            #         time = x[i][-1]
            #         for elem in temp:
            #             x[i].append(elem)
            #             thresh[i].append(thresh[i][-1])
            #             if(layer_id == 0):
            #                 newpotential = potential * \
            #                     np.exp(-(elem - time) /
            #                            (spinet.simple_conf['TAU_M']*1000))
            #             else:
            #                 newpotential = potential * \
            #                     np.exp(-(elem - time) /
            #                            (spinet.complex_conf['TAU_M']*1000))
            #             y[i].append(newpotential)
        if(visualize):
            plt.figure(i+1)
            plt.plot(x[i], y[i], 'b-', label='potential of the neuron')

            plt.plot(x[i], thresh[i], 'r--', label='threshold')
            if(len(x[i]) != 0):
                zxrest = [x[i][0], x[i][-1]]
                zyrest = [-20, -20]
                wyrest = [0, 0]
                plt.plot(zxrest, zyrest, 'k--', label='resting potential')
                plt.plot(zxrest, wyrest, 'm--', label='value of decay')
                plt.plot(x_rest[i], y_rest[i], 'ks', label='spike_rest')
            plt.plot(x_spike[i], y_spike[i], 'rs', label='spike_threshold')

            once = True
            try:
                for inhibition in spinet.stats[i][str(i)][layer_id][str(layer_id)][neuron_z][3]["timing_of_inhibition"][0]:
                    x_temp = [inhibition[2], inhibition[2]]
                    y_temp = [inhibition[0], inhibition[1]]
                    y_temp_before = inhibition[0]
                    y_temp_after = inhibition[1]
                    if(once):
                        plt.plot(x_temp, y_temp, 'g-',
                                 label="static inhibition")
                        plt.plot(x_temp[0], y_temp_before, 'bs',
                                 label="init value before static inhib")
                        plt.plot(x_temp[0], y_temp_after, 'gs',
                                 label="value after static inhib")
                        once = False
                    else:
                        plt.plot(x_temp, y_temp, 'g-')
                        plt.plot(x_temp[0], y_temp_before, 'bs')
                        plt.plot(x_temp[0], y_temp_after, 'gs')
            except ValueError:
                pass

            once = True
            try:
                for inhibition in spinet.stats[i][str(i)][layer_id][str(layer_id)][neuron_z][3]["timing_of_inhibition"][1]:
                    x_temp = [inhibition[2], inhibition[2]]
                    y_temp = [inhibition[0], inhibition[1]]
                    y_temp_before = inhibition[0]
                    y_temp_after = inhibition[1]
                    if(once):
                        plt.plot(x_temp, y_temp, 'c-',
                                 label="lateral inhibition")
                        plt.plot(x_temp[0], y_temp_before, 'c*',
                                 label="init value before lateral inhib")
                        plt.plot(x_temp[0], y_temp_after, 'y*',
                                 label="value after lateral inhib")
                        once = False
                    else:
                        plt.plot(x_temp, y_temp, 'c-')
                        plt.plot(x_temp[0], y_temp_before, 'c*')
                        plt.plot(x_temp[0], y_temp_after, 'y*')
            except ValueError:
                pass

            once = True
            try:
                for inhibition in spinet.stats[i][str(i)][layer_id][str(layer_id)][neuron_z][3]["timing_of_inhibition"][2]:
                    x_temp = [inhibition[2], inhibition[2]]
                    y_temp = [inhibition[0], inhibition[1]]
                    y_temp_before = inhibition[0]
                    y_temp_after = inhibition[1]
                    if(once):
                        plt.plot(x_temp, y_temp, 'r-',
                                 label="topdown inhibition")
                        plt.plot(x_temp[0], y_temp_before, 'r*',
                                 label="init value before topdown inhib")
                        plt.plot(x_temp[0], y_temp_after, 'k*',
                                 label="value after topdown inhib")
                        once = False
                    else:
                        plt.plot(x_temp, y_temp, 'r-')
                        plt.plot(x_temp[0], y_temp_before, 'r*')
                        plt.plot(x_temp[0], y_temp_after, 'k*')
            except ValueError:
                pass

            plt.xlabel('Time (µs) ')
            plt.ylabel('Membrane potential (mV)')
            plt.legend(bbox_to_anchor=(1.1, 1.05))
            # plt.legend()

        if(i != number_of_displays-1):
            x.append([])
            y.append([])
            x_rest.append([])
            y_rest.append([])
            x_spike.append([])
            y_spike.append([])
            excit_x.append([])
            excit_y.append([])
            thresh.append([])

    # plt.tight_layout()
    # plt.show()
    return x, y  


def amount_of_excitation_inhibition(spinet: SpikingNetwork, layer_id, neuron_z, visualize=True, average_neurons=False):
    number_of_displays = len(spinet.stats)
    amounts = 4
    arge = np.arange(0, len(np.array(range(0, number_of_displays))+1)*5, 5)
    arge[0] = 1
    x = [[]]
    y = [[]]
    plt.figure(1)
    if(not average_neurons):
        for i in range(amounts):
            for j in range(number_of_displays):
                x[i].append(j+1)
                y[i].append(spinet.stats[j][str(j)][layer_id][str(
                    layer_id)][neuron_z][0]["amount_of_events"][i])
            if(visualize):
                if(i == 0 and max(y[i]) != 0):
                    plt.plot(x[i], np.array(y[i])/max(y[i]),
                             'r-', label="excitation")
                if(i == 1 and max(y[i]) != 0):
                    plt.plot(x[i], np.array(y[i])/max(y[i]),
                             'y-', label="static inhibition")
                if(i == 2 and max(y[i]) != 0):
                    plt.plot(x[i], np.array(y[i])/max(y[i]),
                             'k-', label="lateral inhibition")
                if(i == 3 and max(y[i]) != 0):
                    plt.plot(x[i], np.array(y[i])/max(y[i]),
                             'b-', label="topdown inhibition")
                plt.xlabel('length in pixels ')
                plt.ylabel('amount of events')
                plt.legend(bbox_to_anchor=(1.1, 1.05))
            x.append([])
            y.append([])

    else:
        deviation = [[]]
        for i in range(amounts):
            for j in range(number_of_displays):
                x[i].append(j+1)
                avg = 0
                std_dev = 0
                max_num = 0
                for value in range(len(spinet.stats[j][str(j)][layer_id][str(layer_id)])):
                    if (len(spinet.stats[j][str(j)][layer_id][str(layer_id)][value][0]["amount_of_events"]) != 0):
                        max_num += 1
                        avg += spinet.stats[j][str(j)][layer_id][str(
                            layer_id)][value][0]["amount_of_events"][i]
                avg /= max_num
                for value in range(len(spinet.stats[j][str(j)][layer_id][str(layer_id)])):
                    if (len(spinet.stats[j][str(j)][layer_id][str(layer_id)][value][0]["amount_of_events"]) != 0):
                        std_dev += (spinet.stats[j][str(j)][layer_id][str(
                            layer_id)][value][0]["amount_of_events"][i]-avg)**2
                    std_dev /= max_num
                    std_dev = np.sqrt(std_dev)
                y[i].append(avg)
                deviation[i].append(std_dev)
            if(visualize):
                if(i == 0 and max(y[i]) != 0):
                    plt.plot(arge, np.array(
                        y[i])/max(y[i]), 'r-', label="excitation") #/max(y[i])
                    print(max(y[i]))
                if(i == 1 and max(y[i]) != 0):
                    plt.plot(arge, np.array(
                        y[i])/max(y[i]), 'y-', label="static inhibition")
                if(i == 2 and max(y[i]) != 0):
                    plt.plot(arge, np.array(
                        y[i])/max(y[i]), 'k-', label="lateral inhibition")
                    for cter, value in enumerate(x[i]):
                        plt.plot([arge[cter], arge[cter]], [
                                 (y[i][cter]+deviation[i][cter])/max(y[i]), (y[i][cter]-deviation[i][cter])/max(y[i])], 'g-')
                if(i == 3 and max(y[i]) != 0):
                    plt.plot(arge, np.array(
                        y[i])/max(y[i]), 'b-', label="topdown inhibition")
                plt.xlabel('length in pixels ')
                plt.ylabel('amount of events')
                plt.legend(bbox_to_anchor=(1.1, 1.05))
            x.append([])
            y.append([])
            deviation.append([])
    plt.xticks(arge)
    plt.title("Amount of excitation and inhibition")
    return x, y

def visualize_inhibition_weights(spinet: SpikingNetwork, layer_id, neuron_id):
    lateral_weights = []
    true_z = []
    avg = []
    for z in range(spinet.l_shape[layer_id][2]):
        lateral_weights.append(spinet.neurons[layer_id][neuron_id].weights_lic)
        true_z.append(z)
    layout = np.load(spinet.path + "weights/layout_" + str(layer_id) + ".npy")
    x_neur = np.where(layout == neuron_id)[0][0]
    y_neur = np.where(layout == neuron_id)[1][0]
    if(len(lateral_weights) != 0):
        avg = []
        counter = 0
        wi = 0
        space = 1
        x = []
        y = []
        range_ = spinet.conf["neuronInhibitionRange"]
        range_x = range_[0]
        range_y = range_[1]
        for value_x in range(x_neur-range_x, x_neur+range_x+1):
            for value_y in range(y_neur-range_y, y_neur+range_y+1):
                if((value_x != x_neur or value_y != y_neur) and value_x >= 0 and value_y >= 0 and value_x < (len((spinet.conf["layerPatches"])[0][0]) * (spinet.conf["layerSizes"])[0][0]) and value_y < (len((spinet.conf["layerPatches"])[0][1]) * (spinet.conf["layerSizes"])[0][1])):
                    x.append(value_x)
                    y.append(value_y)
        initct = 0
        for value in lateral_weights:
            for value_ in value:
                wi += value_
                counter += 1
                if(counter > spinet.l_shape[layer_id][2]):
                    avg.append(wi/spinet.l_shape[layer_id][2])
                    wi = 0
                    counter = 0
            initct+=1
        fig = plt.figure(figsize=(20, 20), dpi=80)
        max_ = max(avg)
        avg = np.array(avg)/max_
        ax = fig.add_subplot(111)
        Blues = plt.get_cmap('Blues')
        rect = []
        for var in range(len(x)):
            rect.append(matplotlib.patches.Rectangle(
                (x[var], y[var]), space, space, color=Blues(avg[var])))
            ax.add_patch(rect[var])
        rect.append(matplotlib.patches.Rectangle(
            (x_neur, y_neur), space, space, color='k'))
        ax.add_patch(rect[-1])
        if(x_neur == 0 and y_neur == 0):
            plt.xlim([x_neur, x[-1]+space])
            plt.ylim([y_neur, y[-1]+space])
        elif(x_neur+space >= x[-1]+space and y_neur+space >= y[-1]+space):
            plt.xlim([x[0], x_neur+space])
            plt.ylim([y[0], y_neur+space])
        else:
            plt.xlim([x[0], x[-1]+space])
            plt.ylim([y[0], y[-1]+space])
        cmapp = matplotlib.cm.ScalarMappable(cmap=Blues)
        cmapp.set_clim(0, max_)
        plt.colorbar(cmapp, ax=ax, ticks=(0, max_/4, max_/2, 3*max_/4, max_))
        plt.gca().set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        plt.axis('off')
        plt.show()
    return avg

def visualize_excitation_weights(spinet: SpikingNetwork, layer_id, neuron_id):
    lateral_weights = []
    true_z = []
    avg = []
    for z in range(spinet.l_shape[layer_id][2]):
        lateral_weights.append(spinet.neurons[layer_id][neuron_id].weights_le)
        true_z.append(z)
    layout = np.load(spinet.path + "weights/layout_" + str(layer_id) + ".npy")
    x_neur = np.where(layout == neuron_id)[0][0]
    y_neur = np.where(layout == neuron_id)[1][0]
    if(len(lateral_weights) != 0):
        avg = []
        counter = 0
        wi = 0
        space = 1
        x = []
        y = []
        range_ = spinet.conf["neuronInhibitionRange"]
        range_x = range_[0]
        range_y = range_[1]
        for value_x in range(x_neur-range_x, x_neur+range_x+1):
            for value_y in range(y_neur-range_y, y_neur+range_y+1):
                if((value_x != x_neur or value_y != y_neur) and value_x >= 0 and value_y >= 0 and value_x < (len((spinet.conf["layerPatches"])[0][0]) * (spinet.conf["layerSizes"])[0][0]) and value_y < (len((spinet.conf["layerPatches"])[0][1]) * (spinet.conf["layerSizes"])[0][1])):
                    x.append(value_x)
                    y.append(value_y)
        initct = 0
        for value in lateral_weights:
            for value_ in value:
                wi += value_
                counter += 1
                if(counter > spinet.l_shape[layer_id][2]):
                    avg.append(wi/spinet.l_shape[layer_id][2])
                    wi = 0
                    counter = 0
            initct+=1
        fig = plt.figure(figsize=(20, 20), dpi=80)
        max_ = max(avg)
        avg = np.array(avg)/max_
        ax = fig.add_subplot(111)
        Blues = plt.get_cmap('Blues')
        rect = []
        for var in range(len(x)):
            rect.append(matplotlib.patches.Rectangle(
                (x[var], y[var]), space, space, color=Blues(avg[var])))
            ax.add_patch(rect[var])
        rect.append(matplotlib.patches.Rectangle(
            (x_neur, y_neur), space, space, color='k'))
        ax.add_patch(rect[-1])
        if(x_neur == 0 and y_neur == 0):
            plt.xlim([x_neur, x[-1]+space])
            plt.ylim([y_neur, y[-1]+space])
        elif(x_neur+space >= x[-1]+space and y_neur+space >= y[-1]+space):
            plt.xlim([x[0], x_neur+space])
            plt.ylim([y[0], y_neur+space])
        else:
            plt.xlim([x[0], x[-1]+space])
            plt.ylim([y[0], y[-1]+space])
        cmapp = matplotlib.cm.ScalarMappable(cmap=Blues)
        cmapp.set_clim(0, max_)
        plt.colorbar(cmapp, ax=ax, ticks=(0, max_/4, max_/2, 3*max_/4, max_))
        plt.gca().set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        plt.axis('off')
        plt.show()
    return avg

#NO LONGER USED
def visualize_td_inhibition(spinet: SpikingNetwork, layer_id, neuron_id):
    weights_tdi = spinet.neurons[layer_id][neuron_id].weights_tdi
    depth = spinet.l_shape[layer_id+1][2]
    max_ = max(weights_tdi)
    weights_tdi = weights_tdi / max_
    space = 1
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = fig.add_subplot(111)
    Blues = plt.get_cmap('Blues')
    rect = []
    ordinate = np.linspace(0,len(weights_tdi)/depth -1,int(len(weights_tdi)/depth))
    y_i=0
    x_i=0
    for i, value in enumerate(weights_tdi):
        if(i%depth==0 and i!=0):
            y_i+=1
        if(y_i==0):
            x_i=i
        else:
            x_i=i-y_i*depth
        rect.append(matplotlib.patches.Rectangle(
            (x_i, ordinate[y_i]), space, space, color=Blues(value)))
        ax.add_patch(rect[i])
    cmapp = matplotlib.cm.ScalarMappable(cmap=Blues)
    cmapp.set_clim(0, max_)
    plt.colorbar(cmapp, ax=ax, ticks=(0, max_/4, max_/2, 3*max_/4, max_))
    plt.gca().set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    plt.xlim([0, depth+space])
    plt.ylim([ordinate[0], ordinate[-1]+space])
    plt.axis('off')
    plt.show()

def visualize_td_excitation(spinet: SpikingNetwork, layer_id, neuron_id):
    weights_tde = spinet.neurons[layer_id][neuron_id].weights_tde
    depth = spinet.l_shape[layer_id+1][2]
    max_ = max(weights_tde)
    weights_tde = weights_tde / max_
    space = 1
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = fig.add_subplot(111)
    Blues = plt.get_cmap('Blues')
    rect = []
    ordinate = np.linspace(0,len(weights_tde)/depth -1,int(len(weights_tde)/depth))
    y_i=0
    x_i=0
    for i, value in enumerate(weights_tde):
        if(i%depth==0 and i!=0):
            y_i+=1
        if(y_i==0):
            x_i=i
        else:
            x_i=i-y_i*depth
        rect.append(matplotlib.patches.Rectangle(
            (x_i, ordinate[y_i]), space, space, color=Blues(value)))
        ax.add_patch(rect[i])
    cmapp = matplotlib.cm.ScalarMappable(cmap=Blues)
    cmapp.set_clim(0, max_)
    plt.colorbar(cmapp, ax=ax, ticks=(0, max_/4, max_/2, 3*max_/4, max_))
    plt.gca().set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    plt.xlim([0, depth+space])
    plt.ylim([ordinate[0], ordinate[-1]+space])
    plt.axis('off')
    plt.show()

#OLD_VER    
def visualize_td_sum_inhibition(spinet: SpikingNetwork, layer_id, neuron_id, neuron_z, sequence):
    number_of_displays=len(spinet.stats)
    max_tdi = 0
    sum_weights_tdi = np.zeros((np.array(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][0][6]["top_down_weights"])).shape)
    for seq in range(number_of_displays):
        sum_seq = np.zeros((np.array(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][0][6]["top_down_weights"])).shape)
        for r in range(len(spinet.stats[seq][str(seq)][layer_id][str(layer_id)])):
            try:
                sum_seq += np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][r][6]["top_down_weights"])
                if(seq==sequence):
                    sum_weights_tdi += np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][r][6]["top_down_weights"])
            except:
                ValueError
        if(max_tdi < np.max(sum_seq)):
            max_tdi= np.max(sum_seq)
    sum_weights_tdi /=len(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)])
    space = 1
    #print(sum_weights_tdi.shape)
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = fig.add_subplot(111)
    Blues = plt.get_cmap('Blues')
    rect = []
    max_tdi /= len(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)])
    depth = spinet.l_shape[layer_id+1][2]
    ordinate = np.linspace(0,len(spinet.neurons[layer_id][neuron_id].out_connections)/depth -1,int(len(spinet.neurons[layer_id][neuron_id].out_connections)/depth))
    y_i=0
    x_i=0
    for i in range(sum_weights_tdi.shape[0]*sum_weights_tdi.shape[1]):
        if(i%depth==0 and i!=0):
            y_i+=1
        if(y_i==0):
            x_i=i
        else:
            x_i=i-y_i*depth
        rect.append(matplotlib.patches.Rectangle(
            (x_i, ordinate[y_i]), space, space, color=Blues(sum_weights_tdi[y_i][x_i])))
        ax.add_patch(rect[i])
    cmapp = matplotlib.cm.ScalarMappable(cmap=Blues)
    cmapp.set_clim(0, max_tdi)
    plt.colorbar(cmapp, ax=ax, ticks=(0, max_tdi/4, max_tdi/2, 3*max_tdi/4, max_tdi))
    plt.gca().set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    plt.xlim([0, depth+space])
    plt.ylim([ordinate[0], ordinate[-1]+space])
    plt.axis('off')
    plt.show()
    
#OLD_VER    
def visualize_sum_inhibition_weights(spinet: SpikingNetwork, layer_id, neuron_id, neuron_z, sequence):
    lateral_weights = spinet.neurons[layer_id][neuron_id].weights_li
    sum_weights_lat = np.zeros((np.array(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][1])).shape)
    #sum_weights_lat = spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][neuron_z][2]["sum_inhib_weights"][1]
    max_lat = 0
    number_of_displays = len(spinet.stats)
    for seq in range(number_of_displays):
        if(seq<5):
            continue
        sum_seq = np.zeros((np.array(spinet.stats[seq][str(seq)][layer_id][str(
            layer_id)][0][2]["sum_inhib_weights"][1])).shape)
        for r in range(len(spinet.stats[seq][str(seq)][layer_id][str(layer_id)])):
            sum_seq += np.array(spinet.stats[seq][str(seq)][layer_id]
                                [str(layer_id)][r][2]["sum_inhib_weights"][1])
            if(seq == sequence):
                sum_weights_lat += np.array(spinet.stats[seq][str(
                    seq)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][1])
        if(max_lat < max(sum_seq)):
            max_lat = max(sum_seq)
            #print(seq)
    sum_weights_lat /= len(spinet.stats[sequence]
                           [str(sequence)][layer_id][str(layer_id)])
    max_lat /= len(spinet.stats[sequence]
                   [str(sequence)][layer_id][str(layer_id)])

    layout = np.load(spinet.path + "weights/layout_" + str(layer_id) + ".npy")
    x_neur = np.where(layout == neuron_id)[0][0]
    y_neur = np.where(layout == neuron_id)[1][0]
    if(len(lateral_weights) != 0):
        avg = []
        wi = 0
        space = 1
        x = []
        y = []
        range_ = spinet.conf["neuronInhibitionRange"]
        range_x = range_[0]
        range_y = range_[1]
        it = 0
        for value_x in range(x_neur-range_x, x_neur+range_x+1):
            for value_y in range(y_neur-range_y, y_neur+range_y+1):
                if(it < len(sum_weights_lat) and (value_x != x_neur or value_y != y_neur)):
                    value = sum_weights_lat[it]
                    it += 1
                if((value_x != x_neur or value_y != y_neur) and value_x >= 0 and value_y >= 0 and value_x < (len((spinet.conf["layerPatches"])[0][0]) * (spinet.conf["layerSizes"])[0][0]) and value_y < (len((spinet.conf["layerPatches"])[0][1]) * (spinet.conf["layerSizes"])[0][1])):
                    x.append(value_x)
                    y.append(value_y)
                    wi = value
                    avg.append(wi)
        fig = plt.figure(figsize=(20, 20), dpi=80)
        #max_=max(avg)
        max_ = max_lat
        avg = np.array(avg)/max_
        ax = fig.add_subplot(111)
        Blues = plt.get_cmap('Greys')
        rect = []
        print(len(x))
        for var in range(len(x)):
            rect.append(matplotlib.patches.Rectangle(
                (x[var], y[var]), space, space, color=Blues(avg[var])))
            ax.add_patch(rect[var])
        rect.append(matplotlib.patches.Rectangle(
            (x_neur, y_neur), space, space, color='k'))
        ax.add_patch(rect[-1])
        if(x_neur == 0 and y_neur == 0):
            ax.set_xlim([x_neur, x[-1]+space])
            ax.set_ylim([y_neur, y[-1]+space])
        elif(x_neur+space >= x[-1]+space and y_neur+space >= y[-1]+space):
            ax.set_xlim([x[0], x_neur+space])
            ax.set_ylim([y[0], y_neur+space])
        else:
            ax.set_xlim([x[0], x[-1]+space])
            ax.set_ylim([y[0], y[-1]+space])
        cmapp = matplotlib.cm.ScalarMappable(cmap=Blues)
        cmapp.set_clim(0, max_)
        plt.colorbar(cmapp, ax=ax, ticks=(0, max_/4, max_/2, 3*max_/4, max_))
        plt.gca().set_aspect('equal', adjustable='box')
        ax.invert_yaxis()
        plt.axis('off')
        plt.show()
    return avg

#OLD_VER
def visualize_evolution_of_inhibition(spinet: SpikingNetwork, layer_id, neuron_id, neuron_z, x_neur_to_look, y_neur_to_look, norm_factor=1):
    lateral_weights = spinet.neurons[layer_id][neuron_id].weights_li
    number_of_displays = len(spinet.stats)
    sum_weights_lat_total = []
    for seq in range(number_of_displays):
        sum_weights_lat = np.zeros((np.array(spinet.stats[seq][str(
            seq)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][1])).shape)
        for r in range(len(spinet.stats[seq][str(seq)][layer_id][str(layer_id)])):
            sum_weights_lat += np.array(spinet.stats[seq][str(
                seq)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][1])
        sum_weights_lat /= len(spinet.stats[seq]
                               [str(seq)][layer_id][str(layer_id)])
        sum_weights_lat_total.append(sum_weights_lat)

    # for sequence in range(number_of_displays):
      #  sum_weights_lat = spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][neuron_z][2]["sum_inhib_weights"][1]
       # sum_weights_lat_total.append(sum_weights_lat)
    layout = np.load(spinet.path + "weights/layout_" + str(layer_id) + ".npy")
    x_neur = np.where(layout == neuron_id)[0][0]
    y_neur = np.where(layout == neuron_id)[1][0]
    if(len(lateral_weights) != 0):
        avg = []
        wi = 0
        range_ = spinet.conf["neuronInhibitionRange"]
        range_x = range_[0]
        range_y = range_[1]
        for sequence in range(number_of_displays):
            it = 0
            for value_x in range(x_neur-range_x, x_neur+range_x+1):
                for value_y in range(y_neur-range_y, y_neur+range_y+1):
                    if((it < len(sum_weights_lat_total[sequence])) and (value_x != x_neur or value_y != y_neur)):
                        value = sum_weights_lat_total[sequence][it]
                        it += 1
                    if(value_x == x_neur and value_y == y_neur):
                        value = 0
                    if((value_x == x_neur_to_look and value_y == y_neur_to_look)):
                        wi = value
                        avg.append(wi)
        if(norm_factor == 1):
            arge = np.arange(
                0, len(np.array(range(0, number_of_displays))+1)*5, 5)
            #arge[0] = 1
            ax = plt.plot(
                arge, avg, 'k-', label="evolution of the amount of inhibition sent by the neuron")
        else:
            avg = (np.array(avg)/max(avg)) * norm_factor
            ax = plt.plot(range(1, number_of_displays+1), avg, 'k-',
                          label="evolution of the amount of inhibition sent by the neuron")
    return avg

#NO LONGER USED
def visualize_total_inhibition_evolution(spinet: SpikingNetwork, layer_id, neuron_id, neuron_z, norm_factor=1):
    lateral_weights = spinet.neurons[layer_id][neuron_id].weights_li
    number_of_displays = len(spinet.stats)
    sum_weights_lat_total = []
    for seq in range(number_of_displays):
        try:
            sum_weights_lat = np.zeros((np.array(spinet.stats[seq][str(
                seq)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][1])).shape)
            for r in range(len(spinet.stats[seq][str(seq)][layer_id][str(layer_id)])):
                """if(np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][1]).shape[0] == 80):
                    print(r)
                    print(r)"""
                sum_weights_lat += np.array(spinet.stats[seq][str(
                    seq)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][1])
            sum_weights_lat /= len(spinet.stats[seq]
                               [str(seq)][layer_id][str(layer_id)])
            sum_weights_lat_total.append(sum_weights_lat)
        except: 
            sum_weights_lat_total.append(np.zeros((np.array(spinet.stats[number_of_displays-1][str(
                number_of_displays-1)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][1])).shape))
    layout = np.load(spinet.path + "weights/layout_" + str(layer_id) + ".npy")
    x_neur = np.where(layout == neuron_id)[0][0]
    y_neur = np.where(layout == neuron_id)[1][0]
    if(len(lateral_weights) != 0):
        avg = []
        wi = 0
        range_ = spinet.conf["neuronInhibitionRange"]
        range_x = range_[0]
        range_y = range_[1]
        for sequence in range(number_of_displays):
            it = 0
            wi = 0
            for value_x in range(x_neur-range_x, x_neur+range_x+1):
                for value_y in range(y_neur-range_y, y_neur+range_y+1):
                    if((it < len(sum_weights_lat_total[sequence])) and (value_x != x_neur or value_y != y_neur)):
                        value = sum_weights_lat_total[sequence][it]
                        it += 1
                    if((value_x != x_neur or value_y != y_neur) and value_x >= 0 and value_y >= 0 and value_x < (len((spinet.conf["layerPatches"])[0][0]) * (spinet.conf["layerSizes"])[0][0]) and value_y < (len((spinet.conf["layerPatches"])[0][1]) * (spinet.conf["layerSizes"])[0][1])):
                        wi += value
            avg.append(wi)
        arge = np.arange(1, len(np.array(range(0, number_of_displays)))+1)
        #arge[0] = 1
        if(norm_factor == 1):
            plt.plot(arge, avg, 'r-',
                          label="total amount of lateral inhibition")
        else:
            avg = (np.array(avg)/max(avg)) * norm_factor
            plt.plot(
                arge, avg, 'r-', label="evolution of the amount of inhibition sent by the neuron")
    return avg

#OLD_VER
def visualize_total_tdinhibition_evolution(spinet: SpikingNetwork, layer_id, neuron_id, neuron_z):
    topdown_weights = spinet.neurons[layer_id][neuron_id].weights_tdi
    number_of_displays = len(spinet.stats)
    sum_weights_td_total = []
    for seq in range(number_of_displays):
        sum_weights_td = np.zeros((np.array(spinet.stats[seq][str(
            seq)][layer_id][str(layer_id)][0][6]["top_down_weights"])).shape)
        for r in range(len(spinet.stats[seq][str(seq)][layer_id][str(layer_id)])):
            sum_weights_td += np.array(spinet.stats[seq][str(
                seq)][layer_id][str(layer_id)][r][6]["top_down_weights"])
        sum_weights_td /= len(spinet.stats[seq]
                               [str(seq)][layer_id][str(layer_id)])
        sum_weights_td_total.append(sum_weights_td)
    if(len(topdown_weights) != 0):
        avg = []
        wi = 0
        for sequence in range(number_of_displays):
            wi = 0
            for cplex_cell in range(len(sum_weights_td_total[sequence])):
                    for depth in range(spinet.l_shape[layer_id+1][2]):
                        #print(np.array(sum_weights_td_total).shape)
                        wi += sum_weights_td_total[sequence][cplex_cell][depth]
            avg.append(wi)
        arge = np.arange(0, len(np.array(range(0, number_of_displays))))
        #arge[0] = 1
        ax = plt.plot(arge, avg, 'g-',label="total amount of topdown inhibition")
    return avg
  
def suppression_metric(spikes, start_space = 45):
    #give as input spikes with length increasing by 1 pixel each time.
    spikes = np.array(spikes)
    max_spikes = np.max(spikes)
    metric = calculate_metric(max_spikes, spikes[-1])
    return metric

def calculate_metric(max_spike, spike_val):
    v = 100 - (spike_val*100/max_spike) 
    return v

def average_over_multiple_stimuli(spinet: SpikingNetwork, save_folder, angles, surr_angles, n_simulation, neuron_id, layer_id = 0, neuron_z = 0, max_depth = 16, thresh = 30, tuned_ori=False, lat_value = 0):
    spinet.load_orientations()
    sim_mean_spikes = []
    
    sim_mean_spikes_same_angle = []
    neur_ids = [1920, 1984, 2048, 2496, 2560, 2624, 3072, 3136, 3200]
    neur_ids_total = range(0,5184,64)
    max_depth = max_depth
    z_neur = 0
    
    for index_angle, main_angle in enumerate(angles):
        for surround_angle in surr_angles:
            angle_mean_spikes = []
            for sim in range(n_simulation):
                print("Main angle {}° and surround angle {}°".format(main_angle, surround_angle))
                spinet.load_statistics_3(main_angle, surround_angle, layer_id, sim, lat_value)
                y_tot = []
                for neuron_id in neur_ids: 
                    print("z_neur = {} and neuron_id = {}".format(z_neur, neuron_id))
                    for i in range(max_depth):
                        theta = np.array(spinet.neurons[layer_id][neuron_id+neuron_z].theta)
                        
                        theta_3 = theta[2]
                        theta = np.append(np.array(theta[0]),np.array(theta[1]))
                        theta = np.append(theta, np.array(theta_3))
                        new_angle = main_angle
                        new_angle2 = surround_angle
                        if(new_angle!=0):
                            condition = ( (np.array(theta)==new_angle).any() ) or ( (np.array(theta)==-new_angle).any() )
                        elif(new_angle==0):
                            condition = ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() )
                        for seq in range(1):
                            temp = np.moveaxis(np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][z_neur][1]["potential_train"]),-1,0)
                            try:
                                if( (temp[0]>thresh).any() ):
                                    ok = True
                                    break
                                else:
                                    ok = False
                            except: 
                                ok = False
                        new_condition = ok
                        """if(condition):
                            print("Condition is true!")
                        else:
                            print("Condition is false.")
                        if(new_condition):
                            print("New condition is true")
                        else:
                            print("New condition is false")"""
                        if(condition):
                            _, y = visualize_potentials(spinet, layer_id, z_neur, False)
                            y_tot.append(len(np.where(np.array(y)>=thresh)[0]))
                            neuron_z+=1
                            z_neur+=1
                            if(neuron_z>=max_depth):
                                break
                        else:
                            neuron_z+=1
                            z_neur+=1
                            if(neuron_z>=max_depth):
                                break
                    neuron_z = 0
                z_neur = 0
                mean_val = []
                mean = 0
                for j in range(len(y_tot)):
                    mean += y_tot[j]
                if(len(y_tot)!=0):
                    mean/=len(y_tot)
                else:
                    mean = 0
                mean_val.append(mean)
                    
                angle_mean_spikes.append(mean_val)
        
            if(main_angle!=surround_angle):
                sim_mean_spikes.append(np.mean(angle_mean_spikes, axis=0))
            else:
                sim_mean_spikes_same_angle.append(np.mean(angle_mean_spikes, axis=0))
            
            std_spikes_err = []
            std_lat_err = []
            std_td_err = []
            std_stat_err = []
            std_lat_tuned_err = []
            std_td_tuned_err = []
            std_stat_tuned_err = []
            
            std_spikes = 0
            for j in range(n_simulation):
                if(main_angle!=surround_angle):
                    std_spikes+=(angle_mean_spikes[j][0]-sim_mean_spikes[-1])**2
                else:
                    std_spikes+=(angle_mean_spikes[j][0]-sim_mean_spikes_same_angle[-1])**2
            
            
            std_spikes_err.append(np.sqrt(std_spikes/(n_simulation-1))/n_simulation)

            if(main_angle!=surround_angle):
                np.save(save_folder+str(main_angle)+"/per_surround/" + str(surround_angle)+"/" + "spikes_avg.npy", np.array(sim_mean_spikes[-1]))
                
                np.save(save_folder+str(main_angle)+"/per_surround/" + str(surround_angle)+"/" + "spikes_stderr.npy", np.array(std_spikes_err))
                

            else:
                np.save(save_folder+str(main_angle)+"/per_surround/" + str(surround_angle)+"/" + "spikes_avg.npy", np.array(sim_mean_spikes_same_angle[-1]))
                
                np.save(save_folder+str(main_angle)+"/per_surround/" + str(surround_angle)+"/" + "spikes_stderr.npy", np.array(std_spikes_err))
                

        
        
    avg_spikes = np.mean(sim_mean_spikes, axis=0)
    
    avg_spikes_same_angle = np.mean(sim_mean_spikes_same_angle, axis=0)
    
    std_spikes_err = []
    
    std_spikes_err_same_angle = []
    
    std_spikes_same_angle = 0
    
    for j in range(n_simulation):
        
        std_spikes+=(sim_mean_spikes[j]-avg_spikes[0])**2
        
        std_spikes_same_angle+=(sim_mean_spikes_same_angle[j]-avg_spikes_same_angle[0])**2
        
    std_spikes_err.append(np.sqrt(std_spikes/(len(angles)-1))/(len(angles)))
    
    std_spikes_err_same_angle.append(np.sqrt(std_spikes_same_angle/(len(angles)-1))/(len(angles)))
        
    np.save(save_folder+"/average/" + "spikes_avg.npy", np.array(avg_spikes))
    np.save(save_folder+"/average/" + "spikes_stderr.npy", np.array(std_spikes_err))
    
    np.save(save_folder+"/average/" + "spikes_avg_same_angle.npy", np.array(avg_spikes_same_angle))
    np.save(save_folder+"/average/" + "spikes_stderr_same_angle.npy", np.array(std_spikes_err_same_angle))

def tuningCurvesOutput(spinet: SpikingNetwork, save_folder, angles, surr_angles, neuron_id, fqcy, layer_id = 0, neuron_z = 0, max_depth = 16, thresh = 30, tuned_ori=False, n_simulation = 0):
    sim_mean_spikes = []
    
    sim_mean_spikes_same_angle = []
    neur_ids = [2560]
    max_depth = max_depth
    z_neur = 0
    
    for depth in range(max_depth):
        for index_angle, main_angle in enumerate(angles):
            for surround_angle in surr_angles:
                angle_mean_spikes = []
                for sim in range(n_simulation):
                    spinet.load_statistics_4(fqcy, surround_angle, layer_id, main_angle,  sim) 
                    y_tot = []
                    for neuron_id in neur_ids: 
                        for i in range(max_depth):
                            if(neuron_z == depth):
                                _, y = visualize_potentials(spinet, layer_id, z_neur, False)
                                y_tot.append(len(np.where(np.array(y)>=thresh)[0]))
                                neuron_z+=1
                                z_neur+=1
                                if(neuron_z>=max_depth):
                                    break
                            else:
                                neuron_z+=1
                                z_neur+=1
                                if(neuron_z>=max_depth):
                                    break
                        neuron_z = 0
                    z_neur = 0
                    mean_val = []
                    mean = 0
                    for j in range(len(y_tot)):
                        mean += y_tot[j]
                    if(len(y_tot)!=0):
                        mean/=len(y_tot)
                    else:
                        mean = 0
                    mean_val.append(mean)
                    
                    angle_mean_spikes.append(mean_val)
                    
                if(main_angle!=surround_angle):
                    sim_mean_spikes.append(np.mean(angle_mean_spikes, axis=0))
                else:
                    sim_mean_spikes_same_angle.append(np.mean(angle_mean_spikes, axis=0))
                
                std_spikes_err = []
                
                std_spikes = 0
                for j in range(n_simulation):
                    if(main_angle!=surround_angle):
                        std_spikes+=(angle_mean_spikes[j][0]-sim_mean_spikes[-1])**2
                    else:
                        std_spikes+=(angle_mean_spikes[j][0]-sim_mean_spikes_same_angle[-1])**2
                
                
                std_spikes_err.append(np.sqrt(std_spikes/(n_simulation-1))/n_simulation)
    
                if(main_angle!=surround_angle):
                    np.save(save_folder+str(depth) + "/" + str(main_angle)+"/per_surround/" + str(surround_angle)+"/" + "spikes_avg.npy", np.array(sim_mean_spikes[-1]))
                    
                    np.save(save_folder+str(depth) + "/" + str(main_angle)+"/per_surround/" + str(surround_angle)+"/" + "spikes_stderr.npy", np.array(std_spikes_err))
                    
    
                else:
                    np.save(save_folder+str(depth) + "/" + str(main_angle)+"/per_surround/" + str(surround_angle)+"/" + "spikes_avg.npy", np.array(sim_mean_spikes_same_angle[-1]))
                    
                    np.save(save_folder+str(depth) + "/" + str(main_angle)+"/per_surround/" + str(surround_angle)+"/" + "spikes_stderr.npy", np.array(std_spikes_err))
                
def tuningCurvesOutputSine(spinet: SpikingNetwork, save_folder, angles, phases, neuron_id, fqcy, layer_id = 0, neuron_z = 0, max_depth = 16, thresh = 30, tuned_ori=False, n_simulation = 0):
    sim_mean_spikes = []
    
    neur_ids = [2560]
    z_neur = 0
    
    for depth in range(max_depth):
        for index_angle, main_angle in enumerate(angles):
            for phase in phases:
                angle_mean_spikes = []
                for sim in range(n_simulation):
                    spinet.load_statistics_4(fqcy, phase, layer_id, main_angle,  sim) 
                    y_tot = []
                    for neuron_id in neur_ids: 
                        for i in range(max_depth):
                            if(neuron_z == depth):
                                _, y = visualize_potentials(spinet, layer_id, z_neur, False)
                                y_tot.append(len(np.where(np.array(y)>=thresh)[0]))
                                neuron_z+=1
                                z_neur+=1
                                if(neuron_z>=max_depth):
                                    break
                            else:
                                neuron_z+=1
                                z_neur+=1
                                if(neuron_z>=max_depth):
                                    break
                        neuron_z = 0
                    z_neur = 0
                    mean_val = []
                    mean = 0
                    for j in range(len(y_tot)):
                        mean += y_tot[j]
                    if(len(y_tot)!=0):
                        mean/=len(y_tot)
                    else:
                        mean = 0
                    mean_val.append(mean)
                    
                    angle_mean_spikes.append(mean_val)
                    
                sim_mean_spikes.append(np.mean(angle_mean_spikes, axis=0))
                
                std_spikes_err = []
                
                std_spikes = 0
                for j in range(n_simulation):
                    std_spikes+=(angle_mean_spikes[j][0]-sim_mean_spikes[-1])**2
                
                std_spikes_err.append(np.sqrt(std_spikes/(n_simulation-1))/n_simulation)
                np.save(save_folder+str(depth) + "/" + str(main_angle)+"/per_surround/" + str(phase)+"/" + "spikes_avg.npy", np.array(sim_mean_spikes[-1]))
                
                np.save(save_folder+str(depth) + "/" + str(main_angle)+"/per_surround/" + str(phase)+"/" + "spikes_stderr.npy", np.array(std_spikes_err))
                    
def phaseCurveOutput(spinet: SpikingNetwork, save_folder, angles, surr_angles, neuron_id, fqcy, layer_id = 0, neuron_z = 0, max_depth = 16, thresh = 30, tuned_ori=False, n_simulation = 0, start_val = 100000, end_val = 2001000):
    sim_mean_spikes = []
    
    sim_mean_spikes_same_angle = []
    neur_ids = [2560]
    max_depth = max_depth
    z_neur = 0
    windows_upperBound = range(start_val,end_val,start_val)
    alpha = 0.6
    
    for depth in range(max_depth):
        for index_angle, main_angle in enumerate(angles):
            for surround_angle in surr_angles:
                if(surround_angle!=main_angle):
                    continue
                angle_mean_spikes = [] 
                for sim in range(n_simulation):
                    print("Main angle {}° and surround angle {}°".format(main_angle, surround_angle))
                    spinet.load_statistics_4(main_angle, surround_angle, layer_id, fqcy,  sim)
                    y_tot = []
                    y_tot_window = []
                    for neuron_id in neur_ids: 
                        print("z_neur = {} and neuron_id = {}".format(z_neur, neuron_id))
                        for i in range(max_depth):
                            if(neuron_z == depth):
                                x, y = visualize_potentials(spinet, layer_id, z_neur, False)
                                temp = np.where(np.array(y)>=thresh)[1]
                                for e in temp:
                                    y_tot.append(x[0][e])
                                neuron_z+=1
                                z_neur+=1
                                if(neuron_z>=max_depth):
                                    break
                            else:
                                neuron_z+=1
                                z_neur+=1
                                if(neuron_z>=max_depth):
                                    break
                        neuron_z = 0
                    z_neur = 0
                    spike_windows = []
                    for idex, j in enumerate(windows_upperBound):
                        spikes = 0
                        for k in range(len(y_tot)):
                            if(idex==0 and y_tot[k] <= j):
                                spikes+=1
                            elif(idex!=0 and y_tot[k] > windows_upperBound[idex-1] and y_tot[k] <= j):
                                spikes+=1
                        
                        spike_windows.append(spikes)
                        print(spike_windows[-1])
                    
                    
                    angle_mean_spikes.append(spike_windows)
                if(main_angle!=surround_angle):
                    sim_mean_spikes.append(np.mean(angle_mean_spikes, axis=0))
                else:
                    sim_mean_spikes_same_angle.append(np.mean(angle_mean_spikes, axis=0))
                
                std_spikes_err = []
                
                std_spikes = np.zeros((np.array(sim_mean_spikes_same_angle)[-1].shape))
                for j in range(n_simulation):
                    if(main_angle!=surround_angle):
                        std_spikes+=(angle_mean_spikes[j][0]-sim_mean_spikes[-1])**2
                    else:
                        std_spikes+=((np.array(angle_mean_spikes)[j]-np.array(sim_mean_spikes_same_angle)[-1])**2)
                
                
                std_spikes_err.append(np.sqrt(std_spikes/(n_simulation-1))/n_simulation)
    
                if(main_angle!=surround_angle):
                    np.save(save_folder+str(depth) + "/" + str(main_angle)+"/per_surround/" + str(surround_angle)+"/" + "spikes_avg.npy", np.array(sim_mean_spikes[-1]))
                    
                    np.save(save_folder+str(depth) + "/" + str(main_angle)+"/per_surround/" + str(surround_angle)+"/" + "spikes_stderr.npy", np.array(std_spikes_err))
                    
    
                else:
                    np.save(save_folder+str(depth) + "/" + str(main_angle)+"/per_surround/" + str(surround_angle)+"/" + "spikes_avg.npy", np.array(sim_mean_spikes_same_angle[-1]))
                    
                    np.save(save_folder+str(depth) + "/" + str(main_angle)+"/per_surround/" + str(surround_angle)+"/" + "spikes_stderr.npy", np.array(std_spikes_err))
                    
def sineTimeResponse(spinet: SpikingNetwork, save_folder, angles, phases, neuron_id, fqcy, layer_id = 0, neuron_z = 0, max_depth = 16, thresh = 30, tuned_ori=False, n_simulation = 0, start_val = 100000, end_val = 2001000):
    sim_mean_spikes = []
    
    neur_ids = [2560]
    max_depth = max_depth
    z_neur = 0
    windows_upperBound = range(start_val,end_val,start_val)
    
    for depth in range(max_depth):
        for index_angle, main_angle in enumerate(angles):
            for phase in phases:
                angle_mean_spikes = [] 
                for sim in range(n_simulation):
                    spinet.load_statistics_4(fqcy, phase, layer_id, main_angle,  sim)
                    y_tot = []
                    y_tot_window = []
                    for neuron_id in neur_ids: 
                        for i in range(max_depth):
                            if(neuron_z == depth):
                                x, y = visualize_potentials(spinet, layer_id, z_neur, False)
                                temp = np.where(np.array(y)>=thresh)[1]
                                for e in temp:
                                    y_tot.append(x[0][e])
                                neuron_z+=1
                                z_neur+=1
                                if(neuron_z>=max_depth):
                                    break
                            else:
                                neuron_z+=1
                                z_neur+=1
                                if(neuron_z>=max_depth):
                                    break
                        neuron_z = 0
                    z_neur = 0
                    spike_windows = []
                    for idex, j in enumerate(windows_upperBound):
                        spikes = 0
                        for k in range(len(y_tot)):
                            if(idex==0 and y_tot[k] <= j):
                                spikes+=1
                            elif(idex!=0 and y_tot[k] > windows_upperBound[idex-1] and y_tot[k] <= j):
                                spikes+=1
                        
                        spike_windows.append(spikes)
                    
                    
                    angle_mean_spikes.append(spike_windows)
                sim_mean_spikes.append(np.mean(angle_mean_spikes, axis=0))
                
                std_spikes_err = []
                
                std_spikes = np.zeros((np.array(sim_mean_spikes)[-1].shape))
                for j in range(n_simulation):
                    std_spikes+=(angle_mean_spikes[j][0]-sim_mean_spikes[-1])**2
                
                std_spikes_err.append(np.sqrt(std_spikes/(n_simulation-1))/n_simulation)
    
                np.save(save_folder+str(depth) + "/" + str(main_angle)+"/per_surround/" + str(phase)+"/" + "spikes_avg.npy", np.array(sim_mean_spikes[-1]))
                
                np.save(save_folder+str(depth) + "/" + str(main_angle)+"/per_surround/" + str(phase)+"/" + "spikes_stderr.npy", np.array(std_spikes_err))

def tuningSelectivities(arr):
    angles = arr[:,0]
    frequencies = arr[:,1]
    phases = arr[:,2]
    id_angles = [0, 23, 45, 68, 90, 113, 135, 158]
    id_fq = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    id_ph = [0, 1, 2, 3, 4]
    ct_ag = np.zeros((8),dtype=np.float16)
    ct_fq = np.zeros((18),dtype=np.float16)
    ct_ph = np.zeros((5),dtype=np.float16)
    
    for ag in angles:
        ct_ag[np.where(id_angles==ag)]+=1
    for fq in frequencies:
        ct_fq[np.where(id_fq==fq)]+=1
    for ph in phases:
        ct_ph[np.where(id_ph==ph)]+=1
        if(ph==4):
            ct_ph[np.where(id_ph==np.float16(0))]+=1

    
    return ct_ag, ct_fq, ct_ph    
    
def average_over_orientations(spinet: SpikingNetwork, case, save_folder, n_thickness, angles, n_direction, n_simulation, neuron_id, layer_id = 0, neuron_z = 0, max_depth = 16, thresh = 30, tuned_ori=False, allthick = False, other_params=False, n_speed  =3):
    depth_complex = 16
    thresh_cells = n_simulation/2 + 1
    spinet.load_orientations()
    if case==0:
        for thickness in range(3,n_thickness+1):
            print("Thickness {}/{}".format(thickness,n_thickness))
            sim_mean_spikes = []
            sim_mean_lat = []
            sim_mean_td = []
            sim_mean_stat = []
            sim_mean_lat_tuned = []
            sim_mean_td_tuned = []
            sim_mean_stat_tuned = []
            for sim in range(n_simulation):
                print("Simulation {}/{}".format(sim+1,n_simulation))
                angle_mean_spikes = []
                angle_mean_lat = []
                angle_mean_td = []
                angle_mean_stat = []
                angle_mean_stat_tuned = []
                angle_mean_lat_tuned = []
                angle_mean_td_tuned = []
                for angle in angles:
                    for direction in range(n_direction):
                        print("Angle {}° in direction {}".format(angle, direction))
                        spinet.load_statistics_2(thickness, angle, direction, layer_id, sim)
                        y_tot = []
                        cells_number = []
                        for i in range(max_depth):
                            number_of_displays = len(spinet.stats)
                            if(not tuned_ori):
                                for seq in range(number_of_displays):
                                    temp = np.moveaxis(np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][neuron_z][1]["potential_train"]),-1,0)
                                    try:
                                        if( (temp[0]>=thresh).any() ):
                                            ok = True
                                            break
                                        else:
                                            ok = False
                                    except: 
                                        ok = False
                                condition = ok
                            else:
                                theta = np.array(spinet.neurons[layer_id][neuron_id+neuron_z].theta)
                                if(allthick):
                                    theta_3 = theta[2]
                                    theta = np.append(np.array(theta[0]),np.array(theta[1]))
                                    theta = np.append(theta, np.array(theta_3))
                                if(angle!=0):
                                    if(allthick):
                                        condition = ( (np.array(theta)==angle).any() ) or ( (np.array(theta)==-angle).any() )
                                    else:
                                        condition = ( (np.array(theta[thickness-1])==angle).any() ) or ( (np.array(theta[thickness-1])==-angle).any() )
                                elif(angle==0):
                                    if(allthick):
                                        condition = ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() )
                                    else:
                                        condition = ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() )
                            if(condition):
                                cells_number.append(neuron_z)
                                _, y = visualize_potentials(spinet, layer_id, neuron_z, False)
                                spikes_number = []
                                for value in range(number_of_displays):
                                    spikes_number.append(len(np.where(np.array(y[value])>=thresh)[0]))
                                y_tot.append(spikes_number)
                                print("Depth {0} approved (spiked)".format(neuron_z))
                                neuron_z+=1
                                if(neuron_z>=max_depth):
                                    break
                            else:
                                neuron_z+=1
                                if(neuron_z>=max_depth):
                                    break
                        neuron_z = 0
                        mean_val = []
                        for i in range(number_of_displays):
                            mean = 0
                            for j in range(len(y_tot)):
                                mean += y_tot[j][i]   
                            if(len(y_tot)!=0):
                                mean/=len(y_tot)
                            else:
                                mean = 0
                            mean_val.append(mean)
                        try:
                            avg0_ = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = max_depth, tuned_simple = True, allthick = True, without = False, untuned_simple = True)     #visualize_total_inhibition_evolution(spinet, layer_id, neuron_id, neuron_z,1)
                            avg1_ = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth_simple = max_depth, depth_complex = depth_complex, tuned_simple = True, allthick = True, untuned_complex=True)  #visualize_total_tdinhibition_evolution(spinet, layer_id, neuron_id, neuron_z)
                            avg2_ = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = max_depth, tuned_simple = True, allthick = True, without = False, untuned_simple = True, inhibition_type=0)
                        
                            avg0bis = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = max_depth, tuned_simple = True, allthick = True, without = False)  
                            avg1bis = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth_simple = max_depth, depth_complex = depth_complex, tuned_simple = True, allthick = True, untuned_complex=False)
                            avg2bis = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle, thickness = thickness, depth = max_depth, tuned_simple = True, allthick = True, without = False, inhibition_type=0)  

                            
                        except:
                            avg0_ = np.zeros((number_of_displays))
                            avg1_ = np.zeros((number_of_displays))
                            avg2_ = np.zeros((number_of_displays))

                            
                            avg0bis = np.zeros((number_of_displays))
                            avg1bis = np.zeros((number_of_displays))
                            avg2bis = np.zeros((number_of_displays))

                            
                        angle_mean_spikes.append(mean_val)
                        angle_mean_lat.append(avg0_)
                        angle_mean_td.append(avg1_)
                        angle_mean_stat.append(avg2_)
                        
                        angle_mean_stat_tuned.append(avg2bis)
                        angle_mean_lat_tuned.append(avg0bis)
                        angle_mean_td_tuned.append(avg1bis)
                
                sim_mean_spikes.append(np.mean(angle_mean_spikes, axis=0))
                sim_mean_lat.append(np.mean(angle_mean_lat, axis=0))
                sim_mean_td.append(np.mean(angle_mean_td, axis=0))
                sim_mean_stat.append(np.mean(angle_mean_stat, axis=0))
                sim_mean_lat_tuned.append(np.mean(angle_mean_lat_tuned, axis=0))
                sim_mean_td_tuned.append(np.mean(angle_mean_td_tuned, axis=0))
                sim_mean_stat_tuned.append(np.mean(angle_mean_stat_tuned, axis = 0))
                
            avg_spikes = np.mean(sim_mean_spikes, axis=0)
            avg_lat = np.mean(sim_mean_lat, axis=0)
            avg_td = np.mean(sim_mean_td, axis=0)
            avg_stat = np.mean(sim_mean_stat, axis=0)
            avg_lat_tuned = np.mean(sim_mean_lat_tuned, axis=0)
            avg_td_tuned = np.mean(sim_mean_td_tuned, axis=0)
            avg_stat_tuned = np.mean(sim_mean_stat_tuned, axis=0)
            
            std_spikes_err = []
            std_lat_err = []
            std_td_err = []
            std_stat_err = []
            std_lat_tuned_err = []
            std_td_tuned_err = []
            std_stat_tuned_err = []
            
            for i in range(number_of_displays):
                std_spikes = 0
                std_lat = 0
                std_td = 0
                std_stat = 0
                std_lat_tuned = 0
                std_td_tuned = 0
                std_stat_tuned = 0
                for j in range(n_simulation):
                    std_spikes+=(sim_mean_spikes[j][i]-avg_spikes[i])**2
                    std_lat+=(sim_mean_lat[j][i]-avg_lat[i])**2
                    std_td+=(sim_mean_td[j][i]-avg_td[i])**2
                    std_stat+=(sim_mean_stat_tuned[j][i]-avg_stat[i])**2
                    std_lat_tuned+=(sim_mean_lat_tuned[j][i]-avg_lat_tuned[i])**2
                    std_td_tuned+=(sim_mean_td_tuned[j][i]-avg_td_tuned[i])**2
                    std_stat_tuned+=(sim_mean_stat_tuned[j][i]-avg_stat_tuned[i])**2
                    
                std_spikes_err.append(np.sqrt(std_spikes/(n_simulation-1))/n_simulation)
                std_lat_err.append(np.sqrt(std_lat/(n_simulation-1))/n_simulation)
                std_td_err.append(np.sqrt(std_td/(n_simulation-1))/n_simulation)
                std_stat_err.append(np.sqrt(std_stat/(n_simulation-1))/n_simulation)
                std_lat_tuned_err.append(np.sqrt(std_lat_tuned/(n_simulation-1))/n_simulation)
                std_td_tuned_err.append(np.sqrt(std_td_tuned/(n_simulation-1))/n_simulation)
                std_stat_tuned_err.append(np.sqrt(std_stat_tuned/(n_simulation-1))/n_simulation)
                
                
            np.save(save_folder+str(thickness)+"/orientations_average/" + "spikes_avg.npy", np.array(avg_spikes))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "lat_avg.npy", np.array(avg_lat))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "td_avg.npy", np.array(avg_td))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "stat_avg.npy", np.array(avg_stat))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "lattuned_avg.npy", np.array(avg_lat_tuned))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "tdtuned_avg.npy", np.array(avg_td_tuned))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "stattuned_avg.npy", np.array(avg_stat_tuned))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "spikes_stderr.npy", np.array(std_spikes_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "lat_stderr.npy", np.array(std_lat_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "td_stderr.npy", np.array(std_td_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "lattuned_stderr.npy", np.array(std_lat_tuned_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "tdtuned_stderr.npy", np.array(std_td_tuned_err))
            np.save(save_folder+str(thickness)+"/orientations_average/" + "stattuned_stderr.npy", np.array(std_stat_tuned_err))
            
            print(avg_spikes)    
                
#PLOTTING FUNCTIONS FOR DIVERSE GRAPHS  

def averaged_graph2(load_folders, case, with_speed = False, n_thickness = 3, thickness = 1, speed = 0, n_speeds = 3, n_displays = 55, avg = False, start_space = 45, space = 3, n_simulation = 5):
            
    colors = ["k", "#FFA500", "#0000FF" ,"r", "m", "y", "g"]
    folder = load_folders
    thness = thickness
    if(case==0): #Plot spikes
        if(len(load_folders) > 1 or avg): #plot spikes of different trials (only lat, only td, both, no inhib etc.) ; works also for speeds.
            avg_spikes = []
            avg_std_err = []
            for i in range(len(load_folders)):
                avg_spikes.append([])
                avg_std_err.append([])
                if(with_speed):
                    for j in range(n_speeds):
                        avg_spikes[i].append([])
                        avg_std_err[i].append([])
            for thness in range(thickness,n_thickness+1):
                print(thness)
                # if(thness==2):
                #     continue
                for i, folder in enumerate(load_folders):
                    if(not with_speed):
                        avg_spikes[i].append(np.load(folder+str(thness)+"/orientations_average/" + "spikes_avg.npy"))
                        avg_std_err[i].append(np.load(folder+str(thness)+"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation))
                    else:
                        for j in range(n_speeds):
                            avg_spikes[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "spikes_avg.npy"))
                            avg_std_err[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation))
            
            if(with_speed):
                avg_spikes = np.mean(avg_spikes, axis = 2)
                avg_std_err = np.mean(avg_std_err, axis = 2)
            # print(avg_spikes)
            avg_spikes = np.mean(avg_spikes, axis=1)
            avg_std_err = np.mean(avg_std_err, axis=1)
            arge = np.arange(1,start_space+1)
            diff = n_displays - start_space
            for value in range(1, diff+1):
                arge = np.append(arge, start_space + space * value)
            arge2 = np.arange(1,31)
            fig = plt.figure(1, figsize = (20,20)) # 15, 8
            size = fig.get_size_inches() # size in pixels
            # print(size)
            ax = fig.add_axes([0,0,1,1])
            ax.set_xlabel("Bar length (in pixels)", fontsize = 40)
            # ax.set_ylabel("Normalized number of spikes", fontsize = 40)
            ax.set_ylabel("Normalized response", fontsize = 40)

            # ax.set_title("Surround suppression", fontsize=40)# for different cases")
            matplotlib.rc('xtick', labelsize=35) 
            matplotlib.rc('ytick', labelsize=35) 
            matplotlib.rcParams['legend.fontsize'] = 35
            max_sp = np.max(avg_spikes)
            avg_spikes = avg_spikes/max_sp
            avg_std_err =avg_std_err/ max_sp

            for i in range(len(load_folders)):
                #ax.plot(arge, avg_spikes[i], colors[i]+"-", label="folder number " + str(i))
                if(i==0):
                    ax.plot(arge, avg_spikes[i], colors[i], linewidth = 3.0 , label= "LOCAL")
                if(i==1):
                    ax.plot(arge, avg_spikes[i], colors[i], linewidth = 3.0 , label= "LOCAL + DIST")
                if(i==2):
                    ax.plot(arge, avg_spikes[i], colors[i], linewidth = 3.0 ,label="LOCAL + TD")
                if(i==3):
                    ax.plot(arge, avg_spikes[i], colors[i], linewidth = 3.0 ,label= "LOCAL + DIST + TD")
                if(i==4):
                    ax.plot(arge, avg_spikes[i], colors[i], linewidth = 3.0 ,label="LAT patch")
                if(i==5):
                    ax.plot(arge, avg_spikes[i], colors[i], linewidth = 3.0 ,label="TD patch")
                
                

                for val in range(len(arge)):
                    ax.plot( [arge[val], arge[val]], [np.maximum(avg_spikes[i][val]-avg_std_err[i][val],0),avg_spikes[i][val]+ avg_std_err[i][val]],colors[i], linestyle="--")
            new_arge = range(1,76,10)
            #new_arge = range(1,31,2)
            #new_arge = range(1,76,2)
            plt.xticks(new_arge)
            # suppr_percentage = suppression_metric(avg_spikes[0])
            # suppr_percentage2 = suppression_metric(avg_spikes[1])
            # suppr_percentage3 = suppression_metric(avg_spikes[2])
            # suppr_percentage4 = suppression_metric(avg_spikes[3])
            # print(suppr_percentage4)
            # plt.axvline(x = np.argmax(avg_spikes[1]), color = '#FFA500', linestyle = '-')
            # plt.axvline(x = np.argmax(avg_spikes[3]), color = 'r', linestyle = '-')
            # plt.axhline(y = avg_spikes[0], color = 'k', linestyle = '-')
            # plt.axhline(y = avg_spikes[1], color = 'y', linestyle = '-')
            print(avg_spikes)
            #plt.axhline(y = avg_spikes[1][-1], color = 'r', linestyle = '-')
            # print(suppr_percentage)
            # print(suppr_percentage2)
            # print(suppr_percentage3)
            y_ids = [0, 0.2, 0.4, 0.6, 0.8, 1]
            ax.set_yticks(y_ids)
            #print(suppr_percentage3)
            # ax.text(10, max(avg_spikes[0])/2, "Suppression up to " + str(int(suppr_percentage)) + "%", fontsize = 42)
#             ax.text(20, max(avg_spikes[1])/4, "(Lateral alone) Suppression up to " + str(int(suppr_percentage2)) + "%", fontsize = 42)
# #31 (ax.text())
#             ax.text(20, max(avg_spikes[2])/6, "(Top down alone) Suppression up to " + str(int(suppr_percentage3)) + "%", fontsize = 42)

            #fig.legend(loc='center', bbox_to_anchor=(0.35, 0, 0.93, 2.22))
            # plt.legend(loc=1,bbox_to_anchor=(0, 0, 0.98, 0.95))
            # plt.legend(loc=1,bbox_to_anchor=(0, 0, 0.21, 1.45))
            # ax.legend(bbox_to_anchor=(-0.05,1.02))

            plt.legend(loc=5, bbox_to_anchor=[1,0.7])
            # plt.legend(loc=1)

            # 'best' (Axes only)
            # 0
            # 'upper right'
            # 1
            # 'upper left'
            # 2
            # 'lower left'
            # 3
            # 'lower right'
            # 4
            # 'right'
            # 5
            # 'center left'
            # 6
            # 'center right'
            # 7
            # 'lower center'
            # 8
            # 'upper center'
            # 9
            # 'center'
            # 10
            plt.show()
            # print(100 - avg_spikes[0][0:30]*100/avg_spikes[1][0:30])

        else:
            avg_spikes = []
            avg_std_err = []
            
            arge = np.arange(1,start_space+1)
            diff = n_displays- start_space
            for value in range(1, diff+1):
                arge = np.append(arge, start_space + space * value)
                
            fig = plt.figure(1,figsize=(15,8))
            ax = fig.add_axes([0,0,1,1])
            ax.set_xlabel("Bar length (in pixels)", fontsize = 40)
            ax.set_ylabel("Normalized response", fontsize = 40)
            # ax.set_title("Evolution of spikes on average for angles [0°, 23°, 45°, 68°, 90°]")
            
            if(with_speed):
                for j in range(n_speeds):
                    avg_spikes.append([])
                    avg_std_err.append([])
                    
                    avg_spikes[j].append(np.load(load_folders[0]+str(thickness)+"/speeds/" + str(j) +"/orientations_average/" + "spikes_avg.npy"))
                    avg_std_err[j].append(np.load(load_folders[0]+str(thickness)+"/speeds/" + str(j) +"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation))
                
                    ax.plot(arge,avg_spikes[j], colors[j]+"-", label="speed number " + str(j))
                    for val in range(len(arge)):
                        ax.plot( [arge[val], arge[val]], [avg_spikes[j][val]-avg_std_err[j][val],avg_spikes[j][val]+ avg_std_err[j][val]],colors[j]+"--")
            else:
                avg_spikes.append(np.load(load_folders[0]+str(thickness)+"/orientations_average/" + "spikes_avg.npy"))
                avg_std_err.append(np.load(load_folders[0]+str(thickness)+"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation))
                max_sp = np.max(avg_spikes)
                avg_spikes = np.array(avg_spikes).flatten() / max_sp
                avg_std_err = np.array(avg_std_err).flatten() / max_sp
                ax.plot(arge,avg_spikes, colors[0]+"-", linewidth = 6.0)# + str(1))
                for val in range(len(arge)):
                    ax.plot( [arge[val], arge[val]], [avg_spikes[val]-avg_std_err[val],avg_spikes[val]+ avg_std_err[val]],colors[0]+"--")
            new_arge = range(1,76,10)
            plt.xticks(new_arge)
            suppr_percentage = suppression_metric(avg_spikes)
            # ax.text(31, max(avg_spikes)/2, "Suppression up to " + str(int(suppr_percentage)) + "%", fontsize = 42)
            # fig.legend(loc='center', bbox_to_anchor=(0.35, 0, 0.7, 0.75))
            plt.show()
    elif(case==1): #plot inhibitions
        if(len(load_folders) > 1 or avg): #plot spikes of different trials (only lat, only td, both, no inhib etc.) ; works also for speeds.
            avg_lat = []
            avg_td = []
            avg_stat = []
            
            avg_lat_err = []
            avg_td_err = []
            avg_stat_err = []
            for i in range(len(load_folders)):
                avg_lat.append([])
                avg_td.append([])
                avg_stat.append([])
                
                avg_lat_err.append([])
                avg_td_err.append([])
                avg_stat_err.append([])
                
                if(with_speed):
                    for j in range(n_speeds):
                        avg_lat[i].append([])
                        avg_td[i].append([])
                        avg_stat[i].append([])
                        
                        avg_lat_err[i].append([])
                        avg_td_err[i].append([])
                        avg_stat_err[i].append([])
                        
            for thness in range(1,n_thickness+1):
                for i, folder in enumerate(load_folders):
                    if(not with_speed):
                        print(folder)
                        avg_lat[i].append(np.load(folder+str(thness)+"/orientations_average/" + "lat_avg.npy"))
                        avg_td[i].append(np.load(folder+str(thness)+"/orientations_average/" + "td_avg.npy"))
                        avg_stat[i].append(np.load(folder+str(thness)+"/orientations_average/" + "stat_avg.npy"))
                        
                        avg_lat_err[i].append(np.load(folder+str(thness)+"/orientations_average/" + "lat_stderr.npy")* np.sqrt(n_simulation))
                        avg_td_err[i].append(np.load(folder+str(thness)+"/orientations_average/" + "td_stderr.npy")* np.sqrt(n_simulation))
                        avg_stat_err[i].append(np.load(folder+str(thness)+"/orientations_average/" + "stat_stderr.npy")* np.sqrt(n_simulation))
                    else:
                        for j in range(n_speeds):
                            
                            avg_lat[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "lat_avg.npy"))
                            avg_td[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "td_avg.npy"))
                            avg_stat[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "stat_avg.npy"))
                            
                            avg_lat_err[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "lat_stderr.npy")* np.sqrt(n_simulation))
                            avg_td_err[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "td_stderr.npy")* np.sqrt(n_simulation))
                            avg_stat_err[i][j].append(np.load(folder+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "stat_stderr.npy")* np.sqrt(n_simulation))
            
            if(with_speed):
                avg_lat = np.mean(avg_lat, axis = 2)
                avg_td = np.mean(avg_td, axis = 2)
                avg_stat = np.mean(avg_stat, axis=2)
                
                avg_lat_err = np.mean(avg_lat_err, axis = 2)
                avg_td_err = np.mean(avg_td_err, axis = 2)
                avg_stat_err = np.mean(avg_stat_err, axis=2)
                
            avg_lat = np.mean(avg_lat, axis = 0)
            avg_td = np.mean(avg_td, axis = 0)
            avg_stat = np.mean(avg_stat, axis=0)
            
            avg_lat_err = np.mean(avg_lat_err, axis = 0)
            avg_td_err = np.mean(avg_td_err, axis = 0)
            avg_stat_err = np.mean(avg_stat_err, axis=0)
            
            arge = np.arange(1,start_space+1)
            diff = n_displays - start_space
            for value in range(1, diff+1):
                arge = np.append(arge, start_space + space * value)
            fig = plt.figure(1)
            ax = fig.add_axes([0,0,1,1])
            ax.set_xlabel("Length of oriented bars (in pixels)")
            ax.set_ylabel("Amount of inhibition")
            ax.set_title("Evolution of spikes for different cases")
            for i in range(len(load_folders)):
                ax.plot(arge, avg_lat[i], colors[i]+"-", label="folder number " + str(i))
                ax.plot(arge, avg_td[i], colors[i]+"o")
                ax.plot(arge, avg_stat[i], colors[i]+"*")

                for val in range(len(arge)):
                    ax.plot( [arge[val], arge[val]], [avg_lat[i][val]-avg_lat_err[i][val],avg_lat[i][val]+ avg_lat_err[i][val]],colors[i]+"--")
                    ax.plot( [arge[val], arge[val]], [avg_td[i][val]-avg_td_err[i][val],avg_td[i][val]+ avg_td_err[i][val]],colors[i]+"--")
                    ax.plot( [arge[val], arge[val]], [avg_stat[i][val]-avg_stat_err[i][val],avg_stat[i][val]+ avg_stat_err[i][val]],colors[i]+"--")
            plt.xticks(arge)
            fig.legend(loc='center', bbox_to_anchor=(0.35, 0, 0.7, 0.75))
            plt.show()
            
        else:
            print("yo")
            avg_lat = []
            avg_td = []
            avg_stat = []
            
            avg_lat_err = []
            avg_td_err = []
            avg_stat_err = []
            
            arge = np.arange(1,start_space+1)
            diff = n_displays - start_space
            for value in range(1, diff+1):
                arge = np.append(arge, start_space + space * value)
            fig = plt.figure(1)
            ax = fig.add_axes([0,0,1,1])
            ax.set_xlabel("Length of oriented bars (in pixels)")
            ax.set_ylabel("Average number of spikes")
            ax.set_title("Evolution of spikes for different cases")
            
            if(with_speed):
                for j in range(n_speeds):
                    avg_lat.append([])
                    avg_td.append([])
                    avg_stat.append([])
                    
                    avg_lat_err.append([])
                    avg_td_err.append([])
                    avg_stat_err.append([])
                    
                    avg_lat[j].append(np.load(load_folders[0]+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "lat_avg.npy"))
                    avg_td[j].append(np.load(load_folders[0]+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "td_avg.npy"))
                    avg_stat[j].append(np.load(load_folders[0]+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "stat_avg.npy"))
                    
                    avg_lat_err[j].append(np.load(load_folders[0]+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "lat_stderr.npy")* np.sqrt(n_simulation))
                    avg_td_err[j].append(np.load(load_folders[0]+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "td_stderr.npy")* np.sqrt(n_simulation))
                    #avg_stat_err[j].append(np.load(load_folders[0]+str(thness)+"/speeds/" + str(j) +"/orientations_average/" + "stat_stderr.npy")* np.sqrt(n_simulation))
                    
                    ax.plot(arge, avg_lat[j], colors[j]+"-", label="speed number " + str(i))
                    ax.plot(arge, avg_td[j], colors[j]+"o")
                    ax.plot(arge, avg_stat[j], colors[j]+"*")

                    for val in range(len(arge)):
                        ax.plot( [arge[val], arge[val]], [avg_lat[j][val]-avg_lat_err[j][val],avg_lat[j][val]+ avg_lat_err[j][val]],colors[j]+"--")
                        ax.plot( [arge[val], arge[val]], [avg_td[j][val]-avg_td_err[j][val],avg_td[j][val]+ avg_td_err[j][val]],colors[j]+"--")
                        #ax.plot( [arge[val], arge[val]], [avg_stat[j][val]-avg_stat_err[j][val],avg_stat[j][val]+ avg_stat_err[j][val]],colors[j]+"--")
                            
            else:
                avg_lat.append(np.load(load_folders[0]+str(thness)+"/orientations_average/" + "lat_avg.npy"))
                avg_td.append(np.load(load_folders[0]+str(thness)+"/orientations_average/" + "td_avg.npy"))
                avg_stat.append(np.load(load_folders[0]+str(thness)+"/orientations_average/" + "stat_avg.npy"))
                
                avg_lat_err.append(np.load(load_folders[0]+str(thness)+"/orientations_average/" + "lat_stderr.npy")* np.sqrt(n_simulation))
                avg_td_err.append(np.load(load_folders[0]+str(thness)+"/orientations_average/" + "td_stderr.npy")* np.sqrt(n_simulation))
                #avg_stat_err.append(np.load(load_folders[0]+str(thness)+"/orientations_average/" + "stat_stderr.npy")* np.sqrt(n_simulation))
                
                avg_lat = np.array(avg_lat).flatten()
                avg_td = np.array(avg_td).flatten()
                avg_stat = np.array(avg_stat).flatten()
                avg_lat_err = np.array(avg_lat_err).flatten()
                avg_td_err = np.array(avg_td_err).flatten()
                ax.plot(arge, avg_lat, colors[0]+"-", label="thickness = " + str(thness))
                ax.plot(arge, avg_td, colors[0]+"o")
                ax.plot(arge, avg_stat, colors[0]+"*")
                print(avg_td)
                for val in range(len(arge)):
                    ax.plot( [arge[val], arge[val]], [avg_lat[val]-avg_lat_err[val],avg_lat[val]+ avg_lat_err[val]],colors[0]+"--")
                    ax.plot( [arge[val], arge[val]], [avg_td[val]-avg_td_err[val],avg_td[val]+ avg_td_err[val]],colors[0]+"--")
                    #ax.plot( [arge[val], arge[val]], [avg_stat[val]-avg_stat_err[val],avg_stat[val]+ avg_stat_err[val]],colors[0]+"--")
    elif(case==2): #plot all thicknesses in one plot ; not for speeds.
        avg_spikes = []
        avg_std_err = []
        for i in range(1, n_thickness+1):
            avg_spikes.append([])
            avg_std_err.append([])
            
            avg_spikes[i-1].append(np.load(load_folders[0]+str(i)+"/orientations_average/" + "spikes_avg.npy"))
            avg_std_err[i-1].append(np.load(load_folders[0]+str(i)+"/orientations_average/" + "spikes_stderr.npy") * np.sqrt(n_simulation))
        print(np.array(avg_spikes).shape)
        print(avg_spikes)

        arge = np.arange(1,start_space+1)
        diff = n_displays - start_space
        for value in range(1, diff+1):
            arge = np.append(arge, start_space + space * value)
        fig = plt.figure(1)
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlabel("Length of oriented bars (in pixels)")
        ax.set_ylabel("Average number of spikes")
        ax.set_title("Evolution of spikes for different cases")
        for i in range(n_thickness):
            ax.plot(arge, avg_spikes[i][0], colors[i]+"-", label="thickness number " + str(i))
            for val in range(len(arge)):
                ax.plot( [arge[val], arge[val]], [avg_spikes[i][0][val]-avg_std_err[i][0][val],avg_spikes[i][0][val]+ avg_std_err[i][0][val]],colors[i]+"--")
        plt.xticks(arge)
        fig.legend(loc='center', bbox_to_anchor=(0.35, 0, 0.7, 0.75))
        plt.show()