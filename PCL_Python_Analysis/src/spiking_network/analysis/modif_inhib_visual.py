#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 10:26:04 2022

@author: comsee
"""


import matplotlib.pyplot as plt
import numpy as np
import matplotlib

from src.spiking_network.network.neuvisys import SpikingNetwork

def visualize_total_inhibition_evolution2(spinet: SpikingNetwork, layer_id, neuron_id, angle, thickness = 1, depth = 64, tuned_simple = False, actual_orientation_sequence = False, actual_sequence_angle = [0, 180], untuned_angles_for_seq = False, allthick = False, without = False, without_angle = [0,180], take_off_ori = False, take_off_angle=[0, 180], untuned_simple = False, inhibition_type = 1):
    #tuned_simple : to pick only the simple cells of the neuron i'm looking @, that are tuned to the specific orientation.
    #without : to say that i don't want an angle in the simple cells i'm looking at. USELESS now since i no longer take oris per thickness
    # -> to leave without at False.
    #allthick : to always leave at true since i have an orientation only regardless of the thickness.
    #actual_orientation_sequence: If I want to pick simple cells that are tuned to the actual orientation of the sequence i'm showing.
    #-> take_off_ori, take_off_angle : when i had multiple orientations per simple cells (for each thickness), again. USELESS now
    #untuned_angle_for_seq : again, useless now since i no longer have multiple orientation per simple cells.
    #untuned_simple: if i want to check the amount of inhibition received by all the depths regardless of their tuning
    
    lateral_weights = spinet.neurons[layer_id][neuron_id].weights_li
    number_of_displays = len(spinet.stats)
    sum_weights_lat_total = [[]]
    for seq in range(number_of_displays):
        for d in range(depth):
            try: 
                # print(inhibition_type)
                # print(np.array(spinet.stats[seq][str(
                #     seq)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][inhibition_type]).shape)
                sum_weights_lat = np.zeros((np.array(spinet.stats[seq][str(
                    seq)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][inhibition_type][d])).shape)
                tuned_neurs = 0
                for r in range(0, depth):
                    theta = np.array(spinet.neurons[layer_id][r].theta)
                    if(tuned_simple):
                        if(allthick):
                            theta_3 = theta[2]
                            theta = np.append(np.array(theta[0]),np.array(theta[1]))
                            theta = np.append(theta, np.array(theta_3))
                        if(angle ==-1):
                            condition = True
                        elif(angle!=0):
                            if(allthick and not without):
                                condition = ( (np.array(theta)==angle).any() ) or ( (np.array(theta)==-angle).any() )
                            elif(allthick and without):
                                condition = ( ( (np.array(theta)==angle).any() ) or ( (np.array(theta)==-angle).any() ) ) and ( ( (np.array(theta)!=without_angle[0]).all() ) and ( (np.array(theta)!=without_angle[1]).all() ) ) 
                            elif(not without):
                                condition = ( (np.array(theta[thickness-1])==angle).any() ) or ( (np.array(theta[thickness-1])==-angle).any() )
                            else:
                                condition = ( ( (np.array(theta[thickness-1])==angle).any() ) or ( (np.array(theta[thickness-1])==-angle).any() ) ) and ( ( (np.array(theta[thickness-1])!=without_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=without_angle[1]).all() ))
    
                        elif(angle==0):
                            if(allthick and not without):
                               condition = ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() )
                            elif(allthick and without):
                                condition = ( ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() ) ) and ( ( (np.array(theta)!=without_angle[0]).all() ) and ( (np.array(theta)!=without_angle[1]).all() ) )
                            elif(not without):
                                condition = ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() )
                            else:
                                condition = ( ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() ) ) and ( ( (np.array(theta[thickness-1])!=without_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=without_angle[1]).all() ) )
                        if(condition):
                            tuned_neurs+=1
                            sum_weights_lat += np.array(spinet.stats[seq][str(
                                seq)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][inhibition_type][d])
                    elif(actual_orientation_sequence): 
                        if(allthick):
                            theta_3 = theta[2]
                            theta = np.append(np.array(theta[0]),np.array(theta[1]))
                            theta = np.append(theta, np.array(theta_3))
                        if(angle ==-1):
                            condition = True
                        else:
                            if(allthick and not take_off_ori):
                                condition = ( ( (np.array(theta)==actual_sequence_angle[0]).any() ) or ( (np.array(theta)==actual_sequence_angle[1]).any() ) )
                            elif(allthick and take_off_ori):
                                condition = ( ( (np.array(theta)==actual_sequence_angle[0]).any() ) or ( (np.array(theta)==actual_sequence_angle[1]).any() ) ) and ( ( (np.array(theta)!=take_off_angle[0]).all() ) and ( (np.array(theta)!=take_off_angle[1]).all() ) )
                            elif(not take_off_ori):
                                condition = ( (np.array(theta[thickness-1])==actual_sequence_angle[0]).any() ) or ( (np.array(theta[thickness-1])==actual_sequence_angle[1]).any() )    
                            else:
                                condition = ( (np.array(theta[thickness-1])==actual_sequence_angle[0]).any() ) or ( (np.array(theta[thickness-1])==actual_sequence_angle[1]).any() ) and ( ( (np.array(theta[thickness-1])!=take_off_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=take_off_angle[1]).all() ) )
                        if(condition):
                            tuned_neurs+=1
                            sum_weights_lat += np.array(spinet.stats[seq][str(
                                seq)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][inhibition_type][d])
                    elif(untuned_angles_for_seq):
                        if(allthick):
                            theta_3 = theta[2]
                            theta = np.append(np.array(theta[0]),np.array(theta[1]))
                            theta = np.append(theta, np.array(theta_3))
                        if(angle ==-1):
                            condition = True
                        else:
                            if(angle!=0):
                                verif_angle = [angle, -angle]
                            else:
                                verif_angle = [0, 180]
                            if(allthick):
                                condition = ( ( (np.array(theta)!=actual_sequence_angle[0]).all() ) and ( (np.array(theta)!=actual_sequence_angle[1]).all() ) ) and ( ( (np.array(theta)!=verif_angle[0]).all() ) and ( (np.array(theta)!=verif_angle[1]).all() ) )
                            else:
                                condition = ( ( (np.array(theta[thickness-1])!=actual_sequence_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=actual_sequence_angle[1]).all() ) ) and ( ( (np.array(theta[thickness-1])!=verif_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=verif_angle[1]).all() ) )
                        if(condition):
                            tuned_neurs+=1
                            sum_weights_lat += np.array(spinet.stats[seq][str(
                                seq)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][inhibition_type][d])
                    else:
                        tuned_neurs+=1
                        sum_weights_lat += np.array(spinet.stats[seq][str(
                            seq)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][inhibition_type][d])
                sum_weights_lat /= tuned_neurs
                sum_weights_lat_total[seq].append(sum_weights_lat)
                # print(sum_weights_lat)
            except: 
                try:
                    sum_weights_lat_total[seq].append(np.zeros((np.array(spinet.stats[number_of_displays-1][str(
                        number_of_displays-1)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][inhibition_type][d])).shape))
                except:
                    pass
                
        if(seq<number_of_displays-1):
            sum_weights_lat_total.append([])
    layout = np.load(spinet.path + "weights/layout_" + str(layer_id) + ".npy")
    # print(np.array(sum_weights_lat_total).shape)
    if(len(lateral_weights) != 0):

        depth_avg = []
        wi = 0
        range_ = spinet.conf["neuronInhibitionRange"]
        range_x = range_[0]
        range_y = range_[1]
        neur_ids = [2560] #[1920, 1984, 2048, 2496, 2560, 2624, 3072, 3136, 3200]
        for neuron_id in neur_ids:
            for d in range(depth):
                x_neur = np.where(layout == neuron_id)[0][0]
                y_neur = np.where(layout == neuron_id)[1][0]
                avg = []
                theta = np.array(spinet.neurons[layer_id][neuron_id+d].theta)
                if(allthick):
                    theta_3 = theta[2]
                    theta = np.append(np.array(theta[0]),np.array(theta[1]))
                    theta = np.append(theta, np.array(theta_3))
                if(untuned_simple):
                    condition = True
                elif(angle!=0):
                    if(allthick and not without):
                        condition = ( ( (np.array(theta)==angle).any() ) or ( (np.array(theta)==-angle).any() ) )
                    elif(allthick and without):
                        condition = ( ( (np.array(theta)==angle).any() ) or ( (np.array(theta)==-angle).any() ) ) and ( ( (np.array(theta)!=without_angle[0]).all() ) and ( (np.array(theta)!=without_angle[1]).all() ) ) 
                    elif(not without):
                        condition = ( (np.array(theta[thickness-1])==angle).any() ) or ( (np.array(theta[thickness-1])==-angle).any() )
                    else:
                        condition = ( ( (np.array(theta[thickness-1])==angle).any() ) or ( (np.array(theta[thickness-1])==-angle).any() ) ) and ( ( (np.array(theta[thickness-1])!=without_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=without_angle[1]).all() ))
    
                elif(angle==0):
                    if(allthick and not without):
                       condition = ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() )
                    elif(allthick and without):
                        condition = ( ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() ) ) and ( ( (np.array(theta)!=without_angle[0]).all() ) and ( (np.array(theta)!=without_angle[1]).all() ) )
                    elif(not without):
                        condition = ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() )
                    else:
                        condition = ( ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() ) ) and ( ( (np.array(theta[thickness-1])!=without_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=without_angle[1]).all() ) )
    
                if(condition):
                    for sequence in range(number_of_displays):
                        it = 0
                        wi = 0
                        for value_x in range(x_neur-range_x, x_neur+range_x+1):
                            for value_y in range(y_neur-range_y, y_neur+range_y+1):
                                if((it < len(sum_weights_lat_total[sequence][d])) and (value_x != x_neur or value_y != y_neur)):
                                    value = sum_weights_lat_total[sequence][d][it]
                                    it += 1
                                if((value_x != x_neur or value_y != y_neur) and value_x >= 0 and value_y >= 0 and value_x < (len((spinet.conf["layerPatches"])[0][0]) * (spinet.conf["layerSizes"])[0][0]) and value_y < (len((spinet.conf["layerPatches"])[0][1]) * (spinet.conf["layerSizes"])[0][1])):
                                    wi += value
                        avg.append(wi)
                    depth_avg.append(avg)
        arge = np.arange(1, len(np.array(range(0, number_of_displays)))+1)
        depth_avg = np.sum(depth_avg, axis = 0)
        #print(np.max(depth_avg))
        #arge[0] = 1
        if(type(depth_avg)!=np.ndarray):
            depth_avg = np.zeros((number_of_displays))
        # plt.plot(arge, depth_avg, 'r-',
        #                   label="total amount of lateral inhibition")
        # plt.xlabel("length of grating (in pixels)")
        # plt.ylabel("amount of inhibition received")
        # plt.title("evolution of lateral inhibition by grating's length")
        # plt.show()
    return np.array(depth_avg)

def visualize_total_tdinhibition_evolution2(spinet: SpikingNetwork, layer_id, neuron_id, angle, surr_angle, thickness = 1, depth_simple = 64, depth_complex = 16, tuned_simple = False, actual_orientation_sequence = False, actual_sequence_angle = [0, 180], untuned_angles_for_seq = False, allthick = False, without = False, without_angle = [0,180], take_off_ori = False, take_off_angle=[0, 180], untuned_complex = False):
    topdown_weights = spinet.neurons[layer_id][neuron_id].weights_tdi
    number_of_displays = len(spinet.stats)
    sum_weights_td_total = [[]]
    # surr_angle = angle
    for seq in range(number_of_displays):
        sum_weights_tdi = np.zeros((np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][0][6]["top_down_weights"])).shape)
        tuned_neurs=0
        g = 0 #1408
        for r in range(0, depth_simple*9): #:1408,1408+depth_simple)
            # print(r)
        # for r in range(0, depth_simple*9):
            try:
                # print(r)
                theta = np.array(spinet.neurons[layer_id][g].theta)
                g+=1
                if(g==depth_simple):
                    g = 0
                if(tuned_simple):
                    if(allthick):
                        theta_3 = theta[2]
                        theta = np.append(np.array(theta[0]),np.array(theta[1]))
                        theta = np.append(theta, np.array(theta_3))
                    if(angle ==-1):
                        condition = True
                    elif(angle!=0):
                        if(allthick and not without):
                            # print("yo??")
                            condition = ( (np.array(theta)==angle).any() ) or ( (np.array(theta)==-angle).any() )
                        elif(allthick and without):
                            condition = ( ( (np.array(theta)==angle).any() ) or ( (np.array(theta)==-angle).any() ) ) and ( ( (np.array(theta)!=without_angle[0]).all() ) and ( (np.array(theta)!=without_angle[1]).all() ) ) 
                        elif(not without):
                            condition = ( (np.array(theta[thickness-1])==angle).any() ) or ( (np.array(theta[thickness-1])==-angle).any() )
                        else:
                            condition = ( ( (np.array(theta[thickness-1])==angle).any() ) or ( (np.array(theta[thickness-1])==-angle).any() ) ) and ( ( (np.array(theta[thickness-1])!=without_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=without_angle[1]).all() ))

                    elif(angle==0):
                        if(allthick and not without):
                           condition = ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() )
                        elif(allthick and without):
                            condition = ( ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() ) ) and ( ( (np.array(theta)!=without_angle[0]).all() ) and ( (np.array(theta)!=without_angle[1]).all() ) )
                        elif(not without):
                            condition = ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() )
                        else:
                            condition = ( ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() ) ) and ( ( (np.array(theta[thickness-1])!=without_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=without_angle[1]).all() ) )
                    if(condition):
                        tuned_neurs+=1
                        sum_weights_tdi += np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][r][6]["top_down_weights"])
                elif(actual_orientation_sequence): 
                    if(allthick):
                        theta_3 = theta[2]
                        theta = np.append(np.array(theta[0]),np.array(theta[1]))
                        theta = np.append(theta, np.array(theta_3))
                    if(angle ==-1):
                        condition = True
                    else:
                        if(allthick and not take_off_ori):
                            condition = ( ( (np.array(theta)==actual_sequence_angle[0]).any() ) or ( (np.array(theta)==actual_sequence_angle[1]).any() ) )
                        elif(allthick and take_off_ori):
                            condition = ( ( (np.array(theta)==actual_sequence_angle[0]).any() ) or ( (np.array(theta)==actual_sequence_angle[1]).any() ) ) and ( ( (np.array(theta)!=take_off_angle[0]).all() ) and ( (np.array(theta)!=take_off_angle[1]).all() ) )
                        elif(not take_off_ori):
                            condition = ( (np.array(theta[thickness-1])==actual_sequence_angle[0]).any() ) or ( (np.array(theta[thickness-1])==actual_sequence_angle[1]).any() )    
                        else:
                            condition = ( (np.array(theta[thickness-1])==actual_sequence_angle[0]).any() ) or ( (np.array(theta[thickness-1])==actual_sequence_angle[1]).any() ) and ( ( (np.array(theta[thickness-1])!=take_off_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=take_off_angle[1]).all() ) )

                    if(condition):
                        tuned_neurs+=1
                        sum_weights_tdi += np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][r][6]["top_down_weights"])
                elif(untuned_angles_for_seq):
                    if(allthick):
                        theta_3 = theta[2]
                        theta = np.append(np.array(theta[0]),np.array(theta[1]))
                        theta = np.append(theta, np.array(theta_3))
                    if(angle ==-1):
                        condition = True
                    else:
                        if(angle!=0):
                            verif_angle = [angle, -angle]
                        else:
                            verif_angle = [0, 180]
                        if(allthick):
                            condition = ( ( (np.array(theta)!=actual_sequence_angle[0]).all() ) and ( (np.array(theta)!=actual_sequence_angle[1]).all() ) ) and ( ( (np.array(theta)!=verif_angle[0]).all() ) and ( (np.array(theta)!=verif_angle[1]).all() ) )
                        else:
                            condition = ( ( (np.array(theta[thickness-1])!=actual_sequence_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=actual_sequence_angle[1]).all() ) ) and ( ( (np.array(theta[thickness-1])!=verif_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=verif_angle[1]).all() ) )
                    if(condition):
                        tuned_neurs+=1
                        sum_weights_tdi += np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][r][6]["top_down_weights"])
                else:
                    tuned_neurs+=1
                    sum_weights_tdi += np.array(spinet.stats[seq][str(seq)][layer_id][str(layer_id)][r][6]["top_down_weights"])
                    #print(r)

            except:
                ValueError
        if(tuned_neurs!=0):
            sum_weights_tdi /= tuned_neurs
            #print(np.sum(np.sum(sum_weights_tdi,axis=1)))
        #print(tuned_neurs)
        sum_weights_td_total[seq].append(sum_weights_tdi)
        if(seq<number_of_displays-1):
            sum_weights_td_total.append([])
        # print(np.array(sum_weights_td_total).shape)
    if(len(topdown_weights) != 0):
        neur_avg = [[]]
        depth = spinet.l_shape[layer_id+1][2]
        y_i=0
        x_i=0
        start_init_neur = 0
        true_surr_angle = surr_angle
        surr_angle = angle
        # print("angle = {}, surr_angle = {}".format(angle, surr_angle))
        # print(surr_angle)
        init_neur = spinet.neurons[layer_id][neuron_id].out_connections[start_init_neur]
        iter_ = 0
        for i in range(sum_weights_tdi.shape[0]*sum_weights_tdi.shape[1]):
            # print(sum_weights_tdi.shape[0]*sum_weights_tdi.shape[1])
            # print(sum_weights_tdi.shape)
            avg = []
            if( (i)%16==0 and i!=(sum_weights_tdi.shape[0]*sum_weights_tdi.shape[1] - 1) and i!=0): #16
                start_init_neur+=16
                #print(start_init_neur)
                init_neur=spinet.neurons[layer_id][neuron_id].out_connections[start_init_neur]
                iter_ = 0
                # print(start_init_neur)
            # print(init_neur+i)
            theta = np.array(spinet.neurons[layer_id+1][init_neur+iter_].theta)
            iter_ +=1
            if(allthick):
                theta_3 = theta[2]
                theta = np.append(np.array(theta[0]),np.array(theta[1]))
                theta = np.append(theta, np.array(theta_3))
            if(untuned_complex):
                condition = True
            elif(surr_angle!=0):
                if(allthick and not without):
                    condition = ( (np.array(theta)==surr_angle).any() ) or ( (np.array(theta)==-surr_angle).any() )
                if(allthick and without):
                    condition = ( ( (np.array(theta)==surr_angle).any() ) or ( (np.array(theta)==-surr_angle).any() ) ) and ( ( (np.array(theta)!=without_angle[0]).all() ) and ( (np.array(theta)!=without_angle[1]).all() ) ) 
                elif(not without):
                    condition = ( (np.array(theta[thickness-1])==surr_angle).any() ) or ( (np.array(theta[thickness-1])==-surr_angle).any() )
                else:
                    condition = ( ( (np.array(theta[thickness-1])==surr_angle).any() ) or ( (np.array(theta[thickness-1])==-surr_angle).any() ) ) and ( ( (np.array(theta[thickness-1])!=without_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=without_angle[1]).all() ))

            elif(surr_angle==0):
                if(allthick and not without):
                   condition = ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() )
                elif(allthick and without):
                    condition = ( ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() ) ) and ( ( (np.array(theta)!=without_angle[0]).all() ) and ( (np.array(theta)!=without_angle[1]).all() ) )
                elif(not without):
                    condition = ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() )
                else:
                    condition = ( ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() ) ) and ( ( (np.array(theta[thickness-1])!=without_angle[0]).all() ) and ( (np.array(theta[thickness-1])!=without_angle[1]).all() ) )
            if((i)%depth==0 and i!=0 and i!=(sum_weights_tdi.shape[0]*sum_weights_tdi.shape[1] - 1)):
                y_i+=1
                x_i = 0
            # print("x_i = {} and y_i = {}".format(x_i,y_i))
            # print("actual angle = {}, surr_angle = {}, theta = {}, weight = {}".format(angle, true_surr_angle, theta[0], sum_weights_td_total[0][0][y_i][x_i]))

            if(condition):

                for sequence in range(number_of_displays):
                    avg.append(sum_weights_td_total[sequence][0][y_i][x_i])
                neur_avg[i].append(np.array(avg))
            else:
                # print(sum_weights_td_total[0][0][y_i][x_i])
                # print(number_of_displays)
                neur_avg[i].append(np.zeros((number_of_displays)))
            if(i<len(spinet.neurons[layer_id][neuron_id].out_connections)-1):
                #print(len(neur_avg))
                neur_avg.append([])
            x_i+=1

        # print(np.array(neur_avg).sum())
        # print()
        # print(neur_avg)
        # print(len(neur_avg))
        # print((neur_avg))
        neur_avg = np.sum(neur_avg, axis=0)/len(neur_avg)
        # print(neur_avg)
        arge = np.arange(1, len(np.array(range(0, number_of_displays)))+1)
        #arge[0] = 1
        # plt.plot(arge, neur_avg[0], 'g-',label="total amount of topdown inhibition")
        # plt.xlabel("length of grating (in pixels)")
        # plt.ylabel("amount of inhibition received")
        # plt.title("evolution of top down inhibition by grating's length")
        # plt.show()
    # print(neur_avg)
    return neur_avg[0]

def td_inhibition_characterization(spinet: SpikingNetwork, layer_id, neuron_id, angle_val, depth_simple = 64, depth_complex = 16, thickness = 3, start_space = 45, space = 3):
    angles = [0, 23, 45, 68, 90, 113, 135, 158]
    number_of_displays = 55
    arge = np.arange(1,start_space+1)
    avg_same_tuned = []
    avg_actual_orientations = []
    avg_other_orientations = []
    avg_total_received = []
    diff = number_of_displays - start_space
    for value in range(1, diff+1):
        arge = np.append(arge, start_space + space * value)
    print(len(arge))
    for i, angle in enumerate(angles):
        spinet.load_statistics_2(thickness, angle, 0, layer_id=0, simulation=0)
        if(angle!=0):
            not_to_look = [angle, -angle]
        else:
            not_to_look = [0, 180]
            
        if(angle!=angle_val):
            without = True
        else:
            without = False

        simple_same_tuned_received = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, angle_val, thickness = thickness, depth_simple = depth_simple, depth_complex = depth_complex, tuned_simple = True, allthick = True, without = without, without_angle = not_to_look)    
        simple_actual_orientations_received = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, angle_val, thickness = thickness, depth_simple = depth_simple, depth_complex = depth_complex, actual_orientation_sequence = True, actual_sequence_angle = not_to_look, allthick = True, without = without, without_angle = not_to_look)
        simple_other_orientations_received = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, angle_val, thickness = thickness, depth_simple = depth_simple, depth_complex = depth_complex, actual_sequence_angle=not_to_look, untuned_angles_for_seq = True, allthick = True, without = without, without_angle = not_to_look)
        simple_total_received = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, angle_val, thickness = thickness, depth_simple = depth_simple, depth_complex = depth_complex, allthick = True, without = without, without_angle = not_to_look)
        
        avg_same_tuned.append(simple_same_tuned_received)
        avg_actual_orientations.append(simple_actual_orientations_received)
        avg_other_orientations.append(simple_other_orientations_received)
        avg_total_received.append(simple_total_received)
            
        fig = plt.figure(i+1)
        ax = fig.add_axes([0,0,1,1])
        plt.plot(arge, simple_same_tuned_received, 'g-',label="td inhibition received by simple neurons from complex neurons both tuned at " + str(angle_val) + "°.")
        plt.plot(arge, simple_actual_orientations_received, 'r-',label="td inhibition received by simple neurons tuned at sequence's orientation from complex neurons tuned at " + str(angle_val) + "°.")
        plt.plot(arge, simple_other_orientations_received, 'y-',label="td inhibition received by simple neurons of orientations different from sequence's and complex neurons'")
        plt.plot(arge, np.array(simple_other_orientations_received)+np.array(simple_same_tuned_received), 'm-',label="td inhibition received by all neurons untuned to orientation")
        plt.plot(arge, simple_total_received, 'b-',label="td inhibition received by all simple neurons from complex neurons of orientation of " + str(angle_val) + "°.")
        
        plt.xlabel("length of grating (in pixels)")
        plt.ylabel("amount of inhibition received")
        plt.title("evolution of top down inhibition by grating's length")
        plt.xticks(arge)
        plt.legend()
    return np.mean(avg_same_tuned), np.mean(avg_actual_orientations), np.mean(avg_other_orientations), np.mean(avg_total_received)

def visualize_sum_inhibition_weights2(spinet: SpikingNetwork, layer_id, neuron_id, sequence, angle, thickness = 1, depth = 64, allthick = False):
    #sum_wi_lat : sum of lateral weights received for depth = d by each different type of neuron regardless of their depth.
    lateral_weights = spinet.neurons[layer_id][neuron_id].weights_li
    sum_weights_lat = [[]]
    sum_wi_lat = np.zeros((np.array(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][1][0])).shape)
    #sum_weights_lat = spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][neuron_z][2]["sum_inhib_weights"][1]
    for d in range(depth):
        sum_wi_lat = np.zeros((np.array(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][0][2]["sum_inhib_weights"][1][0])).shape)
        for r in range(0, depth):
            sum_wi_lat += np.array(spinet.stats[sequence][str(
                sequence)][layer_id][str(layer_id)][r][2]["sum_inhib_weights"][1][d])
            #print(sum_wi_lat)
        #sum_wi_lat/=depth
        sum_weights_lat[d].append(sum_wi_lat)
        if(d<depth-1):
            sum_weights_lat.append([])
    #sum_weights_lat /= len(spinet.stats[sequence]
                           #[str(sequence)][layer_id][str(layer_id)])
    #max_lat /= len(spinet.stats[sequence]
                   #[str(sequence)][layer_id][str(layer_id)])
    print(sum_wi_lat.shape)
    print(np.array(sum_weights_lat).shape)
    layout = np.load(spinet.path + "weights/layout_" + str(layer_id) + ".npy")
    x_neur = np.where(layout == neuron_id)[0][0]
    y_neur = np.where(layout == neuron_id)[1][0]
    if(len(lateral_weights) != 0):
        space = 1
        x = []
        y = []
        range_ = spinet.conf["neuronInhibitionRange"]
        range_x = range_[0]
        range_y = range_[1]
        once = True
        depth_avg = []
        ct_d = 0 
        for d in range(depth):
            avg = []
            it = 0
            wi = 0
            theta = np.array(spinet.neurons[layer_id][neuron_id+d].theta)
            if(allthick):
                theta_3 = theta[2]
                theta = np.append(np.array(theta[0]),np.array(theta[1]))
                theta = np.append(theta, np.array(theta_3))
            if(angle ==-1):
                condition = False
            elif(angle!=0):
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
                print(d)
                #avg.append([])
                for value_x in range(x_neur-range_x, x_neur+range_x+1):
                    for value_y in range(y_neur-range_y, y_neur+range_y+1):
                        if(it < len(sum_weights_lat[d][0]) and (value_x != x_neur or value_y != y_neur)):
                            value = sum_weights_lat[d][0][it]
                            it += 1
                        if((value_x != x_neur or value_y != y_neur) and value_x >= 0 and value_y >= 0 and value_x < (len((spinet.conf["layerPatches"])[0][0]) * (spinet.conf["layerSizes"])[0][0]) and value_y < (len((spinet.conf["layerPatches"])[0][1]) * (spinet.conf["layerSizes"])[0][1])):
                            if(once):
                                x.append(value_x)
                                y.append(value_y)
                            wi = value
                            avg.append(wi)
                if(once):
                    once=False
                #ct_d+=1
                depth_avg.append(avg)
                print("it = {} , depth avg shape = {} and avg shape = {}".format(it, np.array(depth_avg).shape,np.array(avg).shape))
        fig = plt.figure(figsize=(20, 20), dpi=80)
        if(len(depth_avg)!=0):
            avg = np.sum(np.array(depth_avg), axis = 0)
            print(np.sum(avg))
            #print(avg.shape)
        ax = fig.add_subplot(111)
        Blues = plt.get_cmap('Blues')
        rect = []
        if(len(avg)!=0):
            max_ = np.max(avg)
            print(max_)
            avg = np.array(avg)/max_
        else:
            max_=0
            avg = np.zeros((len(x)))
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
    return avg, x, y

def visualize_td_sum_inhibition2(spinet: SpikingNetwork, layer_id, neuron_id, neuron_z, sequence, angle, thickness = 1, depth_simple = 64, allthick = False, tuned_simple = False):
    sum_weights_tdi = np.zeros((np.array(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][0][6]["top_down_weights"])).shape)
    for r in range(0, depth_simple):
        try:
            theta = np.array(spinet.neurons[layer_id][r].theta)
            if(tuned_simple):
                if(allthick):
                    theta_3 = theta[2]
                    theta = np.append(np.array(theta[0]),np.array(theta[1]))
                    theta = np.append(theta, np.array(theta_3))
                if(angle ==-1):
                    condition = True
                elif(angle!=0):
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
                    sum_weights_tdi += np.array(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][r][6]["top_down_weights"])
            else:
                sum_weights_tdi += np.array(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)][r][6]["top_down_weights"])
        
        except:
            ValueError
    sum_weights_tdi /=len(spinet.stats[sequence][str(sequence)][layer_id][str(layer_id)])
    space = 1
    fig = plt.figure(figsize=(20, 20), dpi=80)
    ax = fig.add_subplot(111)
    Blues = plt.get_cmap('Blues')
    rect = []
    depth = spinet.l_shape[layer_id+1][2]
    ordinate = np.linspace(0,len(spinet.neurons[layer_id][neuron_id].out_connections)/depth -1,int(len(spinet.neurons[layer_id][neuron_id].out_connections)/depth))
    y_i=0
    x_i=0
    start_init_neur = 0
    init_neur = spinet.neurons[layer_id][neuron_id].out_connections[start_init_neur]
    count_tuned = 0
    count_ok_tuned = 0
    count_total_good = 0
    for i in range(sum_weights_tdi.shape[0]*sum_weights_tdi.shape[1]):
        if(i%16==0 and i!=0):
            start_init_neur+=16
            init_neur=spinet.neurons[layer_id][neuron_id].out_connections[start_init_neur]
        theta = np.array(spinet.neurons[layer_id+1][init_neur+i].theta)
        if(allthick):
            theta_3 = theta[2]
            theta = np.append(np.array(theta[0]),np.array(theta[1]))
            theta = np.append(theta, np.array(theta_3))
        if(angle ==-1):
            condition = True
        elif(angle!=0):
            if(allthick):
                condition = ( (np.array(theta)==angle).any() ) or ( (np.array(theta)==-angle).any() )
            else:
                condition = ( (np.array(theta[thickness-1])==angle).any() ) or ( (np.array(theta[thickness-1])==-angle).any() )
        elif(angle==0):
            if(allthick):
               condition = ( (np.array(theta)==0).any() ) or ( (np.array(theta)==180).any() )
            else:
                condition = ( (np.array(theta[thickness-1])==0).any() ) or ( (np.array(theta[thickness-1])==180).any() )
        if(i%depth==0 and i!=0):
            y_i+=1
        if(y_i==0):
            x_i=i
        else:
            x_i=i-y_i*depth
        if(not condition):
            rect.append(matplotlib.patches.Rectangle(
                (x_i, ordinate[y_i]), space, space, color=Blues(sum_weights_tdi[y_i][x_i])))
            if(sum_weights_tdi[y_i][x_i]!=0):
                count_total_good+=sum_weights_tdi[y_i][x_i]
        else:
            count_tuned+=1
            if(sum_weights_tdi[y_i][x_i]!=0):
                count_ok_tuned+=sum_weights_tdi[y_i][x_i]
                count_total_good+=sum_weights_tdi[y_i][x_i]
            rect.append(matplotlib.patches.Rectangle(
                (x_i, ordinate[y_i]), space, space, facecolor=Blues(sum_weights_tdi[y_i][x_i]), edgecolor= 'red',linewidth=5))
        ax.add_patch(rect[i])
    cmapp = matplotlib.cm.ScalarMappable(cmap=Blues)
    max_tdi = np.max(sum_weights_tdi)
    cmapp.set_clim(0, max_tdi)
    plt.colorbar(cmapp, ax=ax, ticks=(0, max_tdi/4, max_tdi/2, 3*max_tdi/4, max_tdi))
    plt.gca().set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    ax.set_xlim(0, depth+space)
    ax.set_ylim(ordinate[0], ordinate[-1]+space)
    plt.axis('off')
    plt.show()
    return [count_ok_tuned, count_total_good]

def data_analysis_inhibition(spinet: SpikingNetwork, layer_id, neuron_id, angle_val, thickness, depth_simple = 64, depth_complex = 16, allthick = True):
    
    #lat inhibition received by tuned simple cells from other tuned simple cells
    lat_received_tuned = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle_val, thickness = thickness, depth = depth_simple, tuned_simple = True, allthick = True)    
    #lat inhibition received by tuned simple cells from all simple cells, tuned or not
    lat_received_all = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, angle_val, thickness = thickness, depth = depth_simple,  tuned_simple = True, allthick = True, untuned_simple = True)    
    
    #td inhibition received by tuned simple cells from tuned complex cells
    td_received_tuned = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, angle_val, thickness = thickness, depth_simple = depth_simple, depth_complex = depth_complex, tuned_simple = True, allthick = True) 
    #td inhibition received by tuned simple cells from untuned complex cells
    td_received_all = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, angle_val, thickness = thickness, depth_simple = depth_simple, depth_complex = depth_complex, tuned_simple = True, allthick = True, untuned_complex=True) 
    
    angles = [0, 23, 45, 68, 90, 113, 135, 158]
    if(angle_val!=0):
        not_to_look = [angle_val, -angle_val]
    else:
        not_to_look = [0, 180]
    each_ori_receivedtd = [[]]
    each_ori_receivedlat = [[]]
    
    avg_lat_total = []
    avg_td_total = []
    for i, init_angle in enumerate(angles):
        if(init_angle!=angle_val):
            without=True
        else:
            without=False
        for j, angle in enumerate(angles):
            if(angle!=0):
                new_look = [angle, -angle]
            else:
                new_look = [0, 180]
            if(angle==angle_val):
                take_off_ori = False
            else:
                take_off_ori = True
            
                
            test_oritd = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, init_angle, thickness = thickness, depth_simple = depth_simple, depth_complex = depth_complex, actual_orientation_sequence = True, actual_sequence_angle = new_look, allthick = True, without = without, without_angle = not_to_look, take_off_ori = take_off_ori, take_off_angle = not_to_look)
            each_ori_receivedtd[i].append(np.mean(test_oritd))
        
            test_orilat = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, init_angle, thickness = thickness, depth = depth_simple, actual_orientation_sequence = True, actual_sequence_angle = new_look, allthick = True, without = without, without_angle= not_to_look, take_off_ori = take_off_ori, take_off_angle = not_to_look)
            each_ori_receivedlat[i].append(np.mean(test_orilat))
        if(i < len(angles)-1):
            each_ori_receivedtd.append([])
            each_ori_receivedlat.append([])
                
        lat_total_received = visualize_total_inhibition_evolution2(spinet, layer_id, neuron_id, init_angle, thickness = thickness, depth = depth_simple, allthick = True, without = without, without_angle = not_to_look)
        td_total_received = visualize_total_tdinhibition_evolution2(spinet, layer_id, neuron_id, init_angle, thickness = thickness, depth_simple = depth_simple, depth_complex = depth_complex, allthick = True, without = without, without_angle=not_to_look)
        avg_lat_total.append(np.mean(lat_total_received))
        avg_td_total.append(np.mean(td_total_received))
        
    percentages_td = [[]]
    percentages_lat = [[]]
    for i, init_angle in enumerate(angles):
        for j, angle in enumerate(angles):
            percentages_td[i].append(each_ori_receivedtd[i][j]*100/avg_td_total[i])
            percentages_lat[i].append(each_ori_receivedlat[i][j]*100/avg_lat_total[i])
    
        if(i<len(angles)-1):
            percentages_td.append([])
            percentages_lat.append([])
    
    normalized_max_percentages_tuned_td = []
    normalized_max_percentages_tuned_lat = []
    index_angle = np.where(np.array(angles)==angle_val)[0][0]
    for i, v in enumerate(percentages_td): 
        v = v - min(v)
        normalized_max_percentages_tuned_td.append((v/max(v))[index_angle])
        
    for i, v in enumerate(percentages_lat): 
        v = v - min(v)
        normalized_max_percentages_tuned_lat.append((v/max(v))[index_angle])
        
    index_angle_minus = index_angle-1
    if(index_angle==len(angles)-1):
        index_angle_plus = 0
    else:
        index_angle_plus = index_angle+1
        
    normalized_min_percentages_neighbors_td = [[]]
    normalized_min_percentages_neighbors_lat = [[]]

    for i, v in enumerate(percentages_td):
        v = v - min(v)
        normalized_min_percentages_neighbors_td[i].append([(v/max(v))[index_angle_minus],(v/max(v))[index_angle_plus]] )
        if(i<len(angles)-1):
            normalized_min_percentages_neighbors_td.append([])
            
    for i, v in enumerate(percentages_lat):
        v = v - min(v)
        normalized_min_percentages_neighbors_lat[i].append([(v/max(v))[index_angle_minus],(v/max(v))[index_angle_plus]] )
        if(i<len(angles)-1):
            normalized_min_percentages_neighbors_lat.append([])
            
    return (lat_received_tuned, lat_received_all,  td_received_tuned, td_received_all, 
            normalized_max_percentages_tuned_lat, normalized_max_percentages_tuned_td, 
            normalized_min_percentages_neighbors_lat, normalized_min_percentages_neighbors_td)