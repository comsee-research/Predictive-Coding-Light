#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 18:32:36 2020

@author: thomas
"""

import itertools
import json
import os
import random
import re
import shutil

import numpy as np
import scipy.io as sio
from PIL import Image
from natsort import natsorted


def delete_files(folder):
    for file in os.scandir(folder):
        try:
            if os.path.isfile(file.path) or os.path.islink(file.path):
                os.unlink(file.path)
            elif os.path.isdir(file.path):
                shutil.rmtree(file.path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file.path, e))


def compress_weight(weights, path, max_weight, min_weight):
    img = np.array(255 * ( (weights - min_weight) / (max_weight-min_weight)), dtype=np.uint8)
    Image.fromarray(img).save(path)


def reshape_weights(weights, width, height):
    weights = np.concatenate((weights, np.zeros((1, width, height))), axis=0)
    return np.kron(np.swapaxes(weights, 0, 2), np.ones((3, 3, 1)))


def shuffle_weights(path):
    neurons_paths = natsorted(os.listdir(path))
    for pattern in [".*tdi", ".*li"]:
        weight_files = list(filter(re.compile(pattern).match, neurons_paths))
        shuffled_weight_files = random.sample(weight_files, len(weight_files))

        for old_name, new_name in zip(weight_files, shuffled_weight_files):
            os.rename(path + old_name, path + new_name + "bis")

        for name in weight_files:
            os.rename(path + name + "bis", path + name)


def clean_network(path, layers):
    for i in layers:
        delete_files(path + "weights/" + str(i) + "/")
        delete_files(path + "images/" + str(i) + "/")
    os.remove(path + "networkState.json")


class SpikingNetwork:
    """Spiking Neural Network class"""

    def __init__(self, path, loading=True):
        self.path = path
        try:
            with open(path + "configs/network_config.json") as file:
                self.conf = json.load(file)
            with open(path + "configs/simple_cell_config.json") as file:
                self.simple_conf = json.load(file)
            with open(path + "configs/complex_cell_config.json") as file:
                self.complex_conf = json.load(file)
            with open(path + "networkState.json") as file:
                self.state = json.load(file)
        except FileNotFoundError as e:
            print(e)

        self.nb_neurons = 0
        self.neurons = []
        self.weights = []
        self.weights_local_inhib = []
        self.spikes = []
        self.layout = []
        self.shared_id = []
        self.stats = []
        self.changes_sc = []
        self.changes_cc = []
        self.changes_li = []
        self.changes_tdi = []
        self.weights_tdi = []
        self.weights_li = []
        self.weights_le = []
        self.weights_tde = []
        self.delays = []
        self.delays_lli = []
        self.delays_li = []
        self.delays_tdi = []
        self.delays_le = []
        self.delays_tde = []
        type_to_config = {"SimpleCell": "simple_cell_config.json", "ComplexCell": "complex_cell_config.json"}

        self.p_shape = np.array(self.conf["layerPatches"], dtype=object)
        self.l_shape = np.array(self.conf["layerSizes"])
        self.n_shape = np.array(self.conf["neuronSizes"])

        if loading:
            for layer, neuron_type in enumerate(self.conf["neuronType"]):
                neurons, spikes = self.load_neurons(layer, neuron_type, type_to_config[neuron_type])
                self.neurons.append(neurons)
                self.spikes.append(spikes)
                self.layout.append(np.load(path + "weights/layout_" + str(layer) + ".npy"))
                transition_weights = self.load_weights(layer, neuron_type)
                self.weights.append(transition_weights[0])
                self.weights_local_inhib.append(transition_weights[1])
                self.weights_li.append(transition_weights[2])
                self.weights_tdi.append(transition_weights[3])
                
        for i in range(len(self.spikes)):
            if np.array(self.spikes[i], dtype=object).size > 0:
                self.spikes[i] = np.array(list(itertools.zip_longest(*self.spikes[i], fillvalue=0))).T
                try:
                    self.spikes[i][self.spikes[i] != 0] -= np.min(self.spikes[i][self.spikes[i] != 0])
                except ValueError:
                    pass

        if os.path.exists(self.path + "gabors/0/rotation_response.npy"):
            self.directions = []
            self.orientations = []
            for layer, neuron_type in enumerate(self.conf["layerCellTypes"]):
                if layer < 2:
                    self.directions.append(np.load(self.path + "gabors/"+str(layer)+"/rotation_response.npy"))
                    self.orientations.append(self.directions[layer][0:8] + self.directions[layer][8:16])

        if os.path.exists(self.path + "gabors/data/disparity_response.npy"):
            self.disparities = np.load(self.path + "gabors/data/disparity_response.npy")

    def load_neurons(self, layer, neuron_type, config):
        neurons = []
        spike_train = []
        neurons_paths = natsorted(os.listdir(self.path + "weights/" + str(layer) + "/"))
        config_files = list(filter(re.compile(".*json").match, neurons_paths))
        for index in range(len(config_files)):
            neuron = Neuron(neuron_type, index,
                            self.path + "configs/" + config,
                            self.path + "weights/" + str(layer) + "/"
                            )
            neurons.append(neuron)
            if neuron.conf["TRACKING"] == "partial":
                if(len(neuron.params["spike_train"])!=0):
                    spike_train.append(neuron.params["spike_train"])
                else:
                    spike_train.append([0])
        self.nb_neurons += len(neurons)
        return neurons, spike_train

    def load_weights(self, layer, neuron_type):
        weights = []
        weights_local_inhib = []
        weights_li = []
        weights_tdi = []
        # print(layer)
        if neuron_type == "SimpleCell" and self.conf["sharingType"] == "patch":
            self.shared_id.append([])
            step = self.l_shape[layer, 0] * self.l_shape[layer, 1] * self.l_shape[layer, 2]
            for r_id in range(0, len(self.neurons[layer]), step):
                for i, neuron in enumerate(self.neurons[layer][r_id: r_id + self.l_shape[layer, 2]]):

                    weights.append(np.load(self.path + "weights/" +str(layer) +"/"  + str(neuron.id) + ".npy"))
                    weights_local_inhib.append(np.load(self.path + "weights/" +str(layer) +"/" + str(neuron.id) + "lli.npy"))
                    self.shared_id[-1].append(
                        np.arange(r_id + i, r_id + i + step, self.l_shape[layer, 2]))
                    

            self.shared_id[-1] = np.array(self.shared_id[-1])

            for i, weight in enumerate(weights):
                for shared in self.shared_id[-1][i]:
                    self.neurons[layer][shared].link_weights(weight)
                    # if(layer==0):
                    #     weights_tdi.append(np.load(self.path + "weights/" +str(layer) +"/" + str(self.neurons[layer][shared].id) + "tdi.npy"))


        else:
            for neuron in self.neurons[layer]:
                neuron.link_weights(np.load(self.path + "weights/" + str(layer) + "/" + str(neuron.id) + ".npy"))
                weights.append(neuron.weights)
                weights_local_inhib.append(np.load(self.path + "weights/" +str(layer) +"/"  + str(neuron.id) + "lli.npy"))
        return np.array(weights), np.array(weights_local_inhib), np.array(weights_li), np.array(weights_tdi)

    def load_intermediate_weights(self, layer, neuron_type, count):
        weights = []
        weights_local_inhib = []
        weights_li = []
        weights_tdi = []
        if neuron_type == "SimpleCell" and self.conf["sharingType"] == "patch":
            self.shared_id.append([])
            step = self.l_shape[layer, 0] * self.l_shape[layer, 1] * self.l_shape[layer, 2]
            for r_id in range(0, len(self.neurons[layer]), step):
                for i, neuron in enumerate(self.neurons[layer][r_id: r_id + self.l_shape[layer, 2]]):
                    weights.append(np.load(self.path + "weights/intermediate_" +str(count)+"/"+str(layer) +"/"  + str(neuron.id) + ".npy"))
                    weights_local_inhib.append(np.load(self.path + "weights/intermediate_" +str(count)+"/"+str(layer) +"/" + str(neuron.id) + "lli.npy"))
                    self.shared_id[-1].append(
                        np.arange(r_id + i, r_id + i + step, self.l_shape[layer, 2]))
                    

            self.shared_id[-1] = np.array(self.shared_id[-1])

            for i, weight in enumerate(weights):
                for shared in self.shared_id[-1][i]:
                    self.neurons[layer][shared].link_weights(weight)
                    if(layer==0):
                    #     weights_tdi.append(np.load(self.path + "weights/intermediate_" +str(count) +"/"+str(layer) +"/" + str(self.neurons[layer][shared].id) + "tdi.npy"))
                        weights_li.append(np.load(self.path + "weights/intermediate_" +str(count) +"/"+str(layer) +"/" + str(self.neurons[layer][shared].id) + "li.npy"))



        else:
            for neuron in self.neurons[layer]:
                neuron.link_weights(np.load(self.path + "weights/intermediate_" +str(count)+"/"+str(layer) +"/" + str(neuron.id) + ".npy"))
                weights.append(neuron.weights)
                weights_local_inhib.append(np.load(self.path + "weights/intermediate_" +str(count)+"/"+str(layer) +"/" + str(neuron.id) + "lli.npy"))
                weights_li.append(np.load(self.path + "weights/intermediate_" +str(count)+"/"+str(layer) +"/" + str(neuron.id) + "li.npy"))

        return np.array(weights), np.array(weights_local_inhib), np.array(weights_li), np.array(weights_tdi)
    
    def generate_weight_images(self):
        for layer in range(self.p_shape.shape[0]):
            if layer == 0:
                for i, weights in enumerate(self.weights[layer]):
                    max_weight = np.max(weights)
                    min_weight = 0 
                    for synapse in range(self.conf["neuron1Synapses"]):
                        for camera in range(self.conf["nbCameras"]):
                            n_weight = reshape_weights(
                                weights[:, camera, synapse], self.n_shape[layer, 0, 0], self.n_shape[layer, 0, 1],
                            )
                            path = (self.path + "images/0/" + str(i) + "_syn" + str(synapse) + "_cam" + str(
                                camera) + ".png")
                                    
                            compress_weight(n_weight, path, max_weight, min_weight)
                            if np.any(self.shared_id[layer]):
                                for shared in self.shared_id[layer][i]:
                                    self.neurons[layer][shared].weight_images.append(path)
                            else:
                                self.neurons[layer][i].weight_images.append(path)
            else:
                print(layer)
                for i, neuron in enumerate(self.neurons[layer]):
                    weights = np.mean(neuron.weights, axis=2)
                    weights = np.swapaxes(weights, 0, 1)
                    weights = np.stack((weights, np.zeros(weights.shape), np.zeros(weights.shape)), axis=2)
                    path = self.path + "images/" + str(layer) + "/" + str(i) + ".png"
                    compress_weight(np.kron(weights, np.ones((7, 7, 1))), path, weights.max(), 0)
                    neuron.weight_images.append(path)
    
    def generate_weight_mat(self):
        w = self.n_shape[0, 0,0] * self.n_shape[0, 0,1]
        basis = np.zeros((w, len(self.weights[0])))
        for c in range(self.conf["nbCameras"]):
            for i, weight in enumerate(self.weights[0]):
                basis[c * w: (c + 1) * w, i] = ((weight[0, c, 0] - weight[1, c, 0])).flatten("F")

        return basis

    def generate_weight_mat2(self):
        w = self.n_shape[0, 0,0]
        h = self.n_shape[0, 0,1]
        dim = 10
        basis = np.zeros((len(self.weights[0]),dim,dim))
        for c in range(self.conf["nbCameras"]):
            for i, weight in enumerate(self.weights[0]):
                basis[i] = ((- weight[0, c, 0] + weight[1, c, 0]))

        return basis
    
    def save_rotation_response(self, spikes, rotations):
        self.directions = []
        self.orientations = []
        for layer, response in enumerate(spikes):
            spike_vector = []
            for rot in range(rotations.size):
                spike_vector.append(np.count_nonzero(response[rot], axis=1))
            spike_vector = np.array(spike_vector)
            np.save(self.path + "gabors/"+str(layer)+"/rotation_response", spike_vector)
            self.directions = spike_vector
            self.orientations = self.directions[0:8] + self.directions[8:16]

    def save_complex_disparities(self, spikes, disparities):
        spike_vector = []
        for disp in range(disparities.size):
            spike_vector.append(np.count_nonzero(spikes[disp], axis=1))
        spike_vector = np.array(spike_vector)

        np.save(self.path + "gabors/data/disparity_response", spike_vector)
        self.disparities = spike_vector

    def spike_rate(self):
        time = np.max(self.spikes)
        srates = np.count_nonzero(self.spikes, axis=1) / (time * 1e-6)
        return np.mean(srates), np.std(srates)

    def neurons_spike_rate(self):
        neuron_spikes=[]
        time_1 = np.max(self.spikes[0])
        time_2 = np.max(self.spikes[1])
        time = max([time_1,time_2])
        
        for layer, _ in enumerate(self.conf["layerCellTypes"]):
            srates = []
            neuron_count=len(self.spikes[layer])
            for i in range(0,neuron_count):
                if(time!=0):
                    value = np.count_nonzero(self.spikes[layer][i]) / (time * 1e-6)
                else:
                    value = 0
                srates.append(value)
            neuron_spikes.append(srates)
        return neuron_spikes
    
    def load_statistics(self, layer_id = 0, simulation = 0):
        
        layers = [ f.path for f in os.scandir(self.path + "statistics/") if f.is_dir() ]
        layers = natsorted(layers)
        number_of_sequences = len([ f.path for f in os.scandir(layers[0] + "/" + "/" + str(simulation) + "/") if f.is_dir() ])
        if(len(self.stats)!=0):
            self.stats=[]
        limit_neurs = 608
        cter=0
        for sequence in range(number_of_sequences):
            layer_ = []
            for counter, layer in enumerate(layers):
                dir_exists = os.path.isdir(layer + "/" + "/" + str(simulation) + "/" + str(sequence))
                if(dir_exists):
                    list_of_neurons = natsorted([ f.path for f in os.scandir(layer + "/" + "/"  + str(simulation) + "/" + str(sequence)) ])
                    temp_neurons=[[]]
                    for counter_2, neuron in enumerate(list_of_neurons):
                        cter+=1
                        if(cter>limit_neurs):
                            cter=0
                            break
                        with open(neuron) as file:
                            params = json.load(file)
                        temp_neurons[counter_2].append({"amount_of_events":params["amount_of_events"]})
                        temp_neurons[counter_2].append({"potential_train":params["potential_train"]})
                        temp_neurons[counter_2].append({"sum_inhib_weights":params["sum_inhib_weights"]})
                        temp_neurons[counter_2].append({"timing_of_inhibition":params["timing_of_inhibition"]})
                        temp_neurons[counter_2].append({"potentials_thresholds":params["potentials_thresholds"]})
                        temp_neurons[counter_2].append({"excitatory_events":params["excitatory_ev"]})
                        temp_neurons[counter_2].append({"top_down_weights":params["sum_topdown_weights"]})
                        if(counter_2!=len(list_of_neurons)-1 and cter+1<=limit_neurs):
                            temp_neurons.append([])

                    layer_.append({"{}".format(str(counter)):temp_neurons})
                    cter=0
            if(len(layer_)>1):
                self.stats.append({"{}".format(str(sequence)):layer_})

            
    def load_statistics_2(self, thickness, angle, direction, layer_id=0, simulation=0, separate_speed = False, speed = 0):
        if(direction==0):
            sign = ""
        else:
            sign = "-"
        layers = [ f.path for f in os.scandir(self.path + "statistics/") if f.is_dir() ]
        layers = natsorted(layers)
        if(not separate_speed):
            number_of_sequences = len([ f.path for f in os.scandir(layers[0] + "/" + str(thickness) + "/" + sign + str(angle) + "/" + str(simulation) + "/") if f.is_dir() ])
        else:
            number_of_sequences = len([ f.path for f in os.scandir(layers[0] + "/" + str(thickness) + "/speeds/" + str(speed) + "/" + sign + str(angle) + "/" + str(simulation) + "/") if f.is_dir() ])
        if(len(self.stats)!=0):
            self.stats=[]
        for sequence in range(number_of_sequences):
            layer_ = []
            for counter, layer in enumerate(layers):
                if(not separate_speed):
                    dir_exists = os.path.isdir(layer + "/" + str(thickness) + "/" + sign + str(angle) + "/" + str(simulation) + "/" + str(sequence))
                else:
                    dir_exists = os.path.isdir(layer + "/" + str(thickness) + "/speeds/" + str(speed) + "/" + sign + str(angle) + "/" + str(simulation) + "/" + str(sequence))
                if(dir_exists):
                    if(not separate_speed):
                        list_of_neurons = natsorted([ f.path for f in os.scandir(layer + "/" + str(thickness) + "/" + sign + str(angle) + "/" + str(simulation) + "/" + str(sequence)) ])

                    else:
                        list_of_neurons = natsorted([ f.path for f in os.scandir(layer + "/" + str(thickness) + "/speeds/" + str(speed) + "/" + sign + str(angle) + "/" + str(simulation) + "/" + str(sequence)) ])
                    temp_neurons=[[]]
                    for counter_2, neuron in enumerate(list_of_neurons):
                        with open(neuron) as file:
                            params = json.load(file)
                        temp_neurons[counter_2].append({"amount_of_events":params["amount_of_events"]})
                        temp_neurons[counter_2].append({"potential_train":params["potential_train"]})
                        temp_neurons[counter_2].append({"sum_inhib_weights":params["sum_inhib_weights"]})
                        temp_neurons[counter_2].append({"timing_of_inhibition":params["timing_of_inhibition"]})
                        temp_neurons[counter_2].append({"potentials_thresholds":params["potentials_thresholds"]})
                        temp_neurons[counter_2].append({"excitatory_events":params["excitatory_ev"]})
                        temp_neurons[counter_2].append({"top_down_weights":params["sum_topdown_weights"]})
                        if(counter_2!=len(list_of_neurons)-1):
                            temp_neurons.append([])
                    layer_.append({"{}".format(str(counter)):temp_neurons})
            if(len(layer_)>1):
                self.stats.append({"{}".format(str(sequence)):layer_})

        
        self.load_orientations()
        
    def load_statistics_standard(self, folder_name, layer_id=0, simulation=0):
        layers = [ f.path for f in os.scandir(self.path + "statistics/") if f.is_dir() ]
        layers = natsorted(layers)
        number_of_sequences = len([ f.path for f in os.scandir(layers[0] + "/" + folder_name + "/" + str(simulation) + "/") if f.is_dir() ])
        if(len(self.stats)!=0):
            self.stats=[]
        for sequence in range(number_of_sequences):
            layer_ = []
            for counter, layer in enumerate(layers):
                dir_exists = os.path.isdir(layer + "/" + folder_name + "/" + str(simulation) + "/" + str(sequence+1))
                if(dir_exists):
                    list_of_neurons = natsorted([ f.path for f in os.scandir(layer + "/" + folder_name + "/" + str(simulation) + "/" + str(sequence+1)) ])

                    temp_neurons=[[]]
                    for counter_2, neuron in enumerate(list_of_neurons):
                        with open(neuron) as file:
                            params = json.load(file)
                        temp_neurons[counter_2].append({"amount_of_events":params["amount_of_events"]})
                        temp_neurons[counter_2].append({"potential_train":params["potential_train"]})
                        temp_neurons[counter_2].append({"sum_inhib_weights":params["sum_inhib_weights"]})
                        temp_neurons[counter_2].append({"timing_of_inhibition":params["timing_of_inhibition"]})
                        temp_neurons[counter_2].append({"potentials_thresholds":params["potentials_thresholds"]})
                        temp_neurons[counter_2].append({"excitatory_events":params["excitatory_ev"]})
                        temp_neurons[counter_2].append({"top_down_weights":params["sum_topdown_weights"]})
                        if(counter_2!=len(list_of_neurons)-1):
                            temp_neurons.append([])
                    layer_.append({"{}".format(str(counter)):temp_neurons})
            if(len(layer_)>=1):
                self.stats.append({"{}".format(str(sequence)):layer_})

                
    def load_statistics_3(self, main_angle, surround_angle, layer_id=0, simulation=0, lat_value=0): #LOAD CROSS ORI
        layers = [ f.path for f in os.scandir(self.path + "statistics/") if f.is_dir() ]
        layers = natsorted(layers)
        if(len(self.stats)!=0):
            self.stats=[]
        layer_ = []
        for counter, layer in enumerate(layers):
            dir_exists = os.path.isdir(layer + "/" + str(lat_value) + "/" + str(main_angle) + "/" + str(surround_angle) + "/" + str(simulation) + "/1/")
            if(dir_exists):
                list_of_neurons = natsorted([ f.path for f in os.scandir(layer + "/" + str(lat_value) + "/" + str(main_angle) + "/" + str(surround_angle) + "/" + str(simulation) + "/1/") ])
                temp_neurons=[[]]
                for counter_2, neuron in enumerate(list_of_neurons):
                    with open(neuron) as file:
                        params = json.load(file)
                    temp_neurons[counter_2].append({"amount_of_events":params["amount_of_events"]})
                    temp_neurons[counter_2].append({"potential_train":params["potential_train"]})
                    # temp_neurons[counter_2].append({"sum_inhib_weights":params["sum_inhib_weights"]})
                    # temp_neurons[counter_2].append({"timing_of_inhibition":params["timing_of_inhibition"]})
                    # temp_neurons[counter_2].append({"potentials_thresholds":params["potentials_thresholds"]})
                    # temp_neurons[counter_2].append({"lateralWeights_stats":params["lateralWeights_stats"]})
                    # temp_neurons[counter_2].append({"top_down_weights":params["sum_topdown_weights"]})
                    if(counter_2!=len(list_of_neurons)-1):
                        temp_neurons.append([])

                layer_.append({"{}".format(str(counter)):temp_neurons})
        if(len(layer_)>1):
            self.stats.append({"{}".format(str(0)):layer_})
            #self.stats.append(layer_)
        
        self.load_orientations()
    
    def load_statistics_4(self, main_angle, surround_angle, layer_id=0, thickness=0, simulation = 0): #LOAD TUNING CURVES
        layers = [ f.path for f in os.scandir(self.path + "statistics/") if f.is_dir() ]
        layers = natsorted(layers)
        if(len(self.stats)!=0):
            self.stats=[]
        layer_ = []
        for counter, layer in enumerate(layers):
            dir_exists = os.path.isdir(layer + "/" + str(thickness) + "/" + str(main_angle) + "/" + str(surround_angle) + "/" + str(simulation) + "/1/")
            if(dir_exists):
                list_of_neurons = natsorted([ f.path for f in os.scandir(layer + "/" + str(thickness) + "/" + str(main_angle) + "/" + str(surround_angle) + "/" + str(simulation) + "/1/") ])
                temp_neurons=[[]]
                for counter_2, neuron in enumerate(list_of_neurons):
                    with open(neuron) as file:
                        params = json.load(file)
                    temp_neurons[counter_2].append({"amount_of_events":params["amount_of_events"]})
                    temp_neurons[counter_2].append({"potential_train":params["potential_train"]})
                    # temp_neurons[counter_2].append({"sum_inhib_weights":params["sum_inhib_weights"]})
                    # temp_neurons[counter_2].append({"timing_of_inhibition":params["timing_of_inhibition"]})
                    # temp_neurons[counter_2].append({"potentials_thresholds":params["potentials_thresholds"]})
                    # temp_neurons[counter_2].append({"lateralWeights_stats":params["lateralWeights_stats"]})
                    # temp_neurons[counter_2].append({"top_down_weights":params["sum_topdown_weights"]})
                    if(counter_2!=len(list_of_neurons)-1):
                        temp_neurons.append([])

                layer_.append({"{}".format(str(counter)):temp_neurons})
        if(len(layer_)>1):
            self.stats.append({"{}".format(str(0)):layer_})
            #self.stats.append(layer_)
        
        # self.load_orientations()
   
    def load_orientations(self):
        with open(self.path + "statistics/orientations/orientations.json") as file:
            params = json.load(file)
        cter = 0
        for neuron in self.neurons[0]:
            neuron.theta = params["orientations"][cter]
            if(len(neuron.theta[1])==0 and len(neuron.theta[2])==0):
                neuron.theta[1].append(-1)
                neuron.theta[2].append(-1)
            cter+=1
            if(cter==self.l_shape[0][2]):
                cter = 0
        cter = 0
        
    def load_weightchanges(self):
        with open(self.path + "weights/weightsChanges.json") as file:
            params = json.load(file)
        self.changes_sc = params['weights_changes'][0]
        self.changes_cc = params['weights_changes'][1]
        self.changes_li = params['weights_changes'][2]
        self.changes_tdi = params['weights_changes'][3]
class Neuron:
    """Spiking Neuron class"""

    def __init__(self, neuron_type, index, conf_path, weight_path):
        self.type = neuron_type
        self.id = index
        with open(conf_path) as file:
            self.conf = json.load(file)
        with open(weight_path + str(self.id) + ".json") as file:
            self.params = json.load(file)
        if self.type == "SimpleCell":
            # self.weights_tdi = np.load(weight_path + str(self.id) + "tdi.npy")
            self.weights_li = np.load(weight_path + str(self.id) + "li.npy")

        self.spike_train = np.array(self.params["spike_train"])
        self.weights = 0
        self.weight_images = []
        self.gabor_image = 0
        self.lambd = 0
        self.theta = 0
        self.phase = 0
        self.sigma = 0
        self.error = 0
        self.mu = None
        self.orientation = None
        self.disparity = 0
        self.lateral = np.array(self.params["lateral_dynamic_inhibition"])
        self.topdown = np.array(self.params["topdown_dynamic_inhibition"])
        self.out_connections = np.array(self.params["out_connections"])
        self.in_connections = np.array(self.params["in_connections"])

    def link_weights(self, weights):
        self.weights = weights

    def add_gabor(self, image, mu, sigma, lambd, phase, theta, error):
        self.gabor_image = image
        self.mu = mu
        self.sigma = sigma
        self.lambd = lambd
        self.phase = phase
        self.theta = theta
        self.error = error
        self.orientation = self.theta

    def add_disparity(self, disparity):
        self.disparity = disparity
