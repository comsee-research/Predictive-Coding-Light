#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 10:11:17 2020

@author: thomas
"""

from sklearn.model_selection import ParameterSampler, ParameterGrid
import json
import os
import subprocess
import numpy as np


def create_networks(exec_path, network_path, n_iter, params):
    for i in range(n_iter):
        n_path = network_path + "/network_" + str(i)
        create_network(exec_path, n_path)

        conf = open_config_files(n_path + "/configs/")

        for neuron_key, neuron_value in params.items():
            sample = list(ParameterSampler(neuron_value, 1))
            for key, value in sample[0].items():
                conf[neuron_key][key] = value

            save_config_files(n_path + "/configs/" + neuron_key + ".json", conf[neuron_key])


def random_params(exec_path, network_path, nb_networks):
    params = {"network_config": {"actionRate": [400, 500, 600],
                                 "decayRate": [0.01, 0.02, 0.04],
                                 "explorationFactor": [30, 50, 70],
                                 "nu": [0.5, 1, 2],
                                 "tauR": [0.5, 1, 2]},
              "simple_cell_config": {"ETA_INH": [10, 20, 30],
                                     "ETA_RP": [0.5, 1, 2],
                                     "TAU_RP": [10, 20, 30],
                                     "VTHRESH": [20, 30, 40]},
              "complex_cell_config": {"ETA_INH": [10, 20, 30],
                                      "ETA_RP": [0.5, 1, 2],
                                      "TAU_RP": [10, 20, 30],
                                      "VTHRESH": [2, 3, 4]},
              "critic_cell_config": {"ETA": [0.2, 0.5, 0.8],
                                     "TAU_E": [250, 500, 750],
                                     "NU_K": [150, 200, 250],
                                     "TAU_K": [25, 50, 75],
                                     "VTHRESH": [1, 2, 3]},
              "actor_cell_config": {"ETA": [0.2, 0.5, 0.8],
                                    "TAU_E": [250, 500, 750],
                                    "VTHRESH": [1, 2, 3]}
              }
    create_networks(exec_path, network_path, nb_networks, params)


def open_config_files(config_path):
    conf = {}
    for file in os.listdir(config_path):
        with open(config_path + file, "r") as f:
            conf[file[:-5]] = json.load(f)
    return conf


def save_config_files(config_path, conf):
    with open(config_path, "w") as f:
        json.dump(conf, f)


def change_param(json_path, key, value):
    with open(json_path, "r") as file:
        conf = json.load(file)
    conf[key] = value
    with open(json_path, "w") as file:
        json.dump(conf, file)


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def launch_neuvisys_multi_pass(exec_path, network_path, event_file, nb_pass):
    for path in execute([exec_path, network_path, event_file, str(nb_pass)]):
        print(path, end="")


def launch_neuvisys_ros(exec_path, network_path):
    for path in execute([exec_path, network_path]):
        print(path, end="")


def create_network(exec_path, network_path):
    for path in execute([exec_path, network_path]):
        print(path, end="")


def divide_visual_field(nb_x_region, nb_y_region, size_x_region, size_y_region):
    spacing_x = (346 - (size_x_region * nb_x_region)) / (nb_x_region - 1)
    spacing_y = (260 - (size_x_region * nb_x_region)) / (nb_x_region - 1)

    X = []
    for x in range(nb_x_region):
        X.append(x * (size_x_region + spacing_x))
    Y = []
    for y in range(nb_y_region):
        Y.append(y * (size_y_region + spacing_y))

    return X, Y
