#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 02:47:43 2020

@author: thomas
"""

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from natsort import natsorted
from PIL import Image
from src.spiking_network.network.neuvisys import SpikingNetwork
from src.spiking_network.network.neuvisys import compress_weight
import matplotlib



def pdf_simple_cell(spinet, layer, camera):
    pdf = FPDF(
        "P", "mm", (11 * spinet.l_shape[0, 0], 11 * spinet.l_shape[0, 1] * spinet.conf["neuron1Synapses"],),
    )
    pdf.add_page()

    for neuron in spinet.neurons[0]:
        x, y, z = neuron.params["position"]
        if z == layer:
            for i in range(spinet.conf["neuron1Synapses"]):
                pos_x = x * 11
                pos_y = y * spinet.conf["neuron1Synapses"] * 11 + i * 11
                pdf.image(neuron.weight_images[camera], x=pos_x, y=pos_y, w=10, h=10)
    return pdf


def pdf_complex_cell(spinet, zcell):
    pdf = FPDF(
        "P", "mm", (11 * spinet.l_shape[1, 0], 11 * spinet.l_shape[1, 1],),
    )
    pdf.add_page()

    for neuron in spinet.neurons[1]:
        x, y, z = neuron.params["position"]
        if z == zcell:
            pos_x = x * 11
            pos_y = y * 11
            pdf.image(neuron.weight_images[0], x=pos_x, y=pos_y, w=10, h=10)
    return pdf


def pdf_simple_cell_left_right_combined(spinet, layer):
    pdf = FPDF(
        "P", "mm", (11 * spinet.l_shape[0, 0], 24 * spinet.l_shape[0, 1] * spinet.conf["neuron1Synapses"],),
    )
    pdf.add_page()

    for neuron in spinet.neurons[0]:
        x, y, z = neuron.params["position"]
        if z == layer:
            for i in range(spinet.conf["neuron1Synapses"]):
                pos_x = 11 * x
                pos_y = 24 * y * spinet.conf["neuron1Synapses"] + i * 11
                pdf.image(neuron.weight_images[0], x=pos_x, y=pos_y, w=10, h=10)
                pdf.image(neuron.weight_images[1], x=pos_x, y=pos_y + 11, w=10, h=10)
    return pdf


def pdf_layers(spinet, rows, cols, nb_synapses, nb_layers):
    images = natsorted(os.listdir(spinet.path + "images/simple_cells/"))
    pdf = FPDF("P", "mm", (cols * 11, rows * 11 * nb_layers))
    pdf.add_page()

    count = 0
    for i in range(cols):
        for j in range(rows):
            for l in range(nb_layers):
                pdf.image(
                    spinet.path + "images/simple_cells/" + images[count],
                    x=i * 11,
                    y=j * 11 * nb_layers + l * 10.4,
                    w=10,
                    h=10,
                )
                count += nb_synapses
    return pdf


def pdf_weight_sharing(spinet, nb_cameras, camera):
    side = int(np.sqrt(spinet.l_shape[0, 2]))
    if nb_cameras == 1:
        pad = 11
    else:
        pad = 24
    xpatch = len(spinet.p_shape[0, 0])
    ypatch = len(spinet.p_shape[0, 1])
    pdf = FPDF("P", "mm", (11 * xpatch * side + (xpatch - 1) * 10, pad * ypatch * side + (ypatch - 1) * 10))
    pdf.add_page()

    shift = np.arange(spinet.l_shape[0, 2]).reshape((side, side))
    cell_range = range(0, len(spinet.neurons[0]), spinet.l_shape[0, 2] * spinet.l_shape[0, 0] * spinet.l_shape[0, 1])
    for i in cell_range:
        for neuron in spinet.neurons[0][i: i + spinet.l_shape[0, 2]]:
            x, y, z = neuron.params["position"]
            pos_x = (
                    (x // spinet.l_shape[0, 0]) * side * 11
                    + np.where(shift == z)[0][0] * 11
                    + (x // spinet.l_shape[0, 0]) * 10
            )  
            pos_y = (
                    (y // spinet.l_shape[0, 1]) * side * pad
                    + np.where(shift == z)[1][0] * pad
                    + (y // spinet.l_shape[0, 1]) * 10
            )
            if nb_cameras == 1:
                pdf.image(neuron.weight_images[camera], x=pos_x, y=pos_y, w=10, h=10)
            else:
                pdf.image(neuron.weight_images[0], x=pos_x, y=pos_y, w=10, h=10)
                pdf.image(neuron.weight_images[1], x=pos_x, y=pos_y + 11, w=10, h=10)
    return pdf


def pdf_weight_sharing_full(spinet, nb_cameras, camera):
    side = int(np.sqrt(spinet.l_shape[0, 2]))
    if nb_cameras == 1:
        pad = 11
    else:
        pad = 24
    pdf = FPDF("P", "mm", (11 * side, pad * side))
    pdf.add_page()

    shift = np.arange(spinet.l_shape[0, 2]).reshape((side, side))
    for neuron in spinet.neurons[0][0:spinet.l_shape[0, 2]]:
        x, y, z = neuron.params["position"]
        pos_x = (
                (x // spinet.l_shape[0, 0]) * side * 11
                + np.where(shift == z)[0][0] * 11
        )  # patch size + weight sharing shift
        pos_y = (
                (y // spinet.l_shape[0, 1]) * side * pad
                + np.where(shift == z)[1][0] * pad
        )
        if nb_cameras == 1:
            pdf.image(neuron.weight_images[camera], x=pos_x, y=pos_y, w=10, h=10)
        else:
            pdf.image(neuron.weight_images[0], x=pos_x, y=pos_y, w=10, h=10)
            pdf.image(neuron.weight_images[1], x=pos_x, y=pos_y + 11, w=10, h=10)
    return pdf


def pdf_deepsimple_rf(spinet, layer):
    val = 120
    for c, deepsimple in enumerate(spinet.neurons[layer]):
        if(c > 16):
            break
        ox, oy, oz = deepsimple.params["offset"]
        # print("ox, oy, oz = {}, {}, {}".format(ox, oy, oz))
        heatmap = np.zeros((spinet.n_shape[layer, 0, 0], spinet.n_shape[layer, 0, 1]))
        dimrf = 360 #90 # 120
        heatmap_rf = np.zeros((dimrf, dimrf, 3))

        maximum = np.max(deepsimple.weights)
        minimum = np.min(deepsimple.weights)
        print()
        print(maximum)
        print()
        for i in range(ox, ox + spinet.n_shape[layer, 0, 0]):
            for j in range(oy, oy + spinet.n_shape[layer, 0, 1]):
                once =True
                ct = 0
                for k in range(spinet.n_shape[layer, 0, 2]):
                    complex_cell = spinet.neurons[layer-1][spinet.layout[layer-1][i, j, k]]
                    xs, ys, zs = complex_cell.params["position"]
                    # print("xs, ys, zs = {}, {}, {}".format(xs, ys, zs))
                    weight_sc = ((deepsimple.weights[xs - ox, ys - oy, k]) - minimum)/ (maximum - minimum)
                    avg = (np.mean(deepsimple.weights[xs - ox, ys - oy]) )/ (maximum)
                    heatmap[ys - oy, xs - ox] += weight_sc
                    # print(len(complex_cell.weights[xs - ox, ys - oy]))
                    indices = np.argsort(-deepsimple.weights[xs - ox, ys - oy])
                    indices_lim = np.sort(indices[:4])
                    sorted_values = deepsimple.weights[xs - ox, ys - oy][indices]
                    top = sorted_values[:16]
                    if(once):
                        once = False
                    
                    # if np.argmax(deepsimple.weights[xs - ox, ys - oy]) == k:
                    if (ct < 4): 
                        if(indices_lim[ct] == k):
                            # print(indices)
                            print("ct = {}, k = {}".format(ct, k))
                            # print(deepsimple.weights[xs - ox, ys - oy, k])
                            # print(complex_cell.weight_images[1])
                            sc_weight_image = Image.open(complex_cell.weight_images[1])
                            # print(np.array(sc_weight_image).shape)
                            # print("ys - oy = {}, xs - ox = {}".format(ys - oy, xs - ox))
                            # print(heatmap_rf.shape)
                            # if(weight_sc > 0.85):
                            # heatmap_rf[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                            #     sc_weight_image * weight_sc) 
                            if(ct==0):
                                heatmap_rf[val * (0): val * (1), val * (0): val * (1)] = np.array(
                                    sc_weight_image * weight_sc) 
                            if(ct==1):
                                heatmap_rf[val * (2): val * (3), val * (0): val * (1)] = np.array(
                                    sc_weight_image * weight_sc) 
                            if(ct==2):
                                heatmap_rf[val * (0): val * (1), val * (2): val * (3)] = np.array(
                                    sc_weight_image * weight_sc) 
                            if(ct==3):
                                heatmap_rf[val * (2): val * (3), val * (2): val * (3)] = np.array(
                                    sc_weight_image * weight_sc) 
                            ct += 1
                        # print(weight_sc)
        # fig = plt.figure()
        # plt.matshow(heatmap)
        # plt.savefig(spinet.path + "figures/1/" + str(c), bbox_inches="tight")
        # plt.close(fig)
        Image.fromarray(heatmap_rf.astype("uint8")).save(
            spinet.path + "figures/2/" + str(c) + "_rf.png"
        )

def pdf_complex_receptive_fields(spinet, layer):
    val = 30
    for c, complex_cell in enumerate(spinet.neurons[layer]):
        if(c > 31):
            break
        ox, oy, oz = complex_cell.params["offset"]

        heatmap = np.zeros((spinet.n_shape[layer, 0, 0], spinet.n_shape[layer, 0, 1]))
        dimrf = 120 # 120
        heatmap_rf = np.zeros((dimrf, dimrf, 3))
        heatmap_rf2 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf3 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf4 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf5 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf6 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf7 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf8 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf9 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf10 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf11 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf12 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf13 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf14 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf15 = np.zeros((dimrf, dimrf, 3))
        heatmap_rf16 = np.zeros((dimrf, dimrf, 3))

        maximum = np.max(complex_cell.weights)
        minimum = np.min(complex_cell.weights)
        
        for i in range(ox, ox + spinet.n_shape[1, 0, 0]):
            for j in range(oy, oy + spinet.n_shape[1, 0, 1]):
                once =True
                for k in range(spinet.n_shape[layer, 0, 2]):
                    simple_cell = spinet.neurons[0][spinet.layout[0][i, j, k]]
                    xs, ys, zs = simple_cell.params["position"]

                    weight_sc = ((complex_cell.weights[xs - ox, ys - oy, k]) - minimum)/ (maximum-minimum)
                    heatmap[ys - oy, xs - ox] += weight_sc
                    # print(len(complex_cell.weights[xs - ox, ys - oy]))
                    indices = np.argsort(-complex_cell.weights[xs - ox, ys - oy])
                    sorted_values = complex_cell.weights[xs - ox, ys - oy][indices]
                    top = sorted_values[:16]
                    if(once):
                        print(top)
                        once = False
                    sup = 0
                    if np.argmax(complex_cell.weights[xs - ox, ys - oy]) == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # spinet.neurons[layer][c].weight_images.append(simple_cell.weight_images[0])
                        print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                            
                        #print(weight_sc)
                    if indices[1] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf2[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[2] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf3[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[3] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf4[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[4] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf5[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[5] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf6[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[6] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf7[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[7] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf8[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[8] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf9[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[9] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf10[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[10] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf11[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[11] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf12[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[12] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf13[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[13] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf14[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[14] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf15[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
                    if indices[15] == k:
                        sc_weight_image = Image.open(simple_cell.weight_images[0])
                        # print(np.array(sc_weight_image).shape)
                        # print(heatmap_rf.shape)
                        if(weight_sc > sup):
                            heatmap_rf16[val * (ys - oy): val * (ys - oy + 1), val * (xs - ox): val * (xs - ox + 1)] = np.array(
                                sc_weight_image) * weight_sc
                        #print(weight_sc)
        # fig = plt.figure()
        # plt.matshow(heatmap)
        # plt.savefig(spinet.path + "figures/1/" + str(c), bbox_inches="tight")
        # plt.close(fig)
        Image.fromarray(heatmap_rf.astype("uint8")).save(
            spinet.path + "figures/1/" + str(c) + "_rf.png"
        )
        spinet.neurons[layer][c].weight_images.append(spinet.path + "figures/1/" + str(c) + "_rf.png")

        Image.fromarray(heatmap_rf2.astype("uint8")).save(
            spinet.path + "figures/1/" + str(c) + "_rf2.png"
        )
        Image.fromarray(heatmap_rf3.astype("uint8")).save(
            spinet.path + "figures/1/" + str(c) + "_rf3.png"
        )
        Image.fromarray(heatmap_rf4.astype("uint8")).save(
            spinet.path + "figures/1/" + str(c) + "_rf4.png"
        )
        Image.fromarray(heatmap_rf5.astype("uint8")).save(
            spinet.path + "figures/1/" + str(c) + "_rf5.png"
        )
        Image.fromarray(heatmap_rf6.astype("uint8")).save(
            spinet.path + "figures/1/" + str(c) + "_rf6.png"
        )
        Image.fromarray(heatmap_rf7.astype("uint8")).save(
            spinet.path + "figures/1/" + str(c) + "_rf7.png"
        )
        Image.fromarray(heatmap_rf8.astype("uint8")).save(
            spinet.path + "figures/1/" + str(c) + "_rf8.png"
        )
        Image.fromarray(heatmap_rf9.astype("uint8")).save(
            spinet.path + "figures/1/" + str(c) + "_rf9.png"
        )
        Image.fromarray(heatmap_rf10.astype("uint8")).save(
            spinet.path + "figures/1/" + str(c) + "_rf10.png"
        )
        # Image.fromarray(heatmap_rf11.astype("uint8")).save(
        #     spinet.path + "figures/1/" + str(c) + "_rf11.png"
        # )
        # Image.fromarray(heatmap_rf12.astype("uint8")).save(
        #     spinet.path + "figures/1/" + str(c) + "_rf12.png"
        # )
        # Image.fromarray(heatmap_rf13.astype("uint8")).save(
        #     spinet.path + "figures/1/" + str(c) + "_rf13.png"
        # )
        # Image.fromarray(heatmap_rf14.astype("uint8")).save(
        #     spinet.path + "figures/1/" + str(c) + "_rf14.png"
        # )
        # Image.fromarray(heatmap_rf15.astype("uint8")).save(
        #     spinet.path + "figures/1/" + str(c) + "_rf15.png"
        # )
        # Image.fromarray(heatmap_rf16.astype("uint8")).save(
        #     spinet.path + "figures/1/" + str(c) + "_rf16.png"
        # )

def create_figures(spinet, spike_vector, angles, name):
    vectors = []
    for i in range(spike_vector.shape[1]):
        mean = mean_response(spike_vector[:-1, i], angles[:-1])
        vectors.append(mean)
        plt.figure()
        ax = plt.subplot(111, polar=True)
        # ax.set_title("Cell "+str(i))
        ax.plot(angles, spike_vector[:, i], "darkslategrey")
        ax.arrow(
            np.angle(mean),
            0,
            0,
            2 * np.abs(mean),
            width=0.02,
            head_width=0,
            head_length=0,
            length_includes_head=True,
            edgecolor="firebrick",
            lw=2,
            zorder=5,
        )
        ax.set_thetamax(360)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.setp(ax.get_xticklabels(), fontsize=15)
        plt.setp(ax.get_yticklabels(), fontsize=13)
        ax.set_xticklabels(["0°", "22.5°", "45°", "67.5°", "90°", "112.5°", "135°", "157.5°"])
        plt.savefig(spinet.path + "figures/" + name + "/" + str(i), bbox_inches="tight")
    return vectors


def mean_response(directions, angles):
    return np.mean(directions * np.exp(1j * angles))


def display_network(spinets):
    for spinet in spinets:
        spinet.generate_weight_images()

        if spinet.conf["sharingType"] == "patch":
            for i in range(spinet.conf["nbCameras"]):
                pdf = pdf_weight_sharing(spinet, 1, i)
                pdf.output(spinet.path + "figures/0/weight_sharing_" + str(i) + ".pdf", "F")
            if spinet.conf["nbCameras"] == 2:
                pdf = pdf_weight_sharing(spinet, spinet.conf["nbCameras"], 0)
                pdf.output(spinet.path + "figures/0/weight_sharing_combined.pdf", "F")
        elif spinet.conf["sharingType"] == "full":
            for i in range(spinet.conf["nbCameras"]):
                pdf = pdf_weight_sharing_full(spinet, 1, i)
                pdf.output(spinet.path + "figures/0/weight_sharing_" + str(i) + ".pdf", "F")
            if spinet.conf["nbCameras"] == 2:
                pdf = pdf_weight_sharing_full(spinet, spinet.conf["nbCameras"], 0)
                pdf.output(spinet.path + "figures/0/weight_sharing_combined.pdf", "F")
        elif spinet.conf["sharingType"] == "none":
            for layer in range(spinet.l_shape[0, 2]):
                for i in range(spinet.conf["nbCameras"]):
                    pdf = pdf_simple_cell(spinet, layer, i)
                    pdf.output(spinet.path + "figures/0/" + str(layer) + "_" + str(i) + ".pdf", "F")
                pdf = pdf_simple_cell_left_right_combined(spinet, layer)
                pdf.output(spinet.path + "figures/0/" + str(layer) + "_combined.pdf", "F")

        if len(spinet.neurons[1]) > 0:
            for z in range(spinet.l_shape[1, 2]):
                pdf = pdf_complex_cell(spinet, z)
                pdf.output(spinet.path + "figures/1/complex_weights_depth_" + str(z) + ".pdf", "F")

                # os.mkdir(spinet.path + "figures/1/tmp/")
                # pdf = pdf_complex_to_simple_cell_orientations(spinet, z, 1)
                # pdf.output(spinet.path + "figures/1/complex_weights" + str(z) + ".pdf", "F")
                # shutil.rmtree(spinet.path + "figures/1/tmp/")


        
