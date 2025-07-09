#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 12:47:24 2020

@author: alphat
"""

import os
import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fpdf import FPDF
from natsort import natsorted

from src.spiking_network.network.neuvisys import SpikingNetwork

# %%

# // Step Window
# // if (m_networkConf.STDP == "step_sym") {
# // if (static_cast < double > (m_spikingTime - event.timestamp()) < m_networkConf.TAU_LTP & & static_cast < double > (m_spikingTime - event.timestamp()) >= 0) {
# // m_weights(event.m_jitterSpeed(), event.y(), event.z()) += m_learningDecay * m_networkConf.ETA_LTP;
# //}
# // if (static_cast < double > (event.timestamp() - m_lastSpikingTime) < m_networkConf.TAU_LTD & & static_cast < double > (event.timestamp() - m_lastSpikingTime) >= 0) {
# // m_weights(event.m_jitterSpeed(), event.y(), event.z()) += m_learningDecay * m_networkConf.ETA_LTD;
# //}
# //} else if (m_networkConf.STDP == "step_left") {
# // if (static_cast < double > (event.timestamp() - m_lastSpikingTime) < m_networkConf.TAU_LTD & & static_cast < double > (event.timestamp() - m_lastSpikingTime) >= 0) {
# // m_weights(event.m_jitterSpeed(), event.y(), event.z()) += m_learningDecay * m_networkConf.ETA_LTD;
# //}
# //} else if (m_networkConf.STDP == "lin_sym") {
# // if (static_cast < double > (m_spikingTime - event.timestamp()) < m_networkConf.TAU_LTP & & static_cast < double > (m_spikingTime - event.timestamp()) >= 0) {
# // m_weights(event.m_jitterSpeed(), event.y(), event.z()) += m_learningDecay * m_networkConf.ETA_LTP * (1 - static_cast < double > (m_spikingTime - event.timestamp()));
# //}
# // if (static_cast < double > (event.timestamp() - m_lastSpikingTime) < m_networkConf.TAU_LTD & & static_cast < double > (event.timestamp() - m_lastSpikingTime) >= 0) {
# // m_weights(event.m_jitterSpeed(), event.y(), event.z()) += m_learningDecay * m_networkConf.ETA_LTD * (1 - static_cast < double > (event.timestamp() - m_lastSpikingTime));
# //}
# //} else if (m_networkConf.STDP == "exp_sym") {
# // if (static_cast < double > (m_spikingTime - event.timestamp()) >= 0) {
# // m_weights(event.m_jitterSpeed(), event.y(), event.z()) += m_learningDecay * m_networkConf.ETA_LTP * exp(- static_cast < double > (m_spikingTime - event.timestamp()) / m_networkConf.TAU_LTP);
# //}
# // if (static_cast < double > (event.timestamp() - m_lastSpikingTime) >= 0) {
# // m_weights(event.m_jitterSpeed(), event.y(), event.z()) += m_learningDecay * m_networkConf.ETA_LTD * exp(- static_cast < double > (event.timestamp() - m_lastSpikingTime) / m_networkConf.TAU_LTD);
# //}
# //}


# %%

tau_k = 0.2
nu_k = 0.05
t = np.linspace(0, 2, 10000)


def gaussian(a, b, c, x):
    return a * np.exp(-0.5 * (x - b) ** 2 / c ** 2)


def gaussian2D(x, y, A, x0, y0, sigX, sigY):
    return A * np.exp(-(((x - x0) ** 2 / 2 * sigX ** 2) + (y - y0) ** 2 / 2 * sigY ** 2))


def kernel(time):
    return (np.exp(-time / tau_k) - np.exp(-time / nu_k)) / (tau_k - nu_k)


def kernel_dot(time):
    return (np.exp(-time / nu_k) / nu_k - np.exp(-time / tau_k) / tau_k) / (tau_k - nu_k)


plt.figure()
plt.plot(t, kernel(t))
plt.plot(t, kernel_dot(t))
plt.show()

# %%

network_path = "/home/thomas/neuvisys-dv/configuration/network/"
spinet = SpikingNetwork(network_path)
array = np.array(spinet.state["nb_events"])
plt.figure()
plt.plot(array[100:800])

# %%

Y = np.cumsum(np.random.exponential(np.random.uniform(0, 10), 100))
for i in range(9):
    Y = np.concatenate((Y, np.cumsum(np.random.exponential(np.random.uniform(0, 10), 100)) + Y[-1]))

Y = np.array(Y) / 1000

signal = []
signal_dot = []
for time in np.linspace(0, Y[-1], 1000):
    value = 0
    value_dot = 0
    for spike in Y[Y <= time]:
        value += kernel(time - spike)
        value_dot += kernel_dot(time - spike)
    signal.append(value)
    signal_dot.append(value_dot)

V = np.array(signal)
V_dot = np.array(signal_dot)
plt.figure()
plt.plot(Y)

plt.figure()
plt.plot(V)
plt.plot(V_dot)
# plt.plot(V_dot - V)
plt.show()

# %%

network_path = ""
spinet = SpikingNetwork(network_path)

# %%

state = np.array(spinet.state["td"])

plt.figure()
plt.title("Reward (blue) vs Value (orange)")
plt.plot(state[:, 0], state[:, 1])
plt.plot(state[:, 0], state[:, 2])

plt.figure()
plt.title("Reward (blue) vs Value Derivative (orange)")
plt.plot(state[:, 0], state[:, 1])
plt.plot(state[:, 0], state[:, 3])

plt.figure()
plt.title("Reward (blue) vs TD error (orange)")
plt.plot(state[:, 0], state[:, 1])
plt.plot(state[:, 0], state[:, 4])

plt.figure()
plt.title("Value (blue) vs taur * Value Derivative (orange)")
plt.plot(state[:, 0], state[:, 2])
plt.plot(state[:, 0], -1 * state[:, 3])

# %%

row = 6
col = 6
im_size = (20, 20)
im_size_pad = (22, 22)

pdf = FPDF("P", "mm", (col * im_size_pad[0], row * im_size_pad[1]))
pdf.add_page()

path = "/home/alphat/Desktop/complex_selectivity/figures/complex_directions/"

images = natsorted(os.listdir(path))  # [::-1]
random.shuffle(images)

count = 0
for i in range(row):
    for j in range(col):
        pdf.image(
            path + images[count],
            x=j * im_size_pad[0],
            y=i * im_size_pad[1],
            w=im_size[0],
            h=im_size[1],
        )
        count += 1

pdf.output("/home/alphat/Desktop/images/directions.pdf", "F")

# %%

des = np.load("/home/alphat/Desktop/images/dir_exp_sym.npy")
oes = np.load("/home/alphat/Desktop/images/ori_exp_sym.npy")
dls = np.load("/home/alphat/Desktop/images/dir_lin_sym.npy")
ols = np.load("/home/alphat/Desktop/images/ori_lin_sym.npy")
dsl = np.load("/home/alphat/Desktop/images/dir_step_left.npy")
osl = np.load("/home/alphat/Desktop/images/ori_step_left.npy")
dss = np.load("/home/alphat/Desktop/images/dir_step_sym.npy")
oss = np.load("/home/alphat/Desktop/images/ori_step_sym.npy")
dsr = np.load("/home/alphat/Desktop/images/dir_network.npy")
osr = np.load("/home/alphat/Desktop/images/ori_network.npy")

# fig, ax = plt.subplots(figsize=(16, 12))
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_visible(False)
# ax.yaxis.set_ticks_position("none")
# ax.grid(color="grey", axis="y", linestyle="-", linewidth=0.3, alpha=0.5)

# color1 = dict(color="#2C363F")
# color2 = dict(color="#9E7B9B")

# ax.set_ylabel("normalized vector length")
# ax.boxplot(
#     np.abs(des),
#     positions=[0],
#     labels=["direction exponential"],
#     boxprops=color1,
#     medianprops=color2,
#     whiskerprops=color1,
#     capprops=color1,
#     flierprops=dict(markeredgecolor=color1["color"]),
# )
# ax.boxplot(
#     np.abs(dls),
#     positions=[2],
#     labels=["direction linear"],
#     boxprops=color1,
#     medianprops=color2,
#     whiskerprops=color1,
#     capprops=color1,
#     flierprops=dict(markeredgecolor=color1["color"]),
# )
# ax.boxplot(
#     np.abs(dsr),
#     positions=[4],
#     labels=["direction step left"],
#     boxprops=color1,
#     medianprops=color2,
#     whiskerprops=color1,
#     capprops=color1,
#     flierprops=dict(markeredgecolor=color1["color"]),
# )
# ax.boxplot(
#     np.abs(dsl),
#     positions=[6],
#     labels=["direction step right"],
#     boxprops=color1,
#     medianprops=color2,
#     whiskerprops=color1,
#     capprops=color1,
#     flierprops=dict(markeredgecolor=color1["color"]),
# )
# ax.boxplot(
#     np.abs(dss),
#     positions=[8],
#     labels=["direction step sym"],
#     boxprops=color1,
#     medianprops=color2,
#     whiskerprops=color1,
#     capprops=color1,
#     flierprops=dict(markeredgecolor=color1["color"]),
# )

fig, ax = plt.subplots(figsize=(14, 12))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.yaxis.set_ticks_position("none")
ax.grid(color="grey", axis="y", linestyle="-", linewidth=0.5, alpha=0.5)

ax.boxplot(
    np.abs(oes),
    positions=[1],
    labels=["Exponential"],
    boxprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    medianprops=dict(linestyle="-", linewidth=2, color="#9E7B9B"),
    whiskerprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    capprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(ols),
    positions=[3],
    labels=["Linear"],
    boxprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    medianprops=dict(linestyle="-", linewidth=2, color="#9E7B9B"),
    whiskerprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    capprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(osr),
    positions=[5],
    labels=["Step left"],
    boxprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    medianprops=dict(linestyle="-", linewidth=2, color="#9E7B9B"),
    whiskerprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    capprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(osl),
    positions=[7],
    labels=["Step right"],
    boxprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    medianprops=dict(linestyle="-", linewidth=2, color="#9E7B9B"),
    whiskerprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    capprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(oss),
    positions=[9],
    labels=["Step"],
    boxprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    medianprops=dict(linestyle="-", linewidth=2, color="#9E7B9B"),
    whiskerprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    capprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.set_ylabel("Normalized vector length", fontsize=26)
plt.setp(ax.get_xticklabels(), fontsize=26)
plt.setp(ax.get_yticklabels(), fontsize=26)
plt.savefig("/home/alphat/Desktop/images/stdp_windows.png", bbox_inches="tight")

# %%

fig, ax = plt.subplots(figsize=(8, 12))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.yaxis.set_ticks_position("none")
ax.grid(color="grey", axis="y", linestyle="-", linewidth=0.25, alpha=0.5)

color1 = dict(color="#2C363F")
color2 = dict(color="#9E7B9B")
ax.boxplot(
    np.abs(dir_vec),
    positions=[0],
    labels=["direction"],
    boxprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    medianprops=dict(linestyle="-", linewidth=2, color="#9E7B9B"),
    whiskerprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    capprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.boxplot(
    np.abs(ori_vec),
    positions=[1],
    labels=["orientation"],
    boxprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    medianprops=dict(linestyle="-", linewidth=2, color="#9E7B9B"),
    whiskerprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    capprops=dict(linestyle="-", linewidth=2, color="#2C363F"),
    flierprops=dict(markeredgecolor=color1["color"]),
)
ax.set_ylabel("Normalized vector length", fontsize=26)
plt.setp(ax.get_xticklabels(), fontsize=22)
plt.setp(ax.get_yticklabels(), fontsize=22)
plt.savefig("/home/alphat/Desktop/images/boxplot_ori_dir.png", bbox_inches="tight")

# %%

plt.figure(figsize=(8, 12))
ax = sns.histplot(
    (np.angle(ori_vec)) * 180 / np.pi, bins=np.linspace(-180, 180, 16), color="#2C363F"
)
ax.set_xlabel("Orientation (degree)", fontsize=22)
ax.set_ylabel("Count", fontsize=22)
ax.set_xticks(np.arange(-180, 181, 45))
# ax.set_xtickslabels(np.arange(-90, 91, 45))
plt.setp(ax.get_xticklabels(), fontsize=20)
plt.setp(ax.get_yticklabels(), fontsize=20)

plt.savefig("/home/alphat/Desktop/images/hist_ori.png", bbox_inches="tight")

# %%

eta_ltp = 0.2
eta_ltd = -0.2
tau_ltp = 20
tau_ltd = 20

t = np.linspace(-80, 80, 10000)

step = np.zeros(10000)
step[(t > -tau_ltd) & (t <= 0)] = -eta_ltd
step[(t >= 0) & (t < tau_ltd)] = eta_ltp

lin = np.zeros(10000)
lin[(t > -tau_ltd) & (t <= 0)] = (
        -eta_ltd
        * -t[(t > -tau_ltd) & (t <= 0)][::-1]
        / np.max(-t[(t > -tau_ltd) & (t <= 0)][::-1])
)
lin[(t >= 0) & (t < tau_ltd)] = (
        eta_ltp
        * t[(t >= 0) & (t < tau_ltd)][::-1]
        / np.max(t[(t >= 0) & (t < tau_ltd)][::-1])
)

exp = np.concatenate(
    (eta_ltd * np.exp(t[t < 0] / tau_ltd), eta_ltp * np.exp(-t[t >= 0] / tau_ltp))
)

fig, axs = plt.subplots(1, 3)

for ax in axs.flat:
    ax.grid(alpha=0.2, linestyle="--")
    ax.axhline(0, alpha=0.3, linestyle="-", color="k")
    ax.axvline(0, alpha=0.3, linestyle="-", color="k")
    ax.set_ylim(-0.3, 0.3)
    ax.set_ylabel("Synaptic change (mV)")
    ax.label_outer()

axs[1].set_xlabel("Spike timing (ms)")

axs[0].plot(t, step, color="k")
axs[0].set_title("Step")
axs[1].plot(t, lin, color="k")
axs[1].set_title("Linear")
axs[2].plot(t, exp, color="k")
axs[2].set_title("Exponential")

# %%

block_size = 3
min_disp = 0
num_disp = 16

stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    P1=8 * 1 * block_size ** 2,
    P2=32 * 1 * block_size ** 2,
    disp12MaxDiff=1,
    preFilterCap=0,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=1,
    mode=cv.StereoSGBM_MODE_HH,
)
for i in range(rect_frames.shape[1]):
    disparity = stereo.compute(rect_frames[0, i], rect_frames[1, i])
    cv.imwrite("/home/alphat/Desktop/disp/" + str(i) + ".jpg", disparity)
    # plt.figure()
    # plt.imshow(disparity, 'gray')
