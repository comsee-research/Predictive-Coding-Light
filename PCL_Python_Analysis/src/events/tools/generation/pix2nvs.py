#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 04:47:53 2020

@author: alphat
"""

import os

import numpy as np
from PIL import Image
from natsort import natsorted


class Pix2Eve:
    """Transform frames into an event stream
    time_gap: time between each frames
    log_threshold: pixel luminance intensity threshold at which events are generated
    map_threshold:
    n_max: maximum number of events generated by a pixel in between two frames
    adapt_thresh_coef_shift
    timestamp_noise: time jittering range of the event timestamps in microseconds
    """

    def __init__(
            self,
            time_gap,
            log_threshold=20,
            map_threshold=0.4,
            min_threshold = 0.02,
            n_max=5,
            adapt_thresh_coef_shift=0.05,
            timestamp_noise=20
    ):
        self.time_gap = time_gap
        # self.update_method = update_method
        self.log_threshold = log_threshold
        self.map_threshold = map_threshold
        self.min_threshold = min_threshold
        self.n_max = n_max
        self.adapt_thresh_coef_shift = adapt_thresh_coef_shift
        self.timestamp_noise = timestamp_noise
        self.event_file = "/home/alphat/Desktop/events.npy"

    def write_event(self, events, delta_b, thresh, frame_id, x, y, polarity):
        if( (delta_b/thresh)==float("inf")):
            moddiff = self.n_max +1
        else:
            moddiff = int(delta_b / thresh)
        if moddiff > self.n_max:
            nb_event = self.n_max
        else:
            nb_event = moddiff

        for e in range(nb_event):
            timestamp = int(
                ((self.time_gap * (e + 1) * thresh) / delta_b)*0
                + self.time_gap * frame_id
                + np.random.randint(-self.timestamp_noise // 2, self.timestamp_noise // 2 + 1)
            )
            #print((self.time_gap * (e + 1) * thresh) / delta_b)
            if timestamp < 0:
                timestamp = 0
            events.append([timestamp, x, y, polarity])

        return nb_event

    def convert_frame(self, frame):
        frame = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        np.log(frame, out=frame, where=frame > self.log_threshold)
        return frame

    def frame_to_events(self, frame_id, frame, reference, threshold_map, events):
        delta = frame - reference

        for i, j in zip(*np.nonzero(delta > threshold_map)):
            #if((i>=150 and i<=216) and (j>=100 and j <= 166)):
            if((i>=128 and i<=383) and (j>=128 and j <= 383)):
            # if((i>=218 and i<=473) and (j>=132 and j <= 387)):
            #if(True):
                self.write_event(
                    events, delta[i, j], threshold_map[i, j], frame_id, i, j, 1
                )

        for i, j in zip(*np.nonzero(delta < -threshold_map)):
            # if((i>=150 and i<=216) and (j>=100 and j <= 166)):
            if((i>=128 and i<=383) and (j>=128 and j <= 383)):
            # if((i>=218 and i<=473) and (j>=132 and j <= 387)):
            #if(True):
                self.write_event(
                    events, -delta[i, j], threshold_map[i, j], frame_id, i, j, 0
                )
        threshold_map[(delta > threshold_map) | (delta < -threshold_map)] *= (
                1 + self.adapt_thresh_coef_shift
        )
        threshold_map[(delta <= threshold_map) & (delta >= -threshold_map)] *= (
                1 - self.adapt_thresh_coef_shift
        )
        threshold_map[threshold_map < self.min_threshold ] = self.min_threshold
        #print(threshold_map)

    def run(self, folder):
        events = []
        # threshold_map = np.full((346, 260), self.map_threshold)
        threshold_map = np.full((512,512), self.map_threshold)

        frames = natsorted(os.listdir(folder))
        reference = self.convert_frame(
            np.asarray(Image.open(folder + frames[0])).transpose(1, 0, 2)
        )

        for frame_id, frame in enumerate(frames[1:]):
            frame = self.convert_frame(
                np.asarray(Image.open(folder + frame)).transpose(1, 0, 2)
            )
            self.frame_to_events(frame_id, frame, reference, threshold_map, events)
            reference = frame
            if (100 * frame_id / len(frames) % 5) == 0:
                print(str(100 * frame_id / len(frames)) + "%...")

        # print("Finished conversion")
        return np.array(events, dtype=np.float64)
