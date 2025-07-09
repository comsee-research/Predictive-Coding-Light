#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:23:51 2022

@author: thomas
"""

from pathlib import Path

import h5py
import numpy as np
import skvideo.io
from dv import AedatFile
from tqdm import tqdm
import cv2 as cv


def render(x: np.ndarray, y: np.ndarray, pol: np.ndarray, height: int, width: int) -> np.ndarray:
    assert x.size == y.size == pol.size
    assert height > 0
    assert width > 0
    img = np.full((height, width, 3), fill_value=255, dtype="uint8")
    mask = np.zeros((height, width), dtype="int32")
    pol = pol.astype("int")
    x = x.astype("int")
    y = y.astype("int")
    pol[pol == 0] = -1
    mask1 = (x >= 0) & (y >= 0) & (width > x) & (height > y)
    mask[y[mask1], x[mask1]] = pol[mask1]
    img[mask == 0] = [0, 0, 0]
    img[mask == -1] = [0, 255, 0]
    img[mask == 1] = [255, 0, 0]
    return img


class Events:
    """Events class:
    Container for event based data.
    """

    def __init__(self, *args):
        """
        Parameters
        ----------
        *args : string, or list/tuple of string.
            The event file.
            Input supported:
                - hdf5 file format -> formatted as:
                    group named "events"
                    4 dataset in that group for the timestamps, x, y and polarities
                    possibly a 5th dataset which indicates which camera it belongs to (0 for left, 1 for right)
                - npz format -> formatted as:
                    5 arrays (arr_0, arr_1, arr_2, arr_3, arr_4) for the timestamps, x, y, polarities and cameras.
                - ndarray -> formatted as:
                    np.dtype([("t", "<u4"), ("x", "<u2"), ("y", "<u2"), ("p", "u1"), ("c", "u1")])
                - aedat4 format
            
            example:
                events = Events("path/to/events.h5") # 1 event file
                
                or
                
                events = Events(["path/to/events1.h5", "path/to/events2.npz"]) # 2 event files concatenated
                
                or
                
                events = Events(["path/to/left_events.h5", "path/to/right_events.h5"], [0, 1]) # a stereo event file
        Returns
        -------
        None.

        """
        self.dtype = np.dtype([("t", "<u4"), ("x", "<u2"), ("y", "<u2"), ("p", "u1"), ("c", "u1")])
        self.event_array = np.zeros(0, self.dtype)

        if len(args) > 1:
            if isinstance(args[0], list) or isinstance(args[0], tuple):
                for event_file, camera in zip(args[0], args[1]):
                    self.add_events(event_file, camera)
            else:
                print("Invalid arguments")
        elif isinstance(args[0], str) or isinstance(args[0], np.ndarray):
            self.add_events(args[0])
        elif isinstance(args[0], list) or isinstance(args[0], tuple):
            for event_file in args[0]:
                self.add_events(event_file)
        else:
            print("Invalid arguments")

    def add_events(self, event_file, camera=0):
        if isinstance(event_file, np.ndarray):
            self.event_array = np.hstack((self.event_array, event_file))
        if isinstance(event_file, str):
            if event_file.endswith(".npz"):
                self.load_npz(event_file)
            elif event_file.endswith(".h5"):
                self.load_hdf5(event_file, camera)
            elif event_file.endswith(".aedat4"):
                self.load_aedat4(event_file, camera)

    def load_hdf5(self, filepath, camera):
        with h5py.File(str(filepath), "r") as file:
            event_dataset = file["events"]

            event_array = np.zeros(event_dataset["t"].size, self.dtype)
            event_array["t"] = np.asarray(event_dataset["t"])
            event_array["x"] = np.asarray(event_dataset["x"])
            event_array["y"] = np.asarray(event_dataset["y"])
            event_array["p"] = np.asarray(event_dataset["p"])
            if "c" in event_dataset.keys():
                event_array["c"] = np.asarray(event_dataset["c"])
            else:
                event_array["c"].fill(camera)

        self.event_array = np.hstack((self.event_array, event_array))

    def load_npz(self, filepath):
        with np.load(filepath) as npz:
            event_array = np.zeros(npz["arr_0"].shape[0], self.dtype)
            event_array["t"] = npz["arr_0"]
            event_array["x"] = npz["arr_1"]
            event_array["y"] = npz["arr_2"]
            event_array["p"] = npz["arr_3"]
            event_array["c"] = npz["arr_4"]

        self.event_array = np.hstack((self.event_array, event_array))

    def load_aedat4(self, filepath, camera):
        with AedatFile(filepath) as f:
            aedat4 = np.hstack([packet for packet in f["events"].numpy()])

        event_array = np.zeros(aedat4["timestamp"].size, self.dtype)
        event_array["t"] = aedat4["timestamp"]
        event_array["x"] = aedat4["x"]
        event_array["y"] = aedat4["y"]
        event_array["p"] = aedat4["polarity"]
        event_array["c"].fill(camera)

        del aedat4

        self.event_array = np.hstack((self.event_array, event_array))

    def sort_events(self):
        self.event_array = self.event_array[self.event_array["t"].argsort()]

    def save_as_file(self, dest):
        if dest.endswith(".npz"):
            np.savez(
                dest,
                self.event_array["t"].astype("i8"),
                self.event_array["x"].astype("i2"),
                self.event_array["y"].astype("i2"),
                self.event_array["p"].astype("i1"),
                self.event_array["c"].astype("i1"),
            )
        elif dest.endswith(".h5") or dest.endswith(".hdf5"):
            with h5py.File(dest, "a") as file:
                group = file.create_group("events")
                group.create_dataset("t", self.event_array["t"].shape, dtype=self.event_array["t"].dtype,
                                     data=self.event_array["t"], compression="gzip")
                group.create_dataset("x", self.event_array["x"].shape, dtype=self.event_array["x"].dtype,
                                     data=self.event_array["x"], compression="gzip")
                group.create_dataset("y", self.event_array["y"].shape, dtype=self.event_array["y"].dtype,
                                     data=self.event_array["y"], compression="gzip")
                group.create_dataset("p", self.event_array["p"].shape, dtype=self.event_array["p"].dtype,
                                     data=self.event_array["p"], compression="gzip")
                group.create_dataset("c", self.event_array["c"].shape, dtype=self.event_array["c"].dtype,
                                     data=self.event_array["c"], compression="gzip")

    def to_video(self, dt_miliseconds, dest, width, height):
        # rate = 50
        # writer = skvideo.io.FFmpegWriter(Path(dest + ".gif"), inputdict={
        #       '-r': str(rate),
        #     },
        #     outputdict={
        #       '-r': str(rate),
        # })
        m_x = 0#141
        m_y = 0#96
        pad = 0 #2 #1 #2
        overlap = 3
        writer = skvideo.io.FFmpegWriter(Path(dest + ".gif"))
        ct = 0
        for events in tqdm(EventSlicer(self.get_events(), dt_miliseconds)):
            img = render(events["x"], events["y"], events["p"], height, width)

            img = cv.rectangle(img,(128-m_x-pad,128-m_y-pad),(148-m_x+pad,148-m_y+pad),(255,255,255),1)

            writer.writeFrame(img)
            ct+=1
        writer.close()

    def to_pic(self, dt_miliseconds, dest, width, height):
        writer = skvideo.io.FFmpegWriter(Path(dest + ".mp4"))
        for events in tqdm(EventSlicer(self.get_events(), dt_miliseconds)):
            img = render(events["x"], events["y"], events["p"], height, width)
            img = cv.rectangle(img,(150,100),(216,166),(0,255,0),1)
            writer.writeFrame(img)
        writer.close()

    def remove_events(self, timestamp_starts, timestamp_end):
        for i, j in zip(timestamp_starts, timestamp_end):
            self.event_array = np.delete(
                self.event_array, (self.event_array["t"] >= i) & (self.event_array["t"] <= j)
            )

    def resize_events(self, w_start, h_start, width, height):
        self.event_array = np.delete(self.event_array,
                                     (self.event_array["x"] < w_start) | (self.event_array["x"] >= w_start + width) |
                                     (self.event_array["y"] < h_start) | (self.event_array["y"] >= h_start + height),
                                     axis=0)
        self.event_array["x"] -= np.min(self.event_array["x"])
        self.event_array["y"] -= np.min(self.event_array["y"])
        
    def oversample(self, times):
        # ev = np.repeat(self.event_array,times)
        # self.event_array = ev
        # self.sort_events()
        
        ev = []
        temp = np.copy(self.event_array)
        timestamp_noise = 50
        for i in range(times-1):
            ev.append(temp)
            for j in range(len(ev[i])):
                rd = -10000
                while(rd + np.int(ev[i][j]['t']) <0 or rd==-10000):
                    rd = np.random.randint(-timestamp_noise // 2, timestamp_noise // 2 + 1)
                ev[i][j]['t'] += rd

        for i in range(times-1):
            self.event_array = np.append(self.event_array, ev[i])
        self.sort_events()
        
    def centering(self, actual_center_x, actual_center_y, center_width = 172, center_height = 129):
        self.event_array["x"] = self.event_array["x"].astype(np.float)
        self.event_array["y"] = self.event_array["y"].astype(np.float)
        
        difference_x = center_width - actual_center_x
        difference_y = center_height - actual_center_y
        
        self.event_array["x"] = (self.event_array["x"] + difference_x ) 
        self.event_array["y"] = (self.event_array["y"] + difference_y )

        self.event_array["x"] = self.event_array["x"].astype("<i2")
        self.event_array["y"] = self.event_array["y"].astype("<i2")
        
    def translate(self, tr_x, tr_y): # not here initially
        self.event_array["x"] = self.event_array["x"].astype(np.float)
        self.event_array["y"] = self.event_array["y"].astype(np.float)
        self.event_array["x"] = (self.event_array["x"] + tr_x ) 
        self.event_array["y"] = (self.event_array["y"] + tr_y )

        self.event_array["x"] = self.event_array["x"].astype("<i2")
        self.event_array["y"] = self.event_array["y"].astype("<i2")

    def get_events(self):
        return self.event_array

    def get_timestamps(self):
        return self.event_array["t"]

    def get_x(self):
        return self.event_array["x"]

    def get_y(self):
        return self.event_array["y"]

    def get_polarities(self):
        return self.event_array["p"]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finalizer()

    def __iter__(self):
        return self.event_array.__iter__()

    def __next__(self):
        return self.event_array.__next__()

    def _finalizer(self):
        pass


class EventSlicer:
    def __init__(self, events, dt_milliseconds: int):
        self.events = events
        self.dt_us = int(dt_milliseconds * 1000)
        self.t_start_us = self.events["t"][0]
        self.t_end_us = self.events["t"][-1]

        self._length = (self.t_end_us - self.t_start_us) // self.dt_us

    def __len__(self):
        return self._length

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._finalizer()

    def __iter__(self):
        return self

    def __next__(self):
        t_end_us = self.t_start_us + self.dt_us
        if t_end_us > self.t_end_us:
            raise StopIteration
        events = self.get_events(self.t_start_us, t_end_us)
        if events is None:
            raise StopIteration

        self.t_start_us = t_end_us
        return events

    def get_events(self, t_start_us: int, t_end_us: int):
        return self.events[(self.events["t"] > t_start_us) & (self.events["t"] < t_end_us)]

    def _finalizer(self):
        pass
