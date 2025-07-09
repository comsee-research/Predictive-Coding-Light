import random
from PIL import Image, ImageDraw
import numpy as np

from src.events.Events import Events


def concatenate_files(aedat4_files):
    list_events = []
    last_tmsp = 0

    for i, file in enumerate(aedat4_files):
        events = Events(file).events
        if i != 0:
            events[:, 0] += last_tmsp - events[0][:, 0]
        last_tmsp = events[-1][:, 0]
        list_events.append(events)
    return np.hstack(list_events)


def concatenate_npz(event_files):
    list_events = []
    last_tmsp = 0

    for i, events in enumerate(event_files):
        events = Events(events).events
        if i != 0:
            events[:, 0] += last_tmsp - events[0, 0]
        last_tmsp = events[-1, 0]
        list_events.append(events)
    return Events(np.vstack(list_events))


def divide_events(events, chunk_size):
    first_timestamp = events["timestamp"][0]
    events["timestamp"] -= first_timestamp
    chunk = np.arange(0, events["timestamp"][-1], chunk_size)
    splits = [
        events[(events["timestamp"] > chunk[i]) & (events["timestamp"] < chunk[i + 1])]
        for i in range(chunk.size - 1)
    ]

    for split in splits:
        try:
            split["timestamp"] -= split["timestamp"][0]
        except:
            print("error spliting events")
    return splits, first_timestamp


def build_mixed_file(files, chunk_size):
    splits = []
    f_timestamps = []

    for file in files:
        div, first_timestamp = divide_events(Events(file).events, chunk_size)
        splits += div
        f_timestamps.append(first_timestamp)
    random.shuffle(splits)

    for i, split in enumerate(splits):
        split["timestamp"] += i * chunk_size + f_timestamps[0]

    return np.hstack(splits)


def remove_blank_space(aedat4_file, outfile, x_size, y_size):
    events = Events(aedat4_file).events
    times = events["timestamp"]

    diff = np.diff(times)
    arg = np.argwhere(diff > 1000000)[0][0]
    times[arg + 1:] -= diff[arg]

    return events


def show_event_images(events, time_gap, width, height, dest, rec, side):
    cnt = 0
    time = 0
    img = np.zeros((height, width, 3))

    for time in range(int(events[0, 0]), int(events[-1, 0]), time_gap):
        for event in events[(events[:, 0] >= time) & (events[:, 0] < time + time_gap)]:
            img[int(event[2]), int(event[1]), int(event[3])] = 1

        img = img.astype(np.uint8) * 255
        pilim = Image.fromarray(img)
        draw = ImageDraw.Draw(pilim)
        for x in rec[0]:
            for y in rec[1]:
                draw.rectangle([x, y, x + 31, y + 31], outline=(255, 255, 255, 0))
        pilim.save(dest + "img" + str(cnt) + side + "_" + str(time) + ".png")
        cnt += 1
        time += time_gap
        img = np.zeros((height, width, 3))


def rectify_events(events, lx, ly, rx, ry):
    events[0]["x"] += lx
    events[0]["y"] += ly
    events[1]["x"] += rx
    events[1]["y"] += ry

    l_events = events[0][
        (events[0]["x"] < 346)
        & (events[0]["x"] >= 0)
        & (events[0]["y"] < 260)
        & (events[0]["y"] >= 0)
        ]
    r_events = events[1][
        (events[1]["x"] < 346)
        & (events[1]["x"] >= 0)
        & (events[1]["y"] < 260)
        & (events[1]["y"] >= 0)
        ]

    return l_events, r_events
