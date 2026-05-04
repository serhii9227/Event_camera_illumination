import os
from os.path import join

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

os.sep = '/'
np.set_printoptions(suppress=True)


def stack_event(stack_name, index, root, T_num, stack_amount_1c2c, stack_amount_3c, decay_rate_3c):
    """Process events from CSV and save as GTP stacked images."""

    seq_name = root.split('/')[-1]
    img_path = os.path.join(root, 'img').replace('\\', '/')
    frame_num = len(os.listdir(img_path))

    # Read events from CSV
    csv_path = os.path.join(root, 'events.csv').replace('\\', '/')
    df = pd.read_csv(csv_path)
    events = df[['t', 'x', 'y', 'p']].values.astype(np.int64)

    # Get image shape from first frame
    first_img = cv2.imread(os.path.join(img_path, sorted(os.listdir(img_path))[0]))
    h, w = first_img.shape[:2]
    pic_shape = (h, w)

    # Generate evenly spaced frame timestamps
    t_min = int(events[0, 0])
    t_max = int(events[-1, 0])
    time_series = list(np.linspace(t_min, t_max, frame_num + 1, dtype=np.int64))

    # Filter events to relevant time range
    events = events[(events[:, 0] >= time_series[0]) & (events[:, 0] < time_series[-1])]

    # Create output folder
    stack_path = os.path.join(root, stack_name).replace('\\', '/')
    if not os.path.exists(stack_path):
        os.mkdir(stack_path)

    deal_event(index, events, time_series, pic_shape, stack_path, T_num, stack_amount_1c2c, stack_amount_3c, decay_rate_3c)


def process_event(pos_img, neg_img, event, pic_shape, stack_amount_1c2c):
    """Accumulate a single event into pos or neg image buffer."""
    x, y, p = int(event[1]), int(event[2]), int(event[3])
    if 0 < x < pic_shape[1] and 0 < y < pic_shape[0]:
        if p == 1:
            pos_img[y][x] = min(255, pos_img[y][x] + stack_amount_1c2c)
        else:
            neg_img[y][x] = min(255, neg_img[y][x] + stack_amount_1c2c)


def save_gtp_img(pos_img, neg_img, hidden_img, path):
    """Save 3-channel GTP image (pos, neg, trajectory)."""
    img = np.zeros((pos_img.shape[0], pos_img.shape[1], 3), dtype=np.uint8)
    img[:, :, 0] = pos_img
    img[:, :, 1] = neg_img
    img[:, :, 2] = hidden_img
    cv2.imwrite(path, img)


def update_hidden(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_pic, stack_amount_3c, decay_rate_3c):
    """Update global trajectory channel with decay."""
    new_hidden = hidden_pic * decay_rate_3c
    new_hidden[(last_pos_pic == 0) & (pos_img != 0)] += stack_amount_3c
    new_hidden[(last_neg_pic == 0) & (neg_img != 0)] += stack_amount_3c
    return np.clip(new_hidden, 0, 255)


def deal_event(index, events, frame_timestamp, pic_shape, save_name, T_num, stack_amount_1c2c, stack_amount_3c, decay_rate_3c):
    """Split events into per-frame GTP images with T_num sub-frames and save."""

    flag = False
    last_pos_pic = np.zeros(pic_shape, dtype=np.uint8)
    last_neg_pic = np.zeros(pic_shape, dtype=np.uint8)
    hidden_state = np.zeros(pic_shape, dtype=np.uint8)

    i = 1
    pos_img = np.zeros(pic_shape, dtype=np.uint8)
    neg_img = np.zeros(pic_shape, dtype=np.uint8)
    sub_index = 1

    # T_num+1 points → T_num intervals
    sub_frame = np.linspace(frame_timestamp[0], frame_timestamp[1], T_num + 1, dtype=np.int64)

    for event in tqdm(events, desc="{} Writing {}".format(index, save_name.split('/')[-2])):

        if event[0] >= frame_timestamp[i]:
            # Crossed into next frame — save current sub-frame
            img_path = save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.png'
            if flag:
                hidden_state = update_hidden(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_state, stack_amount_3c, decay_rate_3c)
            else:
                flag = True
            last_pos_pic = pos_img
            last_neg_pic = neg_img
            save_gtp_img(pos_img, neg_img, hidden_state, img_path)

            i += 1
            if i >= len(frame_timestamp):
                break

            sub_frame = np.linspace(frame_timestamp[i - 1], frame_timestamp[i], T_num + 1, dtype=np.int64)
            pos_img = np.zeros(pic_shape, dtype=np.uint8)
            neg_img = np.zeros(pic_shape, dtype=np.uint8)
            sub_index = 1

        elif event[0] < frame_timestamp[i]:
            # Check if we crossed a sub-frame boundary
            if sub_index < T_num and event[0] >= sub_frame[sub_index]:
                img_path = save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.png'
                if flag:
                    hidden_state = update_hidden(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_state, stack_amount_3c, decay_rate_3c)
                else:
                    flag = True
                last_pos_pic = pos_img
                last_neg_pic = neg_img
                save_gtp_img(pos_img, neg_img, hidden_state, img_path)

                pos_img = np.zeros(pic_shape, dtype=np.uint8)
                neg_img = np.zeros(pic_shape, dtype=np.uint8)
                sub_index += 1

            process_event(pos_img, neg_img, event, pic_shape, stack_amount_1c2c)

    # Save last sub-frame
    img_path = save_name + '/' + str(i).zfill(4) + '_' + str(sub_index) + '.png'
    hidden_state = update_hidden(pos_img, neg_img, last_pos_pic, last_neg_pic, hidden_state, stack_amount_3c, decay_rate_3c)
    save_gtp_img(pos_img, neg_img, hidden_state, img_path)


def stack_dataset(root, stack_name, T_num, stack_amount_1c2c, stack_amount_3c, decay_rate_3c):
    """Process all sequences listed in test.txt."""
    text_root = os.path.join(root, "test.txt")
    if not os.path.exists(text_root):
        return

    file_name_list = []
    with open(text_root, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                file_name_list.append(line)

    for index, i in enumerate(sorted(file_name_list)):
        data = os.path.join(root, i).replace('\\', '/')
        stack_path = join(data, stack_name).replace('\\', '/')
        if os.path.exists(stack_path):
            img_dir = join(data, 'img').replace('\\', '/')
            if os.path.exists(img_dir):
                expected = T_num * len(os.listdir(img_dir))
                if expected == len(os.listdir(stack_path)):
                    print(f"Skipping {i} — already done")
                    continue
        stack_event(stack_name, index, data, T_num, stack_amount_1c2c, stack_amount_3c, decay_rate_3c)


def parse_args():
    parser = argparse.ArgumentParser(description='GTP preprocessing for CSV event data')
    parser.add_argument('--target_dir', type=str, required=True, help='Dataset root path')
    parser.add_argument('--stack_name', type=str, default='inter1_stack_3008', help='Output folder name')
    parser.add_argument('--T_num', type=int, default=1, help='Number of sub-frames per frame (1, 2 or 4)')
    parser.add_argument('--stack_amount_1c2c', type=float, default=30)
    parser.add_argument('--stack_amount_3c', type=float, default=30)
    parser.add_argument('--decay_rate_3c', type=float, default=0.8)
    return parser.parse_args()


args = parse_args()
target_dir = args.target_dir

train_root = os.path.join(target_dir, "train")
test_root = os.path.join(target_dir, "test")

stack_dataset(train_root, args.stack_name, args.T_num, args.stack_amount_1c2c, args.stack_amount_3c, args.decay_rate_3c)
stack_dataset(test_root, args.stack_name, args.T_num, args.stack_amount_1c2c, args.stack_amount_3c, args.decay_rate_3c)
