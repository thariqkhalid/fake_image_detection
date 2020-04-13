import os
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import cv2

TRAINING_DIR = "dataset-dist/phase-01/training"
PRISTINE_DIR = "pristine"
FAKE_DIR ="fake"

def get_stats(folder_name):
    imgage_names = os.listdir(os.path.join(TRAINING_DIR,folder_name))
    imgs_full_path = [os.path.join(TRAINING_DIR, "{}/{}".format(folder_name,i)) for i in imgage_names]

    one_channel = []
    three_channel = []
    four_channel = []

    for ip, img_path in enumerate(imgs_full_path):
        img = imread(img_path)

        if len(img.shape) == 2:
            one_channel.append(img_path)
        elif img.shape[2] == 3:
            three_channel.append(img_path)
        elif img.shape[2] == 4:
            four_channel.append(img_path)

    return one_channel, three_channel, four_channel

if __name__ == "__main__":
    one_channel, three_channel, four_channel = get_stats(ORIGINAL_DIR)
    print("one_channel, three_channel, four_channel")
    print(len(one_channel), len(three_channel), len(four_channel))

