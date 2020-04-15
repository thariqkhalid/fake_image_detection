import os
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import cv2

TRAINING_DIR = "D:/Madiha Mariam Ahmed/Image Forgery Detection/phase-01-training/dataset-dist/phase-01/training/"
PRISTINE_DIR = TRAINING_DIR + 'pristine/'
FAKE_DIR = TRAINING_DIR + 'fake/'
MASKS_DIR = TRAINING_DIR + 'masks/'

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
    one_channel, three_channel, four_channel = get_stats(FAKE_DIR)
    print("one_channel, three_channel, four_channel")
    print(len(one_channel), len(three_channel), len(four_channel))

import shutil
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

fakes = os.listdir(FAKE_DIR)
pristines = os.listdir(PRISTINE_DIR)
masks = os.listdir(MASKS_DIR)

def get_final(folder_name):
    imgage_names = os.listdir(os.path.join(TRAINING_DIR, folder_name))
    imgs_full_path = [os.path.join(TRAINING_DIR, "{}/{}".format(folder_name, i)) for i in imgage_names]
    pristines_final = []
    fakes_final = []
    masks_final = []
    img_p = [cv2.imread(file) for file in glob.glob(PRISTINE_DIR + "*.png")]
    img_f = [cv2.imread(file) for file in glob.glob(FAKE_DIR + "*.png")]
    img_m = [cv2.imread(file) for file in glob.glob(MASKS_DIR + "*.png")]

    for img in img_p, img_m, img_f:
        if img.shape[2]==4 in img_p:
            pristines_final.append(img)
            return len(pristines_final)
        elif img.shape[2]==3 or img.shape[2]==4 in img_f:
            fakes_final.append(img)
            return len(fakes_final)
        elif img.shape[2]==3 or img.shape[2]==4 in img_m:
            masks_final.append(img)
            return len(masks_final)
get_final(PRISTINE_DIR)
















