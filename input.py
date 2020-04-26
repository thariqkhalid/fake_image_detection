import os
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
import cv2
import shutil
from sklearn.model_selection import train_test_split
import seaborn as sns

TRAINING_DIR = "dataset-dist/phase-01/training/"
PRISTINE_DIR = TRAINING_DIR + 'pristine/'
FAKE_DIR = TRAINING_DIR + 'fake/'
fakes = os.listdir(FAKE_DIR)
pristines = os.listdir(PRISTINE_DIR)
labels = [0]*1050 + [1]*450
image_names = pristines
image_names = image_names.append(fakes)

def get_stats(folder_name):
    imgage_names = os.listdir(os.path.join(TRAINING_DIR,folder_name))
    imgs_full_path = [os.path.join(TRAINING_DIR, "{}/{}".format(folder_name,i)) for i in imgage_names]

    one_channel = []
    three_channel = []
    four_channel = []

    for ipidx, img_path in enumerate(imgs_full_path):
        img = imread(img_path)

        if len(img.shape) == 2:
            one_channel.append(img_path)
        elif img.shape[2] == 3:
            three_channel.append(img_path)
        elif img.shape[2] == 4:
            four_channel.append(img_path)

    return one_channel, three_channel, four_channel

def data_split():
    fakes = list(os.listdir(FAKE_DIR)[1:])
    pristines = list(os.listdir(PRISTINE_DIR))
    image_names = pristines + fakes
    labels = [0]*450 + [1]*1050
    x_train, x_test, y_train, y_test = train_test_split(image_names, labels, test_size=0.2, random_state=0)



if __name__ == "__main__":
    data_split()
    one_channel, three_channel, four_channel = get_stats(FAKE_DIR)
    print("one_channel, three_channel, four_channel")
    print(len(one_channel), len(three_channel), len(four_channel))






