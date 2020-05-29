import os
import numpy as np
from sklearn import preprocessing as pp
import get_graph as gg
import generate_interpolation_func as intpf
import random
from sklearn import preprocessing
from matplotlib import pyplot as plt

CSV_NAME = "training_cvs"
IMG_SIZE = 256
NUM_OF_SELECTED_SAMPLE = 300
SAMPLE_SPACE = 200


def get_lesion_pixels(case_id,slice_id):
    lesion = []
    no_lesion = []
    file_name = os.path.join("dataset", "case_" + str(case_id), "OT",str(slice_id))
    if os.path.isfile(file_name):
        # data in 2-D format
        data = np.loadtxt(file_name)
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                if data[i,j] != 0.0:
                    lesion.append([i,j])
                else:
                    no_lesion.append([i,j])

        # random choose pixels to fit SAMPLE_SPACE
        lesion_sample_size = len(lesion)
        if lesion_sample_size > NUM_OF_SELECTED_SAMPLE:
            lesion_sample_size = NUM_OF_SELECTED_SAMPLE
        lesion_sample = random.sample(lesion,lesion_sample_size)

        no_lesion_sample_size = len(no_lesion)
        if no_lesion_sample_size > NUM_OF_SELECTED_SAMPLE:
            no_lesion_sample_size = NUM_OF_SELECTED_SAMPLE
        no_lesion_sample = random.sample(lesion, lesion_sample_size)
        return lesion_sample, no_lesion_sample
    return None


def get_training_set(case_id,slice_id):
    intensity_arr = []
    key = []
    lesion, no_lesion = get_lesion_pixels(case_id,slice_id)
    for x,y in lesion:
        sample = intpf.get_interpolate_sample(case_id,slice_id,x,y)
        intensity_arr.append(normalize(sample))
        key.append(1)

    for x,y in no_lesion:
        sample = intpf.get_interpolate_sample(case_id, slice_id, x, y)
        intensity_arr.append(normalize(sample))
        key.append(0)

    return intensity_arr, key



def normalize(sample):
    norm = np.linalg.norm(sample)
    return sample / norm


