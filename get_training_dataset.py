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
NUM_OF_PIXELS = 300
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
                else: no_lesion.append([i,j])

        # random choose pixels to fit SAMPLE_SPACE
        lesion_sample = random.sample(lesion, NUM_OF_PIXELS)
        no_lesion_sample = random.sample(lesion, NUM_OF_PIXELS)
        return lesion_sample,no_lesion_sample
    return None


def get_training_set(case_id,slice_id):
    training_set = []
    lesion, no_lesion = get_lesion_pixels(case_id,slice_id)
    for x,y in lesion:
        sample = intpf.get_interpolate_sample(case_id,slice_id,x,y)
        norm = np.linalg.norm(sample)
        normalized = sample / norm
        training_set.append([1, normalized])

    for x,y in no_lesion:
        sample = intpf.get_interpolate_sample(case_id,slice_id,x,y)
        norm = np.linalg.norm(sample)
        normalized = sample / norm
        training_set.append([0, normalized])

    return training_set



