import os
import numpy as np
from sklearn import preprocessing as pp
import get_graph as gg
import generate_interpolation_func as intpf
import random
import fnmatch
from sklearn import preprocessing
from matplotlib import pyplot as plt

CSV_NAME = "training_cvs"
IMG_SIZE = 256
NUM_OF_VOXELS = 300
BATCH_DIR = "batch"
KEY_DIR = "key"
DATASET_DIR = "dataset"
SAMPLE_SIZE = 49


# redudant
def get_lession_value_of_a_slice(case_id, slice_id):
    values = []
    file_name = os.path.join(DATASET_DIR, "case_" + str(case_id), "OT", str(slice_id))
    if os.path.isfile(file_name):
        data = np.loadtxt(file_name)
        for x in range(IMG_SIZE):
            for y in range(IMG_SIZE):
                values.append(data[x, y])
    return values


# redudant
def get_lesion_pixels(case_id, slice_id):
    lesion = []
    no_lesion = []
    file_name = os.path.join(DATASET_DIR, "case_" + str(case_id), "OT", str(slice_id))
    if os.path.isfile(file_name):
        # data in 2-D format
        data = np.loadtxt(file_name)
        for i in range(IMG_SIZE):
            for j in range(IMG_SIZE):
                if data[i, j] != 0.0:
                    lesion.append([i, j])
                else:
                    no_lesion.append([i, j])

        # random choose pixels to fit SAMPLE_SPACE
        # lesion_sample_size = len(lesion)
        # if lesion_sample_size > NUM_OF_SELECTED_SAMPLE:
        #     lesion_sample_size = NUM_OF_SELECTED_SAMPLE
        # lesion_sample = random.sample(lesion,lesion_sample_size)
        #
        # no_lesion_sample_size = len(no_lesion)
        # if no_lesion_sample_size > NUM_OF_SELECTED_SAMPLE:
        #     no_lesion_sample_size = NUM_OF_SELECTED_SAMPLE
        # no_lesion_sample = random.sample(lesion, lesion_sample_size)

        return lesion, no_lesion
    return None, None


# redudant
def get_training_set_by_case(case_id, slice_id):
    print("getting training set by case %s slice %s", case_id, slice_id)
    intensity_arr = []
    key = []
    lesion, no_lesion = get_lesion_pixels(case_id, slice_id)
    if lesion is None and no_lesion is None:
        print("no info at case_id - slice_id:", case_id, slice_id)
        return None, None

    # randomly choose a sample of size NUM_OF_SELECTED_SAMPLE in population
    lesion_size = len(lesion)
    no_lesion_size = len(no_lesion)
    if lesion_size > NUM_OF_SELECTED_SAMPLE:
        lesion_size = NUM_OF_SELECTED_SAMPLE
    if no_lesion_size > NUM_OF_SELECTED_SAMPLE:
        no_lesion_size = NUM_OF_SELECTED_SAMPLE
    lesion = random.sample(lesion, lesion_size)
    no_lesion = random.sample(no_lesion, no_lesion_size)
    return lesion, no_lesion


# redudant
def get_training_set(case_id, slice_id):
    intensity_arr = []
    key = []
    lesion, no_lesion = get_lesion_pixels(case_id, slice_id)
    for x, y in lesion:
        sample = intpf.get_interpolate_sample(case_id, slice_id, x, y)
        intensity_arr.append(normalize(sample))
        key.append(1)

    for x, y in no_lesion:
        sample = intpf.get_interpolate_sample(case_id, slice_id, x, y)
        intensity_arr.append(normalize(sample))
        key.append(0)

    return intensity_arr, key


# redudant
def get_testing_set(case_id, slice_id):
    testing_set = []
    testing_key = []

    path = os.path.join(DATASET_DIR, "case_" + str(case_id), "PWI", "v" + str(slice_id))
    if os.path.isdir(path):
        for i in range(150, 151):
            for j in range(IMG_SIZE):
                file_name = str(i) + "-" + str(j)
                file_path = os.path.join(path, file_name)
                if os.path.isfile(file_path):
                    print("get_interpolate_sample", case_id, slice_id, i, j)
                    intensity_arr = intpf.get_interpolate_sample(case_id, slice_id, i, j)
                    normalized = normalize(intensity_arr)
                    testing_set.append(normalized)

                key_filename = os.path.join(DATASET_DIR, "case_" + str(case_id), "OT", str(slice_id))
                data = np.loadtxt(key_filename)
                if data[i, j] != 0.0:
                    testing_key.append(1)
                else:
                    testing_key.append(0)

    return testing_set, testing_key


# redudant
def generate_training_batch(start_case, end_case):
    print("generating batch from %d to %d", start_case, end_case)
    if not os.path.isdir(BATCH_DIR):
        os.makedir(BATCH_DIR)
    if not os.path.isdir(KEY_DIR):
        os.makedir(KEY_DIR)

    batch_name = str(start_case) + "-" + str(end_case)
    batch_path = os.path.join(BATCH_DIR, batch_name)
    key_path = os.path.join(KEY_DIR, batch_name)
    if os.path.isfile(batch_path):
        return

    for case_id in range(start_case, end_case + 1):
        # find how many slices for per case_id
        pwi_dir = os.path.join(DATASET_DIR, "case_" + str(case_id), "PWI")
        num_of_slices = len(os.listdir(pwi_dir))
        print("case id - num of slices: ", case_id, num_of_slices)
        if num_of_slices == 0:
            return

        for slice_id in range(num_of_slices):
            batch, key = get_training_set_by_case(case_id, slice_id)
            with open(batch_path, "w+") as f:
                np.savetxt(f, batch, fmt="%.6f")
            with open(key_path, "w+") as g:
                np.savetxt(g, key, fmt="%d")
            f.close()
            g.close()


# redudant
def normalize(sample):
    return preprocessing.normalize(sample.reshape(1, -1))


def combine_batches(start_case=1, end_case=10):
    batches = []
    keys = []
    size = 0
    files = os.listdir(BATCH_DIR)
    for case_id in range(start_case, end_case + 1):
        for file in files:
            if fnmatch.fnmatch(file, str(case_id) + "-*"):
                print("Combining File", file)
                batch_path = os.path.join(BATCH_DIR, file)
                key_path = os.path.join(KEY_DIR, file)
                if os.path.isfile(batch_path) and os.path.isfile(key_path):
                    batch = np.loadtxt(batch_path).reshape((-1,SAMPLE_SIZE))
                    key = np.loadtxt(key_path).flatten()
                    selected_batch, selected_key = get_randomly_sample_of_voxels(batch, key)
                    batches.extend(selected_batch)
                    keys.extend(selected_key)
                    size += 1

    return batches, keys, size


def get_randomly_sample_of_voxels(batch, key):
    lesion_id = []
    non_lesion_id = []
    size = len(key)
    lesion_size = 0
    non_lesion_size = 0
    for i in range(size):
        if key[i] == 0:
            non_lesion_id.append(i)
            non_lesion_size += 1
        else:
            lesion_id.append(i)
            lesion_size += 1

    lesion_size = NUM_OF_VOXELS if lesion_size > NUM_OF_VOXELS else lesion_size
    non_lesion_size = NUM_OF_VOXELS if non_lesion_size > NUM_OF_VOXELS else non_lesion_size
    selected_lesion_vowel = random.sample(lesion_id, lesion_size)
    selected_non_lesion_vowel = random.sample(non_lesion_id, non_lesion_size)
    selected_vowel = selected_lesion_vowel + selected_non_lesion_vowel
    selected_batch = []
    selected_key = []
    for i in selected_vowel:
        selected_batch.extend(batch[i])
        selected_key.append(key[i])

    return selected_batch, selected_key
