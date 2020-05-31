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
BATCH_DIR = "batch"
KEY_DIR = "key"
DATASET_DIR = "dataset"
SAMPLE_SIZE = 300

def get_lession_value_of_a_slice(case_id,slice_id):
    values = []
    file_name = os.path.join(DATASET_DIR, "case_" + str(case_id), "OT", str(slice_id))
    if os.path.isfile(file_name):
        data = np.loadtxt(file_name)
        for x in range(IMG_SIZE):
            for y in range(IMG_SIZE):
                values.append(data[x,y])
    return values

def get_lesion_pixels(case_id,slice_id):
    lesion = []
    no_lesion = []
    file_name = os.path.join(DATASET_DIR, "case_" + str(case_id), "OT",str(slice_id))
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


def get_training_set_by_case(case_id,slice_id):
    print("getting training set by case %s slice %s",case_id,slice_id)
    intensity_arr = []
    key = []
    lesion, no_lesion = get_lesion_pixels(case_id,slice_id)
    if lesion == None and no_lesion == None:
        print("no info at case_id - slice_id:",case_id,slice_id)
        return None, None

    #randomly choose a sample of size NUM_OF_SELECTED_SAMPLE in population
    lesion_size = len(lesion)
    no_lesion_size = len(no_lesion)
    if lesion_size > NUM_OF_SELECTED_SAMPLE:
        lesion_size = NUM_OF_SELECTED_SAMPLE
    if no_lesion_size > NUM_OF_SELECTED_SAMPLE:
        no_lesion_size = NUM_OF_SELECTED_SAMPLE
    lesion = random.sample(lesion, lesion_size)
    no_lesion = random.sample(no_lesion, no_lesion_size)

    for x,y in lesion:
        sample = intpf.get_interpolate_sample(case_id,slice_id,x,y)
        intensity_arr.append(normalize(sample))
        key.append(1)

    for x,y in no_lesion:
        sample = intpf.get_interpolate_sample(case_id, slice_id, x, y)
        intensity_arr.append(normalize(sample))
        key.append(0)

    return intensity_arr, key



def generate_training_batch(start_case,end_case):
    print("generating batch from %d to %d",start_case,end_case)
    if not os.path.isdir(BATCH_DIR):
        os.makedir(BATCH_DIR)
    if not os.path.isdir(KEY_DIR):
        os.makedir(KEY_DIR)

    batch_name = str(start_case) + "-" + str(end_case)
    batch_path = os.path.join(BATCH_DIR,batch_name)
    key_path = os.path.join(KEY_DIR,batch_name)
    if os.path.isfile(batch_path):
        return

    for case_id in range(start_case,end_case+1):
        #find how many slices for per case_id
        pwi_dir = os.path.join(DATASET_DIR, "case_" + str(case_id),"PWI")
        num_of_slices = len(os.listdir(pwi_dir))
        print("case id - num of slices: ",case_id,num_of_slices)
        if num_of_slices == 0:
            return

        for slice_id in range(num_of_slices):
            batch, key = get_training_set_by_case(case_id,slice_id)
            if batch == None:
                break
            with open(batch_path, "w+") as f:
                np.savetxt(f,batch,fmt="%.6f")
            with open(key_path,"w+") as g:
                np.savetxt(g,key,fmt="%d")
            f.close()
            g.close()

def normalize(sample):
    #norm = np.linalg.norm(sample)
    #return sample / norm
    return preprocessing.normalize(sample.reshape(1,-1))

def get_testing_set(case_id,slice_id):
    testing_set = []
    testing_key = []

    path = os.path.join(DATASET_DIR,"case_" + str(case_id),"PWI","v" + str(slice_id))
    if os.path.isdir(path):
        for i in range(150,151):
            for j in range(IMG_SIZE):
                file_name = str(i)  + "-" + str(j)
                file_path = os.path.join(path,file_name)
                if os.path.isfile(file_path):
                    print("get_interpolate_sample",case_id,slice_id,i,j)
                    intensity_arr = intpf.get_interpolate_sample(case_id,slice_id,i,j)
                    normalized = normalize(intensity_arr)
                    testing_set.append(normalized)

                key_filename = os.path.join(DATASET_DIR, "case_" + str(case_id), "OT", str(slice_id))
                data = np.loadtxt(key_filename)
                if data[i, j] != 0.0:
                    testing_key.append(1)
                else:
                    testing_key.append(0)

    return testing_set,testing_key