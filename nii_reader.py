import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
import generate_interpolation_func as intpf
from sklearn import preprocessing
from scipy import interpolate

IMG_SIZE = 256
NUM_OF_SELECTED_SAMPLE = 300
BATCH_DIR = "batch"
KEY_DIR = "key"
DATASET_DIR = "dataset"
SAMPLE_SIZE = 70
NUMBER_OF_PATIENTS = 91


def import_non_pwi():
    f = open("filenames.txt", "r")
    for line in f:
        trimmed_path = line.rstrip()
        splits = trimmed_path.rsplit("/", 3)
        case = splits[1]
        file_name_only = splits[3]
        if file_name_only.endswith(".nii"):
            file_name_only = file_name_only[:-4]

        file_type = "CT"
        if "CBF" in file_name_only:
            file_type = "CBF"
        elif "CBV" in file_name_only:
            file_type = "CBV"
        elif "MTT" in file_name_only:
            file_type = "MTT"
        elif "Tmax" in file_name_only:
            file_type = "Tmax"
        elif "OT" in file_name_only:
            file_type = "OT"

        data = nib.load(trimmed_path)
        img = data.get_fdata()
        shape = np.shape(img)
        num_of_slides = shape[2]
        path = os.path.join(DATASET_DIR, case, file_type)
        if not os.path.isdir(path):
            os.makedirs(path)
            print("generating non pwi at", path)
            for i in range(num_of_slides):
                slice = img[:, :, i]
                np.savetxt(os.path.join(path, str(i)), slice, fmt="%.6f")
    f.close()

#redudant
def import_pwi_img():
    f = open("pwi_filenames.txt", "r")
    for line in f:
        trimmed_path = line.rstrip()
        splits = trimmed_path.rsplit("/", 3)
        case = splits[1]

        data = nib.load(trimmed_path)
        img = data.get_fdata()
        shape = np.shape(img)
        for slice_num in range(shape[2]):
            for x in range(shape[0]):
                for y in range(shape[1]):
                    path = os.path.join("dataset", case, "PWI", "v" + str(slice_num))
                    if not os.path.isdir(path):
                        os.makedirs(path)
                        print("generating pwi at", path)
                        np.savetxt(os.path.join(path, str(x) + "-" + str(y)), img[x, y, slice_num, :], fmt="%.6f")
    f.close()

#redudant
def import_pwi_by_case(case_id):
    if os.path.isdir(os.path.join("dataset", "case_" + str(case_id), "PWI")):
        return

    f = open("pwi_filenames.txt", "r")
    for line in f:
        trimmed_path = line.rstrip()
        splits = trimmed_path.rsplit("/", 3)
        id = splits[1]
        if id == "case_" + str(case_id):
            data = nib.load(trimmed_path)
            img = data.get_fdata()
            shape = np.shape(img)
            for slice_num in range(shape[2]):
                for x in range(shape[0]):
                    for y in range(shape[1]):
                        path = os.path.join("dataset", "case_" + str(case_id), "PWI", "v" + str(slice_num))
                        if not os.path.isdir(path):
                            os.makedirs(path)
                        print("generating pwi at", path)
                        file_name = os.path.join(path, str(x) + "-" + str(y))
                        if not os.path.isfile(file_name):
                            np.savetxt(file_name, img[x, y, slice_num, :], fmt='%.4f')
    f.close()


def import_batches():
    f = open("pwi_filenames.txt", "r")
    for line in f:
        trimmed_path = line.rstrip()
        splits = trimmed_path.rsplit("/", 3)
        id_part = splits[1]
        case_id = int(id_part[5:])

        if not os.path.isdir(BATCH_DIR):
            os.makedirs(BATCH_DIR)

        data = nib.load(trimmed_path).get_fdata()
        shape = np.shape(data)
        num_of_slices = shape[2]
        for slice_id in range(num_of_slices):
            batch_filename = os.path.join(BATCH_DIR, str(case_id) + "-" + str(slice_id))
            if os.path.isfile(batch_filename):
                break

            batch = np.zeros([SAMPLE_SIZE, ])
            for x in range(IMG_SIZE):
                print("interpolate_sample case %d - slice %d at x =  %d" % (case_id, slice_id, x))
                for y in range(IMG_SIZE):
                    intensity_arr = data[x, y, slice_id,]
                    intensity_arr[intensity_arr < 0] = 0
                    f = intpf.get_interpolate_func(intensity_arr, shape[3])
                    xnew = np.linspace(0, shape[3], num=SAMPLE_SIZE)
                    normalized = preprocessing.normalize(f(xnew).reshape(1,-1))
                    batch = np.append(batch, normalized)
            with open(batch_filename, "w+") as f:
                np.savetxt(f, batch, fmt="%.6f")



def import_key_by_case(case_id, slice_id):


    OT_file_name = os.path.join(DATASET_DIR, "case_" + str(case_id), "OT", str(slice_id))
    if os.path.isfile(OT_file_name):
        key_path = os.path.join(KEY_DIR, "case_" + str(case_id))
        if not os.path.isdir(key_path):
            os.makedirs(key_path)
        key_file_name = os.path.join(key_path, str(slice_id))
