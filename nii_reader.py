import os
import numpy as np
import nibabel as nib
import generate_interpolation_func as intpf
from sklearn import preprocessing
from scipy import interpolate

IMG_SIZE = 256
NUM_OF_SELECTED_SAMPLE = 300
BATCH_DIR = "batch"
KEY_DIR = "key"
DATASET_DIR = "dataset"
SAMPLE_SIZE = 49
NUMBER_OF_PATIENTS = 94


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

        if not os.path.isfile(trimmed_path):
            break

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


def import_batches(start=1, end=94):
    f = open("pwi_filenames.txt", "r")
    for line in f:
        trimmed_path = line.rstrip()
        splits = trimmed_path.rsplit("/", 3)
        id_part = splits[1]
        case_id = int(id_part[5:])
        if case_id not in range(start, end+1):
            continue

        if not os.path.isdir(BATCH_DIR):
            os.makedirs(BATCH_DIR)

        data = nib.load(trimmed_path).get_fdata()
        shape = np.shape(data)
        num_of_slices = shape[2]
        for slice_id in range(num_of_slices):
            batch_filename = os.path.join(BATCH_DIR, str(case_id) + "-" + str(slice_id))
            batch = []
            if not os.path.isfile(batch_filename):
                for x in range(IMG_SIZE):
                    print("interpolate_sample case %d - slice %d at x =  %d time_period = %d" % (case_id, slice_id, x, shape[3]))
                    for y in range(IMG_SIZE):
                        intensity_arr = data[x, y, slice_id,]
                        intensity_arr[intensity_arr < 0] = 0
                        f = intpf.get_interpolate_func(intensity_arr, shape[3])
                            new_x = np.linspace(0.0, float(shape[3]), num=SAMPLE_SIZE)
                        batch.append(f(new_x))
                with open(batch_filename, "w+") as f:
                    np.savetxt(f, np.array(batch).reshape(1, -1), fmt="%.6f")


def import_keys():
    if not os.path.isdir(KEY_DIR):
        os.makedirs(KEY_DIR)

    batches = os.listdir(BATCH_DIR)
    num_of_batches = len(batches)
    if num_of_batches > 0:
        for batch_name in batches:
            key_file_name = os.path.join(KEY_DIR, batch_name)
            if not os.path.isfile(key_file_name) and batch_name != ".DS_Store":
                tripped = batch_name.split("-")
                case_id = int(tripped[0])
                slice_id = int(tripped[1])
                print("importing key at case - slice: %d - %d" % (case_id, slice_id))

                # read OT file
                OT_file_name = os.path.join(DATASET_DIR, "case_" + str(case_id), "OT", str(slice_id))
                if os.path.isfile(OT_file_name):
                    data = np.loadtxt(OT_file_name)
                    np.savetxt(key_file_name, data,fmt="%d")




