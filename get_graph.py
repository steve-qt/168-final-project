import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

IMG_SIZE = 256
BATCH_DIR = "batch"
FIGURE_DIR = "figure"
SAMPLE_SIZE = 49


def get_OT_figure(case_id,slice_id):
    file_name = os.path.join("dataset", "case_" + str(case_id), "OT", str(slice_id))
    if os.path.isfile(file_name):
        data = np.loadtxt(file_name)
        plt.imshow(data, cmap='gray', origin="lower")
        plt.show()


def get_OT_firgure_by_1D_array(arr, file_name):
    if not os.path.isdir(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
    fir_path = os.path.join(FIGURE_DIR,file_name)
    reshaped = np.array(arr).reshape((IMG_SIZE,IMG_SIZE))
    plt.imshow(reshaped, cmap='gray', origin="lower")
    plt.savefig(fir_path)
    # plt.show()


def get_interpolate_graph_for_a_voxel(slice_name, x, y):
    slice_path = os.path.join(BATCH_DIR, slice_name)
    if os.path.isfile(slice_path):
        intensity_arrs = np.loadtxt(slice_path)
        intensity_arrs[intensity_arrs < 0] = 0
        reshaped = intensity_arrs.reshape((IMG_SIZE, IMG_SIZE, SAMPLE_SIZE))
        intensity_arr = reshaped[x, y, :]
        print(str(intensity_arr))
        if intensity_arr[0] == 0:
            return

        x_s = np.arange(0, SAMPLE_SIZE)
        plt.plot(x_s, intensity_arr)
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel('Intensity Value')
        title = slice_name + "[" + str(x) + "-" + str(y) + "]"
        plt.title(title)
        plt.savefig(os.path.join(FIGURE_DIR, title))
        #plt.show()


def generate_interpolate_func_figure_for_a_slice(slice_name):
    if not os.path.isdir(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)

    slice_path = os.path.join(BATCH_DIR, slice_name)
    if os.path.isfile(slice_path):
        intensity_arrs = np.loadtxt(slice_path)
        intensity_arrs[intensity_arrs < 0] = 0
        reshaped = intensity_arrs.reshape((IMG_SIZE, IMG_SIZE, SAMPLE_SIZE))
        for x in range(64,80):
            for y in range(30, IMG_SIZE - 30):
                intensity_arr = reshaped[x, y, :]
                #print(str(intensity_arr))
                if intensity_arr[0] == 0:
                    continue

                x_s = np.arange(0, SAMPLE_SIZE)
                plt.plot(x_s, intensity_arr)
                plt.grid()
                plt.xlabel('Time')
                plt.ylabel('Intensity Value')
                title = slice_name + "[" + str(x) + "-" + str(y) + "]"
                plt.title(title)
                plt.savefig(os.path.join(FIGURE_DIR, title))
                plt.clf()


