import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt
IMG_SIZE = 256

def get_OT_figure(case_id,slice_id):
    file_name = os.path.join("dataset", "case_" + str(case_id), "OT", str(slice_id))
    if os.path.isfile(file_name):
        data = np.loadtxt(file_name)
        plt.imshow(data, cmap='gray', origin="lower")
        plt.show()


def get_OT_firgure_by_1D_array(arr):
    reshaped = np.array(arr).reshape((IMG_SIZE,IMG_SIZE))
    plt.imshow(reshaped, cmap='gray', origin="lower")
    plt.show()