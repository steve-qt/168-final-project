import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

def get_OT_figure(case_id,slice_id):
    file_name = os.path.join("dataset", "case_" + str(case_id), "OT", str(slice_id))
    if os.path.isfile(file_name):
        data = np.loadtxt(file_name)
        plt.imshow(data, cmap='gray', origin="lower")
        plt.show()