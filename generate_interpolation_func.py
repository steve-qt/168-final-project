import os
import numpy as np
import nibabel as nib
from scipy import interpolate
from matplotlib import pyplot as plt

IMG_SIZE = 256
SAMPLE_SIZE = 200

def get_intensity_of_a_voxel(case_id,slice_id,x,y):
    file_name = os.path.join("dataset", "case_" + str(case_id), "PWI","v" + str(slice_id),str(x) + "-" + str(y))
    if os.path.isfile(file_name):
        return np.loadtxt(file_name,)
    return None

def get_interpolate_func(arr,size):
    x = np.arange(0, size)
    y = arr
    f = interpolate.interp1d(x, y, kind="cubic")
    return f

def display_interpolate_func(arr, size,func):
    xnew = np.arange(0, size)
    ynew = func(xnew)  # use interpolation function returned by `interp1d`
    plt.plot(xnew, arr, 'o', xnew, ynew, '-')
    plt.show()

if __name__ == "__main__":
    arr1 = get_intensity_of_a_voxel(56,1,1,2)
    arr2 = get_intensity_of_a_voxel(56,1,41,89)
    arr3 = get_intensity_of_a_voxel(56,1, 41, 111)

    if arr3 is not None:
        size = len(arr3)
        f = get_interpolate_func(arr3,size)
        display_interpolate_func(arr3,size,f)

