import os
import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

IMG_SIZE = 256
SAMPLE_SIZE = 300


def get_intensity_of_a_voxel(case_id,slice_id,x,y):
    file_name = os.path.join("dataset", "case_" + str(case_id), "PWI","v" + str(slice_id),str(x) + "-" + str(y))
    if os.path.isfile(file_name):
        data = np.loadtxt(file_name)
        data[data < 0] = 0
        return data
    return None


def get_interpolate_func(arr,size):
    f = interpolate.interp1d(np.arange(0, size), arr, kind="cubic", fill_value="extrapolate")
    return f


def get_interpolate_sample(case_id,slice_id,x,y):
    intensity_arr = get_intensity_of_a_voxel(case_id, slice_id, x, y)
    size = len(intensity_arr)
    f = get_interpolate_func(intensity_arr,size)
    xnew = np.linspace(0,size,num=SAMPLE_SIZE)
    return f(xnew)


def display_interpolate_func(arr,size,func):
    xnew = np.arange(0,size)
    ynew = func(xnew)
    plt.plot(xnew,arr,"",xnew,ynew)
    plt.show()



