import numpy as np
import os
from matplotlib import pyplot as plt

REPORT_DIR = "report"
REPORT_FILE_NAME = "report/report.txt"
IMG_SIZE = 256


def generate_report(model, f1_score, acc_score):
    if not os.path.isdir(REPORT_DIR):
        os.makedirs(REPORT_DIR)
    f = open(REPORT_FILE_NAME, "a+")
    f.write("Model: %s - Accuracy Score: %f - F1 Score: %f\n"
            % (model, acc_score, f1_score))


def display_interpolate_func(arr, size, func):
    xnew = np.arange(0, size)
    ynew = func(xnew)
    plt.plot(xnew, arr, "", xnew, ynew)
    plt.show()
