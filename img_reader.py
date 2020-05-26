import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt

#hello
# example_filename = "ISLES2018/TRAINING/case_1/SMIR.Brain.XX.O.CT_CBF.345563/SMIR.Brain.XX.O.CT_CBF.345563.nii"
# data = nib.load(example_filename)
# img = data.get_fdata()
# slides = img[:, :, 5]
# plt.figure()
# plt.imshow(slides.T,cmap='gray',origin="lower")
# plt.show()

f = open("filenames.txt", "r")
for line in f:
    trimmed_path = line.rstrip()
    file_name_only = trimmed_path.rsplit("/", 1)[1]
    if file_name_only.endswith(".nii"):
        file_name_only = file_name_only[:-4]

    data = nib.load(trimmed_path)
    img = data.get_fdata()
    shape = np.shape(img)
    print(shape)
    num_of_slides = shape[2]
    for i in range(num_of_slides):
        slides = img[:, :, i]
        plt.imshow(slides, cmap='gray', origin="lower")
        path = os.path.join("IMG", file_name_only)
        if not os.path.isdir(path):
            os.makedirs(path)
        plt.savefig(fname=path + "/" + str(i), format="png")
        plt.show()
f.close()
