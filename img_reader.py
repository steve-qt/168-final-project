import os
import numpy as np
import nibabel as nib
from matplotlib import pyplot as plt


def import_non_pwi_img():
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
        path = os.path.join("IMG", case, file_type)
        if not os.path.isdir(path):
            os.makedirs(path)
            for i in range(num_of_slides):
                slide = img[:, :, i]
                plt.imshow(slide.T, cmap='gray', origin="lower")
                plt.savefig(fname=path + "/" + str(i), format="png")
                plt.show()
    f.close()


def import_pwi_img():
    f = open("pwi_filenames.txt", "r")
    for line in f:
        trimmed_path = line.rstrip()
        splits = trimmed_path.rsplit("/", 3)
        case = splits[1]
        file_name_only = splits[3]

        data = nib.load(trimmed_path)
        img = data.get_fdata()
        shape = np.shape(img)

        path = os.path.join("IMG", case, "pwi")
        if not os.path.isdir(path):
            os.makedirs(path)
            for i in range(shape[2]):
                for j in range(shape[3]):
                    slide = img[:, :, i, j]
                    plt.imshow(slide.T, cmap='gray', origin="lower")
                    print(path)
                    plt.savefig(fname=path + "/" + str(i) + "-" + str(j), format="png")
                    #plt.show()

def main():
    import_non_pwi_img()
    import_pwi_img()


if __name__ == "__main__":
    main()
