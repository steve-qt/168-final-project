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
        path = os.path.join("dataset", case, file_type)
        if not os.path.isdir(path):
            os.makedirs(path)
            for i in range(num_of_slides):
                slide = img[:, :, i]
                np.savetxt(os.path.join(path, str(i)), slide)
                # plt.imshow(slide.T, cmap='gray', origin="lower")
                # plt.savefig(fname=path + "/" + str(i), format="png")
                # plt.show()
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
        print(shape)
        for slice_num in range(shape[2]):
            for x in range(shape[0]):
                for y in range(shape[1]):
                    path = os.path.join("dataset", case, "PWI", "v" + str(slice_num))
                    if not os.path.isdir(path):
                        os.makedirs(path)
                    print(path)
                    np.savetxt(os.path.join(path,str(x) + "-" + str(y)), img[x, y, slice_num, :])
    f.close()

def main():
    import_pwi_img()
    # import_non_pwi_img()


if __name__ == "__main__":
    main()
