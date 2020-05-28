import nii_reader as nii
import generate_interpolation_func
import get_training_dataset as trainer
import get_graph

def main():
    # 1.import non pwi
    nii.import_non_pwi()

    # 2.import pwi by case
    case_arr = [1,2,3,4,5,6,7,8,9,10]
    for item in case_arr:
        nii.import_pwi_by_case(item)

    # 3.get training set by case
    training_set = []
    for case_id in case_arr:
        # default slice_id = 0
        slice_id = 0
        training_set.append(trainer.get_training_set(case_id, slice_id))
    print(training_set)


if __name__ == "__main__":
    main()