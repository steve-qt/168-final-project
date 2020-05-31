import nii_reader as nii
import generate_interpolation_func as gif
import get_training_dataset as trainer
import get_graph
import os
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
SAMPLE_SIZE = 300
BATCH_DIR = "batch"
KEY_DIR = "key"
COMBI_BATCH  = "batch/big_batch"
COMBI_KEY = "key/big_key"

def main():
    # 1.import non pwi
    # nii.import_non_pwi()

    # 2.import pwi by case
    # for case_id in range(70):
    #     nii.import_pwi_by_case(case_id)

    # 3.generate training set
    # for case_id in range(1,71):
    #     trainer.genrate_batch_by_case(case_id)

    # # 5.feed to decision tree machine learning
    # combine_batch()
    # X_train = np.loadtxt(COMBI_BATCH)
    # Y_train = np.loadtxt(COMBI_KEY)
    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(X_train,Y_train)

    # # 4. testing dataset
    # print("generating testing dataset")
    # X_test,Y_test  = trainer.get_testing_set(51,1)

    # # 6. predict
    # Y_predicted = clf.predict(X_test)
    # score = accuracy_score(Y_test, Y_predicted)
    # print("Accuracy score = ",score)

    nii.import_batches()


if __name__ == "__main__":
    main()