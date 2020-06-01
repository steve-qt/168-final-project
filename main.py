import nii_reader as nii
import generate_interpolation_func as gif
import get_training_dataset as trainer
import get_graph as gg
import os
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold

SAMPLE_SIZE = 70
IMG_SIZE = 256
BATCH_DIR = "batch"
KEY_DIR = "key"
COMBI_BATCH = "batch/big_batch"
COMBI_KEY = "key/big_key"


def main():
    #nii.import_non_pwi()
    nii.import_batches()
    nii.import_keys()

    X, Y, size = trainer.combine_batches(55,69)
    X = np.array(X).flatten().reshape((size*IMG_SIZE*IMG_SIZE, SAMPLE_SIZE))
    Y = np.array(Y).flatten().reshape(size*IMG_SIZE*IMG_SIZE, 1)

    # split = 5, train= 80%, test =20%
    # rpkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=2652124)
    # for train_index, test_index in rpkf.split(X):
    #     x_train, x_test = X[train_index], X[test_index]
    #     y_train, y_test = Y[train_index], Y[test_index]
    #
    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(x_train,y_train)
    # y_predicted = clf.predict(x_test)
    # score = accuracy_score(y_test, y_predicted)
    # print("Accuracy score = ", score)

    #testing case 93-0
    batch_path = os.path.join(BATCH_DIR,"51-1")
    key_path = os.path.join(KEY_DIR,"51-1")
    batch = np.array(np.loadtxt(batch_path)).reshape((IMG_SIZE*IMG_SIZE, SAMPLE_SIZE))
    key = np.array(np.loadtxt(key_path)).reshape((IMG_SIZE*IMG_SIZE, 1))

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, Y)
    y_predicted = clf.predict(batch)
    score = accuracy_score(key, y_predicted)
    print("Accuracy score = ", score)

    gg.get_OT_firgure_by_1D_array(y_predicted)
    gg.get_OT_firgure_by_1D_array(key)

if __name__ == "__main__":
    main()
