import nii_reader as nii
import generate_interpolation_func as gif
import get_training_dataset as trainer
import get_graph
import os
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score

def main():
    # 1.import non pwi
    # nii.import_non_pwi()

    case_arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # 2.import pwi by case
    # for item in case_arr:
    #     nii.import_pwi_by_case(item)

    # 3.get training set and save it
    # batch_name = "1-10"
    # if not os.path.isdir("batch"):
    #     os.makedirs("batch")
    # if not os.path.isdir("key"):
    #     os.makedirs("key")
    #
    # file_name = os.path.join("batch",batch_name)
    # key_path = os.path.join("key",batch_name)
    #
    # for case_id in case_arr:
    #     slice_id = 0
    #     batch,key = trainer.get_training_set(case_id, slice_id)
    #     with open(file_name,"w+") as f:
    #         np.savetxt(f,batch,fmt="%.6f")
    #     with open(key_path,"w+") as g:
    #         np.savetxt(g,key,fmt="%d")
    #     f.close()
    #     g.close()

    # 4. testing dataset
    nii.import_pwi_by_case(12)
    batch, key = trainer.get_training_set(12, 1)
    with open("batch/12", "w+") as f:
        np.savetxt(f, batch, fmt="%.6f")
    with open("key/12", "w+") as g:
        np.savetxt(g, key, fmt="%d")
    f.close()
    g.close()

    # 5.feed to decision tree machine learning
    X = np.loadtxt(os.path.join("batch","1-10"))
    Y = np.loadtxt(os.path.join("key","1-10"))
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X,Y)

    X_test = np.loadtxt(os.path.join("batch","12"))
    Y_test = np.loadtxt(os.path.join("key", "12"))
    Y_predicted = clf.predict(X_test)

    print(Y_predicted)
    score = accuracy_score(Y_test, Y_predicted)
    print("Accuracy score = ",score)
if __name__ == "__main__":
    main()