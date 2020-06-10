import nii_reader as nii
import get_graph as gg
import os
import numpy as np
import get_training_dataset as trainer
import joblib
import report as reporter
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier,RandomForestClassifier,ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier  # Backpropagation

SAMPLE_SIZE = 49
IMG_SIZE = 256
BATCH_DIR = "batch"
KEY_DIR = "key"
MODEL_DIR = "model"
COMBI_BATCH = "batch/big_batch"
COMBI_KEY = "key/big_key"

models = {
        'BaggingClassifier': BaggingClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier( max_depth=7, n_estimators=300, loss='exponential'),
        'DecisionTree': DecisionTreeClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'RandomForestClassifier': RandomForestClassifier(n_estimators=50),
        'ExtraTreesClassifier': ExtraTreesClassifier(n_estimators=50),
        'MLPClassifier_identity': MLPClassifier(activation='identity', max_iter=1000),
        'MLPClassifier_logistic': MLPClassifier(activation='logistic', max_iter=1000),
        'MLPClassifier_tanh': MLPClassifier(activation='tanh', max_iter=1000),
        'MLPClassifier_relu': MLPClassifier(activation='relu', max_iter=1000),
        'LogisticRegression': linear_model.LogisticRegression(random_state=0),
        'NearestCentroid': NearestCentroid(),
        'SGDClassifier_hinge': linear_model.SGDClassifier(),
        'SGDClassifier_log': linear_model.SGDClassifier(loss='log'),
        'SGDClassifier_modified_huber': linear_model.SGDClassifier(loss='modified_huber'),
        'SGDClassifier_perceptron': linear_model.SGDClassifier(loss='perceptron'),
        'SGDClassifier_squared_hinge': linear_model.SGDClassifier(loss='squared_hinge'),
        'SVC_rbf': SVC(),
        'SVC_sigmoid': SVC(kernel='sigmoid'),
        'SVC_linear': SVC(kernel='linear'),
}

def main():
    # # 1 import non pwi
    # nii.import_non_pwi()

    # # 2. import batches
    nii.import_batches(1, 94)

    # # 3. import keys
    # nii.import_keys()

    # 4. combine batches into a single batch file, and keys into a single key file
    start_case = 1
    end_case = 90
    X, Y, size = trainer.combine_batches(start_case, end_case)
    normalized_x = preprocessing.scale(X)
    X = np.reshape(normalized_x, (-1, SAMPLE_SIZE))
    Y = np.array(Y).flatten()

    # 5.1 training and testing with Kfolds (split = 5, train= 80%, test =20%)
    rpkf = RepeatedKFold(n_splits=5, n_repeats=50, random_state=2652124)
    for train_index, test_index in rpkf.split(X):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    for model_name in models:
        print("training with ", model_name)
        model_file_path = os.path.join(MODEL_DIR,
                    model_name + "[" + str(start_case) + "-" + str(end_case) + "].sav")
        if not os.path.isfile(model_file_path):
            clf = models[model_name].fit(x_train, y_train)
            joblib.dump(clf, model_file_path)
        else:
            clf = joblib.load(model_file_path)

        # predict and report
        y_predicted = clf.predict(x_test)
        f1_s = f1_score(y_test, y_predicted)
        acc_s = accuracy_score(y_test, y_predicted)
        print("Model: %s  [%d - %d] f1_score = %f and acc_score = %f"
                    % (model_name,start_case,end_case, f1_s, acc_s))
        reporter.generate_report(model_name + "[" + str(start_case) + "-" + str(end_case) + "]", f1_s, acc_s)



def predict_single_slice(slice_name, model_name):
    batch_path = os.path.join(BATCH_DIR, slice_name)
    key_path = os.path.join(KEY_DIR, slice_name)
    if os.path.isfile(batch_path) and os.path.isfile(key_path):
        batch = np.reshape(np.loadtxt(batch_path).flatten(), (-1, SAMPLE_SIZE))
        normalized = preprocessing.scale(batch)
        key = np.loadtxt(key_path).flatten()
        model = joblib.load(os.path.join(MODEL_DIR, model_name))
        y_predicted = model.predict(normalized)
        f1_s = f1_score(key, y_predicted)
        acc_s = accuracy_score(key, y_predicted)
        print("%f and  %f" % (f1_s, acc_s))
        # reporter.generate_report(model_name + "on slice: " + slice_name, f1_s, acc_s)
        gg.get_OT_firgure_by_1D_array(y_predicted, model_name[:-4] + "-predict:" + slice_name)
        gg.get_OT_firgure_by_1D_array(key, "key:" + slice_name)


def get_highest_f1_score():
    max_f1 = 0.0
    cur_slice = "1-1"
    # 'GradientBoostingClassifier[1-94].sav', 'KNeighborsClassifier[1-94].sav',
    text_model = ['ExtraTreesClassifier[1-94].sav','MLPClassifier_tanh[1-94].sav','SVC_sigmoid[1-94].sav']

    for model in text_model:
        if not os.path.isfile(os.path.join(MODEL_DIR, model)):
            continue
        print(model)
        clf = joblib.load(os.path.join(MODEL_DIR, model))
        for batch_path in os.listdir(BATCH_DIR):
            if batch_path == '.DS_Store':
                continue

            # batch_name = batch_path.rsplit("/")[-1]
            batch = np.reshape(np.loadtxt(os.path.join(BATCH_DIR, batch_path)).flatten(),(-1, SAMPLE_SIZE))
            normalized = preprocessing.scale(batch)
            key = np.loadtxt(os.path.join(KEY_DIR,batch_path)).flatten()
            predicted = clf.predict(normalized)
            f1 = f1_score(predicted,key)
            max_f1 = f1 if f1 > max_f1 else max_f1
            cur_slice = batch_path if f1 > max_f1 else cur_slice
            print("%s : %f" % (batch_path, f1))

    return max_f1, cur_slice


if __name__ == "__main__":
    # for i in range(40,80):
    #     for j in range(2):
    #         predict_single_slice(str(i) + "-" + str(j))

    # main()

    # max_f1, cur_slice = get_highest_f1_score()
    # print("Max f1 = ",max_f1)
    # print("slice = ", cur_slice)

    # gg.get_OT_figure(44, 1)
    # gg.generate_interpolate_func_figure_for_a_slice("44-1")
    # test()

    for model_name in models:
        predict_single_slice("41-1", model_name + '[1-94].sav')


