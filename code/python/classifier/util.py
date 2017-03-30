import pickle

import datetime

from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def load_classifier_model(classifier_pickled=None):
    if classifier_pickled:
        with open(classifier_pickled, 'rb') as model:
            classifier = pickle.load(model)
        return classifier


def saveOutput(prediction, model_name):
    filename = os.path.join(os.path.dirname(__file__), "prediction-%s-%s.csv" % (model_name, TASK_NAME))
    file = open(filename, "w")
    for entry in prediction:
        if (isinstance(entry, float)):
            file.write(str(entry) + "\n")
            # file.write("\n")
        else:
            if (entry[0] > entry[1]):
                file.write("0\n")
            else:
                file.write("1\n")
    file.close()


def index_max(values):
    return max(range(len(values)), key=values.__getitem__)


def save_classifier_model(model, outfile):
    if model:
        with open(outfile, 'wb') as model_file:
            pickle.dump(model, model_file)


def print_eval_report(best_params, cv_score, prediction_dev,
                      time_predict_dev,
                      time_train, y_test, nfold):
    print("%s fold CV score [%s]; best params: [%s]" %
          (nfold, cv_score, best_params))
    print("\nTraining time: %fs; "
          "Prediction time for 'dev': %fs;" %
          (time_train, time_predict_dev))
    print("\n %fs fold cross validation score:" % cv_score)
    print("\n test set result:")
    print("\n" + classification_report(y_test, prediction_dev))


def timestamped_print(msg):
    ts = str(datetime.datetime.now())
    print(ts + " :: " + msg)


def validate_training_set(training_set):
    """
    validate training data set (i.e., X) before scaling, PCA, etc.
    :param training_set: training set, test data
    :return:
    """
    # print("np any isnan(X): ", np.any(np.isnan(training_set)))
    # print("np all isfinite: ", np.all(np.isfinite(training_set)))
    # check any NaN row
    row_i = 0
    for i in training_set:
        row_i += 1
        if np.any(np.isnan(i)):
            print("ERROR: [", row_i, "] is nan: ")
            print(i)


def feature_scaling_mean_std(feature_set):
    scaler = StandardScaler(with_mean=True, with_std=True)
    return scaler.fit_transform(feature_set)


def feature_scaling_min_max(feature_set):
    """
    Input X must be non-negative for multinomial Naive Bayes model
    :param feature_set:
    :return:
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler.fit_transform(feature_set)


def under_sampling(_X, _y):
    """
    under-sampling for unbalanced training set

    :return: X_resampled, y_resampled
    """
    rus = RandomUnderSampler()
    return rus.fit_sample(_X, _y)
