#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os

from sklearn.linear_model import LogisticRegression

import classifier_learn as cl
import classifier_tag as ct
import numpy
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.cross_validation import StratifiedKFold
import numpy as np
import pandas as pd
import util

import tensorflow as tf

tf.python.control_flow_ops = tf
#####################################################
# GLOBAL VARIABLES
DATA_ORG = "/home/zqz/Work/scholarlydata/data/train/training_org(expanded)_features_o.csv"
TASK_NAME = "scholarlydata_org"
DATA_COLS_START = 3  # inclusive
DATA_COLS_END = 20  # exclusive 16
DATA_COLS_FT_END = 16  # exclusive 12
DATA_COLS_TRUTH = 16  # inclusive 12

# when combined with presence feature
# DATA_COLS_START=3 #inclusive
# DATA_COLS_END=24 #exclusive 16
# DATA_COLS_FT_END=20 #exclusive 12
# DATA_COLS_TRUTH=20 #inclusive 12

# DATA_ORG = "/home/zqz/Work/scholarlydata/data/train/training_per_features.csv"
# TASK_NAME = "scholarlydata_per"
# DATA_COLS_START = 3  # inclusive
# DATA_COLS_END = 36  # exclsive
# DATA_COLS_FT_END = 32  # exclusive
# DATA_COLS_TRUTH = 32  # inclusive

# when combined with presence feature
# DATA_COLS_START = 3  # inclusive
# DATA_COLS_END = 44  # exclsive
# DATA_COLS_FT_END = 40  # exclusive
# DATA_COLS_TRUTH = 40  # inclusive

# DATA_COLS_END = 46  # exclsive
# DATA_COLS_FT_END = 42  # exclusive
# DATA_COLS_TRUTH = 42  # inclusive

# Model selection
WITH_SGD = True
WITH_SLR = True
WITH_RANDOM_FOREST = True
WITH_LIBLINEAR_SVM = True
WITH_RBF_SVM = True
WITH_ANN = False

# Random Forest model(or any tree-based model) do not ncessarily need feature scaling
SCALING = True
# feature scaling with bound [0,1] is ncessarily for MNB model
SCALING_STRATEGY_MIN_MAX = 0
# MEAN and Standard Deviation scaling is the standard feature scaling method
SCALING_STRATEGY_MEAN_STD = 1
SCALING_STRATEGY = SCALING_STRATEGY_MEAN_STD

# DIRECTLY LOAD PRE-TRAINED MODEL FOR PREDICTION
# ENABLE THIS VARIABLE TO TEST NEW TEST SET WITHOUT TRAINING
LOAD_MODEL_FROM_FILE = False

# set automatic feature ranking and selection
AUTO_FEATURE_SELECTION = False
FEATURE_SELECTION_WITH_MAX_ENT_CLASSIFIER = False
FEATURE_SELECTION_WITH_EXTRA_TREES_CLASSIFIER = True
FEATURE_SELECTION_MANUAL_SETTING = False
# set manually selected feature index list here
# check random forest setting when changing this variable
MANUAL_SELECTED_FEATURES = []

# The number of CPUs to use to do the computation. -1 means 'all CPUs'
NUM_CPU = -1

N_FOLD_VALIDATION = 10


#####################################################


class ObjectPairClassifer(object):
    """
    supervised org/per pair classifier

    """

    def __init__(self):
        self.training_data = numpy.empty
        self.training_label = numpy.empty
        self.test_data = numpy.empty

    def load_training_data(self, training_file):
        df = pd.read_csv(training_file, header=0, delimiter=",", quoting=0,
                         usecols=range(DATA_COLS_START, DATA_COLS_END)).as_matrix()

        util.timestamped_print("load training data [%s] from [%s]" % (len(df), training_file))

        X, y = df[:, :DATA_COLS_FT_END], \
               df[:,
               DATA_COLS_TRUTH]  # X selects all rows (:), then up to columns 9; y selects all rows, and column 10 only
        self.training_data = X
        self.training_label = y

    def load_testing_data(self, testing_file):
        df = pd.read_csv(testing_file, header=0, delimiter=",", quoting=0,
                         usecols=range(DATA_COLS_START, DATA_COLS_END)).as_matrix()
        self.test_data = df[:, :DATA_COLS_FT_END]

    def training(self):
        print("training data size:", len(self.training_data))
        print("train with CPU cores: [%s]" % NUM_CPU)
        # X_resampled, y_resampled = self.under_sampling(self.training_data, self.training_label)
        # Tuning hyper-parameters for precision

        # split the dataset into two parts, 0.75 for train and 0.25 for testing
        X_train, X_test, y_train, y_test = train_test_split(self.training_data, self.training_label, test_size=0.25,
                                                            random_state=42)

        ######################### SGDClassifier #######################
        if WITH_SGD:
            cl.learn_generative(NUM_CPU, N_FOLD_VALIDATION, TASK_NAME, LOAD_MODEL_FROM_FILE, "sgd", X_train, y_train,
                                X_test, y_test)

        ######################### Stochastic Logistic Regression#######################
        if WITH_SLR:
            cl.learn_generative(NUM_CPU, N_FOLD_VALIDATION, TASK_NAME, LOAD_MODEL_FROM_FILE, "lr", X_train, y_train,
                                X_test, y_test)

        ######################### Random Forest Classifier #######################
        if WITH_RANDOM_FOREST:
            cl.learn_discriminative(NUM_CPU, N_FOLD_VALIDATION, TASK_NAME, LOAD_MODEL_FROM_FILE, "rf", X_train, y_train,
                                    X_test, y_test)

        ###################  liblinear SVM ##############################
        if WITH_LIBLINEAR_SVM:
            cl.learn_discriminative(NUM_CPU, N_FOLD_VALIDATION, TASK_NAME, LOAD_MODEL_FROM_FILE, "svm-l", X_train,
                                    y_train, X_test, y_test)

        ##################### RBF svm #####################
        if WITH_RBF_SVM:
            cl.learn_discriminative(NUM_CPU, N_FOLD_VALIDATION, TASK_NAME, LOAD_MODEL_FROM_FILE, "svm-rbf", X_train,
                                    y_train, X_test, y_test)

        ################# Artificial Neural Network #################
        if WITH_ANN:
            cl.learn_dnn(NUM_CPU, N_FOLD_VALIDATION, TASK_NAME, LOAD_MODEL_FROM_FILE, "ann", DATA_COLS_FT_END, X_train,
                         y_train, X_test, y_test)

        print("complete!")

    def testing(self):
        print("start testing stage :: testing data size:", len(self.test_data))
        print("test with CPU cores: [%s]" % NUM_CPU)

        ######################### SGDClassifier #######################
        if WITH_SGD:
            ct.tag(NUM_CPU, "sgd", TASK_NAME, self.test_data)

        ######################### Stochastic Logistic Regression#######################
        if WITH_SLR:
            ct.tag(NUM_CPU, "lr", TASK_NAME, self.test_data)

        ######################### Random Forest Classifier #######################
        if WITH_RANDOM_FOREST:
            ct.tag(NUM_CPU, "rf", TASK_NAME, self.test_data)

        ###################  liblinear SVM ##############################
        if WITH_LIBLINEAR_SVM:
            ct.tag(NUM_CPU, "svm-l", TASK_NAME, self.test_data)
        ##################### RBF svm #####################
        if WITH_RBF_SVM:
            ct.tag(NUM_CPU, "svm-rbf", TASK_NAME, self.test_data)
        print("complete!")

    def feature_selection_with_max_entropy_classifier(self):
        print("automatic feature selection by maxEnt classifier ...")
        rfe = RFECV(estimator=LogisticRegression(class_weight='auto'),
                    cv=StratifiedKFold(self.training_label, 10), scoring='roc_auc', n_jobs=NUM_CPU)
        rfe.fit(self.training_data, self.training_label)

        self.training_data = rfe.transform(self.training_data)
        print("Optimal number of features : %d" % rfe.n_features_)

    def feature_selection_with_extra_tree_classifier(self):
        print("feature selection with extra tree classifier ...")
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.feature_selection import SelectFromModel

        clf = ExtraTreesClassifier()
        clf = clf.fit(classifier.training_data, classifier.training_label)

        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1].tolist()
        model = SelectFromModel(clf, prefit=True)
        X_n = model.transform(self.training_data).shape[1]
        features_selected = indices[0:X_n]
        features_selected.sort()

        self.training_data = self.training_data[:, features_selected]

        print("Optimal number of features : %s" % str(features_selected))

    def feature_selection_with_manual_setting(self):
        print("feature selection with manual setting ...")
        if MANUAL_SELECTED_FEATURES is None or len(MANUAL_SELECTED_FEATURES) == 0:
            raise ArithmeticError("Manual selected feature is NOT set correctly!")

        self.training_data = self.training_data[:, MANUAL_SELECTED_FEATURES]

        print("Optimal number of features : %s" % str(MANUAL_SELECTED_FEATURES))

    def saveOutput(self, prediction, model_name):
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


if __name__ == '__main__':

    classifier = ObjectPairClassifer()
    classifier.load_training_data(DATA_ORG)
    classifier.load_testing_data(DATA_ORG)
    util.validate_training_set(classifier.training_data)

    if AUTO_FEATURE_SELECTION:
        if FEATURE_SELECTION_WITH_EXTRA_TREES_CLASSIFIER:
            classifier.feature_selection_with_extra_tree_classifier()
        elif FEATURE_SELECTION_WITH_MAX_ENT_CLASSIFIER:
            classifier.feature_selection_with_max_entropy_classifier()
        elif FEATURE_SELECTION_MANUAL_SETTING:
            classifier.feature_selection_with_manual_setting()
        else:
            raise ArithmeticError("Feature selection method IS NOT SET CORRECTLY!")

    # ============== feature scaling =====================
    if SCALING:
        print("feature scaling method: [%s]" % SCALING_STRATEGY)

        # print("example data before scaling:", classifier.training_data[0])

        if SCALING_STRATEGY == SCALING_STRATEGY_MEAN_STD:
            classifier.training_data = util.feature_scaling_mean_std(classifier.training_data)
            classifier.test_data = util.feature_scaling_mean_std(classifier.test_data)
        elif SCALING_STRATEGY == SCALING_STRATEGY_MIN_MAX:
            classifier.training_data = util.feature_scaling_min_max(classifier.test_data)
            classifier.test_data = util.feature_scaling_min_max(classifier.test_data)
        else:
            raise ArithmeticError("SCALING STRATEGY IS NOT SET CORRECTLY!")

            # print("example training data after scaling:", classifier.training_data[0])
    else:
        print("training without feature scaling!")

    # ============= random sampling =================================
    # print("training data size before resampling:", len(classifier.training_data))
    # X_resampled, y_resampled = classifier.under_sampling(classifier.training_data,                                                         classifier.training_label)
    # print("training data size after resampling:", len(X_resampled))
    # enable this line to visualise the data
    # classifier.training_data = X_resampled
    # classifier.training_label = y_resampled

    classifier.training()
    # classifier.testing()
