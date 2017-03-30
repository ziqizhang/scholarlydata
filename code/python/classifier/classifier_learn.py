from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

import util
from sklearn.model_selection import GridSearchCV
import os
from time import time
import numpy as np


def learn_discriminative(cpus, nfold, task, load_model, model, X_train, y_train, X_test, y_test):
    classifier = None
    model_file = None

    if (model == "rf"):
        print("=== Random Forest ...")
        classifier = RandomForestClassifier(n_estimators=20, n_jobs=cpus)
        rfc_tuning_params = {"max_depth": [3, 5, None],
                             "max_features": [1, 3, 5, 7, 10],
                             "min_samples_split": [1, 3, 10],
                             "min_samples_leaf": [1, 3, 10],
                             "bootstrap": [True, False],
                             "criterion": ["gini", "entropy"]}
        classifier = GridSearchCV(classifier, param_grid=rfc_tuning_params, cv=nfold,
                                  n_jobs=cpus)
        if (load_model):
            model_file = os.path.join(os.path.dirname(__file__), "random-forest_classifier-%s.m" % task)
    if (model == "svm-l"):
        tuned_parameters = [{'gamma': np.logspace(-9, 3, 3), 'probability': [True], 'C': np.logspace(-2, 10, 3)},
                            {'C': [1e-1, 1e-3, 1e-5, 0.2, 0.5, 1, 1.2, 1.3, 1.5, 1.6, 1.7, 1.8, 2]}]

        print("== SVM, kernel=linear ...")
        classifier = svm.LinearSVC()
        classifier = GridSearchCV(classifier, tuned_parameters[1], cv=nfold, n_jobs=cpus)
        if (load_model):
            model_file = os.path.join(os.path.dirname(__file__), "liblinear-svm-linear-%s.m" % task)

    if (model == "svm-rbf"):
        print("== SVM, kernel=rbf ...")
        classifier = svm.SVC()
        classifier = GridSearchCV(classifier, param_grid=tuned_parameters[0], cv=nfold, n_jobs=cpus)
        if (load_model):
            model_file = os.path.join(os.path.dirname(__file__), "liblinear-svm-rbf-%s.m" % task)

    best_param = []
    cv_score = 0
    best_estimator = None

    t0 = time()
    if load_model:
        print("model is loaded from [%s]" % str(model_file))
        best_estimator = util.load_classifier_model(model_file)
    else:
        classifier.fit(X_train, y_train)
        print(
            "Model is selected with GridSearch and trained with [%s] fold cross-validation ! " % nfold)

        best_estimator = classifier.best_estimator_
        best_param = classifier.best_params_
        cv_score = classifier.best_score_
        util.save_classifier_model(best_estimator, model_file)

    t1 = time()
    prediction_test = best_estimator.predict(X_test)
    t4 = time()

    time_train = t1 - t0
    time_test = t4 - t1

    print("\tResults:")
    util.print_eval_report(best_param, cv_score, prediction_test,
                           time_test,
                           time_train, y_test)


def learn_generative(cpus, nfold, task, load_model, model, X_train, y_train, X_test, y_test):
    classifier = None
    model_file = None
    if (model == "sgd"):
        print("== SGD ...")
        sgd_params = {"loss": ["log", "modified_huber", "squared_hinge", 'squared_loss'],
                      "penalty": ['l2', 'l1'],
                      "alpha": [0.0001, 0.001, 0.01, 0.03, 0.05, 0.1],
                      "n_iter": [1000],
                      "learning_rate": ["optimal"]}
        classifier = SGDClassifier(loss='log', penalty='l2', n_jobs=cpus)

        classifier = GridSearchCV(classifier, param_grid=sgd_params, cv=nfold,
                                  n_jobs=cpus)
        if (load_model):
            model_file = os.path.join(os.path.dirname(__file__), "sgd-classifier-%s.m" % task)
    if (model == "lr"):
        print("== Stochastic Logistic Regression ...")
        slr_params = {"penalty": ['l2'],
                      "solver": ['liblinear'],
                      "C": list(np.power(10.0, np.arange(-10, 10))),
                      "max_iter": [10000]}
        classifier = LogisticRegression(random_state=111)
        classifier = GridSearchCV(classifier, param_grid=slr_params, cv=nfold,
                                  n_jobs=cpus)
        if (load_model):
            model_file = os.path.join(os.path.dirname(__file__), "stochasticLR-%s.m" % task)

    best_param = []
    cv_score = 0
    best_estimator = None

    t0 = time()
    if load_model:
        print("model is loaded from [%s]" % str(model_file))
        best_estimator = util.load_classifier_model(model_file)
    else:
        classifier.fit(X_train, y_train)
        print(
            "Model is selected with GridSearch and trained with [%s] fold cross-validation ! " % nfold)

        best_estimator = classifier.best_estimator_
        best_param = classifier.best_params_
        cv_score = classifier.best_score_
        util.save_classifier_model(best_estimator, model_file)

    t1 = time()
    classes = classifier.best_estimator_.classes_
    probabilities_dev = best_estimator.predict_proba(X_test)
    prediction_dev = [classes[util.index_max(list(probs))] for probs in probabilities_dev]

    t4 = time()

    time_rfc_train = t1 - t0
    time_rfc_predict_dev = t4 - t1

    print("\tResults:")
    util.print_eval_report(best_param, cv_score, prediction_dev,
                           time_rfc_predict_dev,
                           time_rfc_train, y_test)


def learn_dnn(cpus, nfold, task, load_model, model, input_dim, X_train, y_train, X_test, y_test):
    print("== Perform ANN ...")  # create model
    model = KerasClassifier(build_fn=create_model(input_dim), verbose=0)
    # define the grid search parameters
    batch_size = [10, 20]
    epochs = [50, 100]
    dropout = [0.1, 0.3, 0.5, 0.7]
    param_grid = dict(dropout_rate=dropout, batch_size=batch_size, nb_epoch=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=cpus,
                        cv=nfold)

    t0 = time()
    cv_score_ann = 0
    best_param_ann = []
    ann_model_file = os.path.join(os.path.dirname(__file__), "ann-%s.m" % task)

    if load_model:
        print("model is loaded from [%s]" % str(ann_model_file))
        best_estimator = util.load_classifier_model(ann_model_file)
    else:
        grid.fit(X_train, y_train)
        print(
            "Model is selected with GridSearch and trained with [%s] fold cross-validation ! " % nfold)

        cv_score_ann = grid.best_score_
        best_param_ann = grid.best_params_
        best_estimator = grid.best_estimator_

        # self.save_classifier_model(best_estimator, ann_model_file)

    t1 = time()
    print("testing on development set ....")
    dev_data_prediction_ann = best_estimator.predict(X_test)
    t4 = time()

    time_ann_train = t1 - t0
    time_ann_predict_dev = t4 - t1

    util.print_eval_report(best_param_ann, cv_score_ann, dev_data_prediction_ann,
                           time_ann_predict_dev,
                           time_ann_train, y_test)

def create_model(input_dim,dropout_rate=0.0):
    # create model
    model = Sequential()
    model.add(Dense(80,
                    input_dim=input_dim,
                    init='uniform', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
