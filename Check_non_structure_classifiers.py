from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import time
from datetime import datetime
import matplotlib.pyplot as plt
import csv
from NonStructureFeatures import NonStructureFeatures

from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import scipy


# Display progress logs on stdout
LOG_FILENAME = datetime.now().strftime('C:\\gitprojects\\ML_PROJECT\\non_structure\\LogFileNonStructure_%d_%m_%Y_%H_%M.log')
logging.basicConfig(filename=LOG_FILENAME,
                    level=logging.INFO,)


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm", default=True,
              help="Print the confusion matrix.")
op.add_option("--k_fold",
              action='store', type=int, default=100,
              help='k_fold when using cross validation')
op.add_option("--use_CV",
              action="store_true", dest="print_cm", default=False,
              help="Run cross validation, if False: train on some chromes, and predict others")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()

###############################################################################
class Classifier:
    def __init__(self, features_obj):
        self.X_train =\
            features_obj.all_train_samples_features.ix[:, features_obj.all_train_samples_features.columns != 'IsGen']
        self.Y_train = features_obj.all_train_samples_features['IsGen']
        self.X_test =\
            features_obj.all_test_samples_features.ix[:, features_obj.all_test_samples_features.columns != 'IsGen']
        self.Y_test = features_obj.all_test_samples_features['IsGen']
        print('data loaded')

###############################################################################
# benchmark classifiers
    def benchmark(self, clf, clf_name='default'):
        print('_' * 80)
        print('{}: Traininig: {}'.format((time.asctime(time.localtime(time.time()))), clf))
        logging.info('_' * 80)
        logging.info('{}: Traininig: {}'.format((time.asctime(time.localtime(time.time()))), clf))
        t0 = time.time()
        if clf_name == 'GaussianNB':
            self.X_train = self.X_train.toarray()
        if opts.use_CV:  # run cross validation
            predicted = cross_val_predict(clf, self.X_train, self.Y_train, cv=opts.k_fold)
            score = metrics.accuracy_score(self.Y_train, predicted)
        else:  # fir on train and predict test data
            model = clf.fit(self.X_train, self.Y_train)
            predicted = model.predict(self.X_test)
            score = metrics.accuracy_score(self.Y_test, predicted)

        train_time = time.time() - t0
        print("Train / cross validation time: {}".format(train_time))
        logging.info("Train / cross validation time: {}".format(train_time))

        if opts.print_cm:
            print("confusion matrix:")
            print(metrics.confusion_matrix(self.Y_test, predicted, labels=[-1, 1]))
            logging.info("confusion matrix:")
            logging.info(metrics.confusion_matrix(self.Y_test, predicted, labels=[-1, 1]))

            clf_descr = str(clf).split('(')[0]
        print("Accuracy: {} (+/- {})".format(score.mean(), score.std() * 2))
        logging.info("Accuracy: {} (+/- {})".format(score.mean(), score.std() * 2))

        auc = metrics.roc_auc_score(self.Y_test, predicted, average='samples')
        print('AUC: {}'.format(auc))
        logging.info('AUC: {}'.format(auc))

        return [clf_descr, score, auc, train_time]

    def ModelsIteration(self):
        results = []
        for clf, name in (
                (RidgeClassifier(tol=1e-2, solver="sag"), "Ridge Classifier"),
                (Perceptron(n_iter=50), "Perceptron"),
                (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10), "kNN"),
                # (RandomForestClassifier(n_estimators=100), "Random forest"),
                (SVC(C=1e-8, gamma=1.0/self.X_train.shape[1], kernel='rbf'), "SVM with rbf kernel")):

            print('=' * 80)
            print(name)
            results.append(self.benchmark(clf, name))

        for penalty in ["l2", "l1"]:
            print('=' * 80)
            print("%s penalty" % penalty.upper())
            # Train Liblinear model
            # results.append(self.benchmark(LinearSVC(loss='l2', penalty=penalty,
            #                                         dual=False, tol=1e-3), 'LinearSVC'))
            results.append(self.benchmark(LinearSVC(), 'LinearSVC'))

            # Train SGD model
            results.append(self.benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty), 'SGDClassifier'))

        # Train SGD with Elastic Net penalty
        print('=' * 80)
        print("Elastic-Net penalty")
        results.append(self.benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")))

        # Train NearestCentroid without threshold
        print('=' * 80)
        print("NearestCentroid (aka Rocchio classifier)")
        results.append(self.benchmark(NearestCentroid()))

        # Train sparse Naive Bayes classifiers
        print('=' * 80)
        print("Naive Bayes")
        results.append(self.benchmark(MultinomialNB(alpha=.01), 'MultinomialNB'))
        results.append(self.benchmark(BernoulliNB(alpha=.01), 'BernoulliNB'))
        # results.append(self.benchmark(GaussianNB(), 'GaussianNB'))

        print('=' * 80)
        print("LinearSVC with L1-based feature selection")
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.
        results.append(self.benchmark(Pipeline([
          ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
          ('classification', LinearSVC())
        ])))

        # make some plots

        indices = np.arange(len(results))

        results = [[x[i] for x in results] for i in range(4)]
        file_name = datetime.now().strftime('C:\\gitprojects\\ML_PROJECT\\non_structure\\results_%d_%m_%Y_%H_%M.csv')
        with open(file_name, "wb") as ResultsFile:
            writer = csv.writer(ResultsFile)
            writer.writerows(results)
        ResultsFile.close()

        clf_names, score, auc, training_time = results
        training_time = np.array(training_time) / np.max(training_time)

        plt.figure(figsize=(12, 8))
        plt.title("Score")
        plt.barh(indices, score, .2, label="score", color='navy')
        # plt.barh(indices + .3, training_time, .2, label="training time",
        #          color='c')
        plt.barh(indices + .3, auc, .2, label="ACU", color='darkorange')
        plt.yticks(())
        plt.legend(loc='best')
        plt.subplots_adjust(left=.25)
        plt.subplots_adjust(top=.95)
        plt.subplots_adjust(bottom=.05)

        for i, c in zip(indices, clf_names):
            plt.text(-.3, i, c)

        plt.show()
        plot_name = datetime.now().strftime('C:\\gitprojects\\ML_PROJECT\\non_structure\\plot_%d_%m_%Y_%H_%M.png')
        plt.savefig(plot_name, bbox_inches='tight')
        return


if __name__ == '__main__':
    chrome_train_list = ['8', '5', '11', '14', '2', '13', '10', '16', '12', '7', '15', '4']
    chrome_test_list = ['17', '1', '6', '3', '9']
    logging.info('{}: Train list is (long chromes): {}, test list is (shore chromes): {}'
                 .format(time.asctime(time.localtime(time.time())), chrome_train_list, chrome_test_list))
    NonStructureFeatures_obj = NonStructureFeatures(chrome_train_list, chrome_test_list)
    classifier = Classifier(NonStructureFeatures_obj)
    classifier.ModelsIteration()
