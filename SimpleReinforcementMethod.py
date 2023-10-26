"""
A simple approach to ordinal classification:
Ordinal classification is a problem that combines multi-class classification and basic binary classifiers.

Steps:

Divide the problem into three binary classifiers:
(1) {0}, {1, 2, 3} --> 0, 1 [label > 0]
(2) {0, 1}, {2, 3} --> 0, 1 [label > 1]
(2) {0, 1, 2}, {3} --> 0, 1 [label > 2]

Modify the labels of the training data to either 0 or 1 based on the classifiers.

Split the original training data into three separate sets for each classifier, and train them individually.

Obtain three binary classifiers after training.

When encountering a new sample (test sample), calculate the conditional probabilities based on its features:
Classifier 1: Pr(label > 0 | x1, x2, ...)
Classifier 2: Pr(label > 1 | x1, x2, ...)
Classifier 3: Pr(label > 2 | x1, x2, ...)

Based on the probabilities from the three classifiers, calculate the specific probabilities for each category:
Calculate the probability for y = 0: 1 - Pr(label > 0 | x1, x2, ...)
Calculate the probability for y = 1: Pr(label > 0 | x1, x2, ...) * (1 - Pr(label > 1 | x1, x2, ...))
Calculate the probability for y = 2: Pr(label > 1 | x1, x2, ...) * (1 - Pr(label > 2 | x1, x2, ...))
Calculate the probability for y = 3: Pr(label > 2 | x1, x2, ...)

Finally, select the category with the highest probability as the predicted label for the test sample.
"""
import numpy as np
import Configuration as cf
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
import sklearn.pipeline as pl
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor


class SimpleOrdinalClassifier:

    def __init__(self,
                 basic_classifier=None):
        """
        initialization
        :param basic_classifier: base binary classifier
        """
        self.basic_classifier = basic_classifier
        self.basic_classifier_all = []
        self.k = 0
        self.label_all = []

    def fit(self, X, y):
        """
        train model
        :param X: feature
        :param y: label
        :return: return odel
        """
        label_all = np.unique(y)
        self.label_all = label_all
        k = len(label_all)
        self.k = k
        print("classification number:" + str(k) + " classification problem")
        for i in range(k - 1):
            self.basic_classifier_all.append(clone(self.basic_classifier))
        train_y_all = []
        zero_list = []
        for i in range(k - 1):
            zero_list.append(label_all[i])
            train_y = [0 if j in zero_list else 1 for j in y]
            train_y_all.append(train_y)
        for i in range(k - 1):
            self.basic_classifier_all[i].fit(X, train_y_all[i])
        return self

    def predict(self, X):
        predict_proba = []
        for i in range(self.k - 1):
            predict_proba.append(self.basic_classifier_all[i].predict_proba(X)[:, 1])

        predict_y = []
        for i in range(len(X)):
            proba_x = []
            for j in range(self.k):
                if j == 0:
                    proba_x.append(1 - predict_proba[j][i])
                elif j == self.k - 1:
                    proba_x.append(predict_proba[self.k - 2][i])
                else:
                    proba_x.append(predict_proba[j - 1][i] * (1 - predict_proba[j][i]))
            index = proba_x.index(max(proba_x))
            predict_y.append(self.label_all[index])

        return np.array(predict_y)


if __name__ == "__main__":
   pass

