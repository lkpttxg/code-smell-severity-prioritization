# coding=utf-8
from itertools import combinations, permutations
import numpy as np
from sklearn import svm, linear_model
import pandas as pd

class RankSVM(svm.LinearSVC):


    def transform_pairwise(self, X, y):

        X_new = []

        y_new = []

        y = np.asarray(y)

        if y.ndim == 1:

            y = np.c_[y, np.ones(y.shape[0])]

        comb = combinations(range(X.shape[0]), 2)

        for k, (i, j) in enumerate(comb):

            #    continue

            X_new.append(X[i] - X[j])

            y_new.append(np.sign(y[i, 0] - y[j, 0]))  # -1/1

        X_pairs = np.asarray(X_new)

        y_pairs = np.asarray(y_new)

        return X_pairs, y_pairs



    def tst_transform_pairwise(self, X):

        X_new = []

        perm = permutations(range(X.shape[0]), 2)

        # print(X.shape[0])

        for k, (i, j) in enumerate(perm):

            X_new.append(X[i] - X[j])

        return np.asarray(X_new)





    def fit(self, X, y):

        X_pairs, y_pairs = self.transform_pairwise(X, y)

        super(RankSVM, self).fit(X_pairs, y_pairs)

        return self







    def predict(self, X):

        X_tests = self.tst_transform_pairwise(X)

        y_pred = super(RankSVM, self).predict(X_tests)



        length = X.shape[0]

        count = self.rank_list(y_pred, length)

        pred_bug_rank = self.trans(count=count)



        return pred_bug_rank





    def rank_list(self, y, length):

        count_list = []

        for i in range(length):

            count = 0

            for j in range(length - 1):

                n = i * (length - 1) + j

                if(y[n] == -1):

                    count = count + 1

            count_list.append(count)



        k = 0

        max_list = []

        # Select the minimum, process the same number of -1, and mark the processed positions as -1.

        while(k < len(count_list)):

            k = k + 1

            large = max(count_list)

            max_index = [m for m in range(length) if count_list[m] == large]

            if len(max_index) > 1:

                for i in range(len(max_index) - 1):

                    max_i = max_index[i]

                    n = max_i * (length - 1) + max_index[i + 1] - 1

                    if(y[n] == 1):

                        max_i = max_index[i + 1]

                max_list.append(max_i)

                count_list[max_i] = -1

            else:

                max_list.append(max_index[0])

                count_list[max_index[0]] = -1

        # The returned indices are in ascending order.

        return max_list


    def trans(self, count):

        t = []

        for i, j in enumerate(count):

            t.append((i, j))



        t = sorted(t, key=lambda k: k[1])

        res = []



        for i in t:

            res.append(i[0])



        return res





if __name__ == '__main__':

    X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6],[7, 7], [8, 8], [9, 9], [10, 10], [11, 11], [12, 12]])

    y = np.array([5, 4, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])


    predy=RankSVM().fit(X,y).predict(X)

    xpairs,ypairs=RankSVM().transform_pairwise(X,y)


    print(xpairs)
    print(ypairs)






