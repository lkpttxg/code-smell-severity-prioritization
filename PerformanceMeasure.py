import csv
import os.path

import numpy as np
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import Configuration as cf
import time
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from rankboostYin import *


class PerformanceMeasure:
    def __init__(self, real_list=None, pred_list=None):
        copy_real = np.array(real_list)
        copy_pred = np.array(pred_list)
        copy_real = copy_real.reshape(len(copy_real), 1)
        copy_pred = copy_pred.reshape(len(copy_pred), 1)
        y_all = np.hstack((copy_real, copy_pred))
        y_all = y_all[np.argsort(-y_all[:, 1])]
        copy_real = y_all[:, 0]
        copy_pred = y_all[:, 1]

        self.real = copy_real.astype(int)
        self.pred = np.round(copy_pred)

    def accuracy_percentile_front(self, percent):
        percentile = int(len(self.real) * percent)
        new_real = self.real[:percentile]
        new_predict = self.pred[:percentile]
        accuracy = metrics.accuracy_score(new_real, new_predict)
        return accuracy * 1.00

    def percentile_num1_to_num2(self, num1, num2):
        copy_real = np.array(self.real)
        copy_pred = np.array(self.pred)
        count = np.sum(copy_real == num1)
        # if count == 0:
        #     if num1 == num2:
        #         return 1.0
        #     return 0.0
        error_count = 0
        for i in range(len(copy_real)):
            if copy_real[i] == num1:
                if copy_pred[i] == num2:
                    error_count += 1
        percentile = error_count * 1.0 / count
        return percentile * 1.00

    def severity_percentile_front(self, percent):
        length = int(len(self.real) * percent)
        severity_all = np.sum(self.real)
        severity_front = np.sum(self.real[:length])
        percentile = severity_front * 1.0 / severity_all
        return percentile * 1.00

    def relative_severity_percentile_front(self, percent):
        length = int(len(self.real) * percent)
        new_real_predict = self.real.copy()
        new_real_optimal = self.real[np.argsort(-self.real[:])]

        severity_predict = np.sum(new_real_predict[:length])
        severity_optimal = np.sum(new_real_optimal[:length])

        percentile = severity_predict * 1.0 / severity_optimal

        return percentile * 1.00

    def spearman(self):
        new_real = self.real.copy()
        new_pred = self.pred.copy()
        corr = spearmanr(new_pred, new_real)[0]
        return corr * 1.00

    def MAE(self):
        err = metrics.mean_absolute_error(self.real, self.pred)
        return err

    def MSE(self):
        err = metrics.mean_squared_error(self.real, self.pred)
        return err

    def kendall(self):
        new_real = self.real.copy()
        new_pred = self.pred.copy()
        corr = kendalltau(new_pred, new_real)[0]
        return corr * 1.00

    def FPA(self):
        '''
        There are four modules: m1, m2, m3, m4. The actual number of defects for each module is [1, 4, 2, 1] (self.real).
        The predicted number of defects for m1 is 0, for m2 is 3, for m3 is 5, and for m4 is 1 (self.pred).
        The predicted ranking is m3(2) > m2(4) > m4(1) > m1(1).
        The fpa (False Positive Added) can be calculated as follows:
        fpa = (1/4) * (1/8) * ((42) + (34) + (21) + (11)) = 0.71875.
        '''
        K = len(self.real)
        N = np.sum(self.real)
        sort_axis = np.argsort(self.pred)  #
        testBug = np.array(self.real)
        testBug = testBug[sort_axis]
        FPA = sum(np.sum(testBug[m:]) / N for m in range(K + 1)) / K
        return FPA

    def CLC(self):
        K = len(self.real)
        N = np.sum(self.real)
        sort_axis = np.argsort(self.pred)
        testBug = np.array(self.real)
        testBug = testBug[sort_axis]
        P = sum(np.sum(testBug[m:]) / N for m in range(K + 1)) / K
        CLC = P - (1.0 / (2 * K))
        return CLC


def evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time):
    try:
        if model_name in ["EALTR", "EALTR-CLC", "EALTR-Rel"]:
            ltr_w = model.process()
            predict_y = model.predict(test_x, ltr_w)
        elif model_name in ["RankBoost", "RankNet", "LambdaRank", "ListNet", "AdaRank", "CoordinateAscent"]:
            model_path = train_modelYin(model, train_x)
            result_dat_path = pred_test_dataYin(test_x, model_path)
            predict_y = get_pred_bug(result_dat_path)
        else:
            model.fit(train_x, train_y)
            predict_y = model.predict(test_x)
            predict_y = np.array([i if 0 <= i <= 3 else 0 if i < 0 else 3 for i in np.round(predict_y).astype(int)])

        predictor = PerformanceMeasure(test_y, predict_y)

        accuracy_100 = predictor.accuracy_percentile_front(1)

        accuracy_40 = predictor.accuracy_percentile_front(0.4)

        accuracy_20 = predictor.accuracy_percentile_front(0.2)

        accuracy_ordinal_3 = predictor.percentile_num1_to_num2(3, 3)

        accuracy_ordinal_2 = predictor.percentile_num1_to_num2(2, 2)

        accuracy_ordinal_1 = predictor.percentile_num1_to_num2(1, 1)

        accuracy_ordinal_0 = predictor.percentile_num1_to_num2(0, 0)

        error_3_to_2 = predictor.percentile_num1_to_num2(3, 2)
        error_3_to_1 = predictor.percentile_num1_to_num2(3, 1)
        error_3_to_0 = predictor.percentile_num1_to_num2(3, 0)

        error_2_to_3 = predictor.percentile_num1_to_num2(2, 3)
        error_2_to_1 = predictor.percentile_num1_to_num2(2, 1)
        error_2_to_0 = predictor.percentile_num1_to_num2(2, 0)

        error_1_to_3 = predictor.percentile_num1_to_num2(1, 3)
        error_1_to_2 = predictor.percentile_num1_to_num2(1, 2)
        error_1_to_0 = predictor.percentile_num1_to_num2(1, 0)

        error_0_to_3 = predictor.percentile_num1_to_num2(0, 3)
        error_0_to_2 = predictor.percentile_num1_to_num2(0, 2)
        error_0_to_1 = predictor.percentile_num1_to_num2(0, 1)

        severity_20 = predictor.severity_percentile_front(0.2)
        rel_severity_20 = predictor.relative_severity_percentile_front(0.2)
        severity_10 = predictor.severity_percentile_front(0.1)
        rel_severity_10 = predictor.relative_severity_percentile_front(0.1)

        severity_5 = predictor.severity_percentile_front(0.05)
        rel_severity_5 = predictor.relative_severity_percentile_front(0.05)

        clc = predictor.CLC()

        spearmanr = predictor.spearman()

        kendalltau = predictor.kendall()

        MAE = predictor.MAE()

        MSE = predictor.MSE()

        header = ["Dataset", "Algorithm", "severity@20%", "rel_severity@20%", "severity@10%",
                  "rel_severity@10%", "severity@5%", "rel_severity@5%", "acc_ord_3", "acc_ord_2", "acc_ord_1",
                  "acc_ord_0", "err_3_to_0", "err_3_to_1", "err_3_to_2", "err_2_to_0",
                  "err_2_to_1", "err_2_to_3", "err_1_to_0", "err_1_to_2",
                  "err_1_to_3", "err_0_to_1", "err_0_to_2", "err_0_to_3", "accuracy@100%",
                  "accuracy@40%", "accuracy@20%", "CLC", "spearmanr", "kendalltau", "MAE", "MSE", "time"]
        end_time = time.time()
        run_time = end_time - start_time
        result = [data_type, model_name, str(severity_20), str(rel_severity_20), str(severity_10),
                  str(rel_severity_10), str(severity_5), str(rel_severity_5), str(accuracy_ordinal_3),
                  str(accuracy_ordinal_2), str(accuracy_ordinal_1), str(accuracy_ordinal_0),
                  str(error_3_to_0), str(error_3_to_1), str(error_3_to_2), str(error_2_to_0),
                  str(error_2_to_1), str(error_2_to_3), str(error_1_to_0), str(error_1_to_2),
                  str(error_1_to_3), str(error_0_to_1), str(error_0_to_2), str(error_0_to_3),
                  str(accuracy_100), str(accuracy_40), str(accuracy_20), str(clc),
                  str(spearmanr), str(kendalltau), str(MAE), str(MSE), str(run_time)]

        result_dir_path = os.path.join(cf.prediction_path, "%s/%s" % (model_name, data_type))
        Path(result_dir_path).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(result_dir_path,
                                "%s_%s_metrics_result.csv" % (data_type.lower(), model_name.lower()))
        result_file = Path(filename)
        if result_file.exists():
            True
        else:
            csv_file = open(filename, "w", newline='', encoding='utf-8')
            csv_write = csv.writer(csv_file)
            csv_write.writerow(header)
            csv_file.close()

        print("-----Performance of dataset (%s) under model (%s)-----" % (data_type, model_name))
        csv_file = open(filename, 'a+', newline='', encoding='utf-8')
        csv_write = csv.writer(csv_file)
        print(header)
        print(result)
        csv_write.writerow(result)
        csv_file.close()
        print("-----Performance of dataset (%s) under model (%s)-----\n" % (data_type, model_name))
    except Exception as ex:
        print("error %s" % ex)


def gird_search(model, model_name, parameters, train_x, train_y):
    origin_model = model
    model = GridSearchCV(model, parameters, refit=True, cv=10, verbose=1, n_jobs=-1)
    model.fit(train_x, train_y)
    parm = model.best_params_
    print("%s best parm:" % model_name)
    print(parm)
    for key, value in parm.items():
        origin_model.__setattr__(key, value)
    return parm, origin_model


if __name__ == "__main__":
    real = [1, 3, 0, 2, 0, 1, 0, 3, 0, 0, 3, 2, 1]
    pred = [3.6, 0.6, -1, -0.5, 2.6, 1.2, 1.5, 2.1, 0, 3, 1, 0, 0]
    predictor = PerformanceMeasure(real, pred)
    print(predictor.pred)
    print(predictor.real)
    print(type(predictor.pred))
