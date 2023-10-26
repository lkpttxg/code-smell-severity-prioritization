import math
import os
import pandas as pd
import numpy as np
import Configuration as cf
from pathlib import Path


metrics_name = ["severity@20%", "CLC", "spearmanr", "MAE", "acc_ord_3", "acc_ord_2",
                "acc_ord_1", "acc_ord_0", "accuracy@100%", ]
datasets_name = ["GodClass", "DataClass", "LongMethod", "FeatureEnvy"]



def get_datasets_metrics_csv(origin_path=cf.origin_results_path, target_path=cf.datasets_metrics_path):

    Path(target_path).mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(origin_path):
        if root == origin_path:
            for dir in dirs:
                model_name = dir
                model_path = os.path.join(origin_path, dir)
                df = pd.DataFrame(columns=metrics_name, index=datasets_name)
                for root1, dirs1, files1 in os.walk(model_path):
                    for dir1 in dirs1:
                        datasets_path = os.path.join(model_path, dir1)
                        for root2, dirs2, files2 in os.walk(datasets_path):
                            for file2 in files2:
                                file_path = os.path.join(datasets_path, file2)
                                file_data = pd.read_csv(file_path)
                                for metric in metrics_name:
                                    # metric_median = file_data[metric].iloc[0:100].median()
                                    metric_mean = file_data[metric].iloc[0:100].mean()
                                    # df.loc[dir1, metric] = metric_median
                                    df.loc[dir1, metric] = metric_mean

                target_file_path = os.path.join(target_path, "datasets_metrics_" + model_name + ".csv")
                print(target_file_path)
                print(df)
                df.to_csv(target_file_path, index=False)


def calcMean(x, y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean


def calcPearson(x,y):
    x_mean, y_mean = calcMean(x,y)	# Calculate the average value of the x and y vectors.
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p

if __name__ == '__main__':

    # get_datasets_metrics_csv()

    path = cf.datasets_metrics_path
    target_path = cf.metircs_pearson_data_path
    Path(target_path).mkdir(parents=True, exist_ok=True)


    for root, dirs, files in os.walk(path):
        for file in files:
            data = pd.read_csv(os.path.join(path, file))
            data_list = np.array(data)

            # print(data_list)
            # print(len(data_list))
            result_list = []
            for i in range(len(data_list[0])):
                x = list(data_list.T[i])
                tmp = []
                for j in range(len(data_list[0])):
                    y = list(data_list.T[j])
                    tmp.append(calcPearson(x, y))
                result_list.append(tmp)

            # print(result_list)
            index = ["Severity@20%", "CLC", "Spearman", "MAE", "Acc_3", "Acc_2",
                "Acc_1", "Acc_0", "Accuracy", ]

            df = pd.DataFrame(data=result_list, columns=index)
            file_path = os.path.join(target_path, file.split('_')[2])
            print(file_path)
            df.to_csv(file_path, index=None)






