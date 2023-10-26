"""
Perform Wilcoxon significance test on classifiers and their corresponding enhanced classifiers.

For each classifier and its corresponding enhanced classifier:
(1) Calculate the p-value.
(2) Adjust the p-value.
(3) Calculate the cliff value.

Input format:
Compare rel_severity@20% for significance level, using a list.

Output format:
Output the adjusted p-value.
"""
import os
from pathlib import Path
from scipy import stats
import Configuration as cf
import pandas as pd

classifiers_name = ["NB", "CART", "SVM", "KNN", "LogR", "RF", "XGB", "DF", "Bagging", "AdaBoost"]
datasets_name = ["GodClass", "DataClass", "LongMethod", "FeatureEnvy"]
column_name = ["p-value", "cliff"]
metrics = ["severity@20%", "rel_severity@20%", "severity@10%", "rel_severity@10%",
           "severity@5%", "rel_severity@5%", "acc_ord_3", "acc_ord_2", "acc_ord_1",
           "acc_ord_0", "err_3_to_0", "err_3_to_1", "err_3_to_2", "err_2_to_0",
           "err_2_to_1", "err_2_to_3", "err_1_to_0", "err_1_to_2",
           "err_1_to_3", "err_0_to_1", "err_0_to_2", "err_0_to_3", "accuracy@100%",
           "accuracy@40%", "accuracy@20%", "CLC", "spearmanr", "kendalltau", "MAE", "MSE", "time"]


def wilcoxon(l1, l2):
    if l1 == l2:
        return 1
    else:
        w, p_value = stats.wilcoxon(l1, l2, correction=False)
        return p_value



def cliff(l1, l2):
    total = 0
    for i in l1:
        temp = 0
        for j in l2:
            if i < j:
                temp -= 1
            elif i > j:
                temp += 1
        total += temp
    cliff_value = total / (len(l1)*len(l2))
    return cliff_value


def pvalue_and_cliff_process(classifiers_path, classifiers_name,
                             wilcoxon_path ="empirical_results/wilcoxon_results",
                             metrics=None):
    for metric in metrics:
        for root, dirs, files in os.walk(classifiers_path):
            if root == classifiers_path:
                for dir in dirs:
                    if dir in classifiers_name:
                        classifier_name = dir
                        origin_classifier_path = os.path.join(classifiers_path, classifier_name)
                        origin_o_classifier_path = os.path.join(classifiers_path, "O-" + classifier_name)
                        target_classifier_path = os.path.join(wilcoxon_path, classifier_name)
                        Path(target_classifier_path).mkdir(parents=True, exist_ok=True)
                        df = pd.DataFrame(index=datasets_name, columns=column_name)
                        data = None
                        o_data = None

                        for root1, dirs1, files1 in os.walk(origin_classifier_path):
                            for dir1 in dirs1:
                                data_type = dir1
                                data_type_path = os.path.join(origin_classifier_path, data_type)
                                o_data_type_path = os.path.join(origin_o_classifier_path, data_type)
                                # data
                                for root2, dirs2, files2 in os.walk(data_type_path):
                                    for file2 in files2:
                                        file_path = os.path.join(data_type_path, file2)
                                        data = pd.read_csv(file_path)
                                # o-data
                                for root3, dirs3, files3 in os.walk(o_data_type_path):
                                    for file3 in files3:
                                        file_path = os.path.join(o_data_type_path, file3)
                                        o_data = pd.read_csv(file_path)

                                list = data[metric].tolist()[:-1]
                                o_list = o_data[metric].tolist()[:-1]
                                p_value = wilcoxon(list, o_list)
                                cliff_ = cliff(list, o_list)
                                df.loc[data_type, "p-value"] = p_value
                                df.loc[data_type, "cliff"] = cliff_

                        target_file = os.path.join(target_classifier_path, classifier_name + "_O-" + classifier_name + "_" + metric + ".csv")
                        print(target_file)
                        df.to_csv(target_file)



if __name__ == "__main__":
    pvalue_and_cliff_process(cf.origin_results_path, classifiers_name, metrics=metrics)
