"""
Extract the performance measures of each model on each dataset separately and compare the same measures horizontally.

First-level directory: Performance measures.
Second-level directory: Dataset types (GodClass, DataClass, LongMethod, FeatureEnvy).
Third-level file: Performance measures.csv
row[0]: Model name.
row[1-100]: Results of 10 iterations of 10-fold cross-validation.
row[101]: Average value.
Conversion:
Iterate through all the original data.xls files and add their respective performance results to different horizontal comparison files.
"""
import csv
import os
import Configuration as cf
from pathlib import Path

header = ["NB", "CART", "SVM", "KNN", "LogR", "RF", "XGB", "DF",
          "Bagging", "AdaBoost", "O-NB", "O-CART", "O-SVM", "O-KNN",
          "O-LogR", "O-RF", "O-XGB", "O-DF", "O-Bagging", "O-AdaBoost",
          "LR", "PR", "RFR", "DTR", "BRR", "NNR", "SVR", "KNR", "GBR", "GPR", "SDGR", "RankingSVM", "RankBoost",
          "RankNet", "LambdaRank", "ListNet", "AdaRank", "CoordinateAscent", "EALTR"]


def transform_result(root_path, target_path):
    list = os.listdir(root_path)
    for i_path in list:
        path = os.path.join(root_path, i_path)
        if os.path.isdir(path):
            transform_result(path, target_path)
        elif os.path.isfile(path):
            print(path)
            transform(path, target_path)


def transform(file_path, target_path):
    root_path = target_path
    data_type = os.path.basename(os.path.dirname(file_path))
    model_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
    csv_file = open(file_path, encoding='utf-8')
    csv_reader = csv.reader(csv_file)
    performance_name = next(csv_reader)[2:]
    values = [row for row in csv_reader]
    csv_file.close()

    for name in performance_name:
        path = os.path.join(root_path, name)
        Path(path).mkdir(parents=True, exist_ok=True)
        file_path_new = os.path.join(path, "%s_%s.csv" % (data_type.lower(), name))
        file_path_new = Path(file_path_new)
        print(file_path_new)

        if file_path_new.exists():
            True
        else:
            csv_file_new = open(file_path_new, "a+", newline='', encoding='utf-8')
            csv_file_writer = csv.writer(csv_file_new)
            csv_file_writer.writerow(header)
            for i in range(101):
                csv_file_writer.writerow(["" for i in range(len(header))])
            csv_file_new.close()

        csv_file = open(file_path_new, encoding='utf-8')
        csv_reader = csv.reader(csv_file)
        col_index = header.index(model_name)
        performance_index = performance_name.index(name)
        performance_value_not_include_mean = [row[performance_index + 2] for row in values[:-1]]

        data = [row for row in csv_reader]
        for i in range(len(performance_value_not_include_mean)):
            data[i + 1][col_index] = performance_value_not_include_mean[i]

        csv_file.close()
        csv_file = open(file_path_new, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(data)
        csv_file.close()


if __name__ == "__main__":
    pass
