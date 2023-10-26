import os

import pandas as pd
from pathlib import Path

import Configuration as cf


datasets = ["GodClass", "DataClass", "LongMethod", "FeatureEnvy"]
model_name_header = ["NB", "CART", "SVM", "KNN", "LogR", "RF", "XGB", "DF",
          "Bagging", "AdaBoost", "O-NB", "O-CART", "O-SVM", "O-KNN",
          "O-LogR", "O-RF", "O-XGB", "O-DF", "O-Bagging", "O-AdaBoost",
          "LR", "PR", "RFR", "DTR", "BRR", "NNR", "SVR", "KNR", "GBR", "GPR", "SDGR", "RankingSVM", "RankBoost",
          "RankNet", "LambdaRank", "ListNet", "AdaRank", "CoordinateAscent", "EALTR"]


def traverse_root_path(root_path):
    for root, dirs, files in os.walk(root_path):
        if root == root_path:
            this_root = root
            for file in files:
                file_path = os.path.join(this_root, file)
                print(file_path)
            for dir in dirs:
                dir_path = os.path.join(this_root, dir)
                traverse_root_path(dir_path)


def get_dataset_name(dataset_name):
    name = ""
    if dataset_name == "dataclass":
        name = "DataClass"
    elif dataset_name == "featureenvy":
        name = "FeatureEnvy"
    elif dataset_name == "godclass":
        name = "GodClass"
    elif dataset_name == "longmethod":
        name = "LongMethod"
    return name


def gen_heat_result(type="mean"):
    root_path = cf.transform_path
    save_path = cf.median_data_path
    for root, dirs, files in os.walk(root_path):
        if root == root_path:
            this_root = root
            for dir in dirs:
                dir_path = os.path.join(this_root, dir)
                df = pd.DataFrame(index=datasets, columns=model_name_header)
                for root, dirs, files in os.walk(dir_path):
                    if root == dir_path:
                        for file in files:
                            file_path = os.path.join(root, file)
                            dateset_type = get_dataset_name(file.split("_")[0])
                            data = pd.read_csv(file_path)
                            columns = data.columns
                            for column in columns:
                                if type == "mean":
                                    save_path=cf.mean_data_path
                                    result = data[column].mean(skipna=True)
                                elif type == "median":
                                    save_path = cf.median_data_path
                                    result = data[column].median(skipna=True)
                                df.loc[dateset_type, column] = result
                    Path(save_path).mkdir(parents=True, exist_ok=True)
                    file_path = os.path.join(save_path, "%s.xls" % dir)
                    print(file_path)
                    df.to_excel(file_path)

if __name__ == "__main__":
    # traverse_root_path(root_path)
    pass




