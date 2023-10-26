"""
The purpose of this file is to calculate the mean of the test results.

Use the mean to fill in NaN values.
If there are fewer than 100 results, fill in the remaining with the mean to make it 100.
Add the mean to the 101st row.
"""
import csv
import pandas as pd
import Configuration as cf
import os
from pathlib import Path


def cal_result(filename, type):
    data = pd.read_csv(filename)
    row_length = data.shape[0]
    length = data.shape[1]
    columns = data.columns

    results = [0.0 for i in range(length)]
    results[0] = ""
    results[1] = type
    for col in range(2, length):
        if type == "mean":
            mean = data.iloc[:, col].mean(skipna=True)
            results[col] = results[col] if pd.isna(mean) else mean
        elif type == "median":
            median = data.iloc[:, col].median(skipna=True)
            results[col] = results[col] if pd.isna(median) else median
        else:
            print("mode error")
            exit(0)
    print(results)

    b_results = results.copy()
    b_results[0] = data.iloc[0, 0]
    b_results[1] = data.iloc[0, 1]
    print(b_results)
    for i in range(row_length, 100):
        data.loc[len(data.index)] = b_results

    for i in range(2, len(columns)):
        data[columns[i]].fillna(results[i], inplace=True)
    # if row_length < 101:
    #     data.loc[len(data.index)] = results

    print(data)
    data.to_csv(filename, index=False)


def cal_all(root_dir, type="mean"):
    list = os.listdir(root_dir)
    for i in range(len(list)):
        path = os.path.join(root_dir, list[i])
        if os.path.isdir(path):
            cal_all(path, type)
        if os.path.isfile(path):
            print(path)
            if type in ["mean", "median"]:
                cal_result(path, type)
            elif type == "del_last_line":
                delete_last_line(path)


def delete_last_line(path):
    csv_file = open(path, encoding="utf-8")
    csv_reader = csv.reader(csv_file)
    data = [row for row in csv_reader][:-1]
    csv_file.close()
    csv_file = open(path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(data)
    csv_file.close()


def transfer100to10(origin_path, target_path):
    for root, dirs, files in os.walk(origin_path):
        if root == origin_path:
            for dir in dirs:
                model_path = os.path.join(origin_path, dir)
                model_name = dir
                for root, dirs, files in os.walk(model_path):
                    if root == model_path:
                        for dir in dirs:
                            dataset_path = os.path.join(model_path, dir)
                            datasets_name = dir
                            for root, dirs, files in os.walk(dataset_path):
                                if root == dataset_path:
                                    for file in files:
                                        file_path = os.path.join(dataset_path, file)
                                        print("read file:" + file_path)
                                        target_dir = os.path.join(target_path, model_name)
                                        target_dir = os.path.join(target_dir, datasets_name)
                                        Path(target_dir).mkdir(parents=True, exist_ok=True)
                                        target_file = os.path.join(target_dir, file)
                                        data_100 = pd.read_csv(file_path)
                                        colums = data_100.columns
                                        data = []
                                        for i in range(1, 11):
                                            s_10 = data_100.iloc[:i*10]
                                            s_1 = []
                                            s_1.append(s_10.iloc[0, 0])
                                            s_1.append(s_10.iloc[0, 1])
                                            for i in range(2, len(colums)):
                                                s_1.append(s_10[colums[i]].mean())
                                            data.append(s_1)
                                        data_10 = pd.DataFrame(data=data, columns=colums)
                                        data_10.to_csv(target_file, index=False)
                                        print("output file:" + target_file)







if __name__ == "__main__":
    transfer100to10(cf.origin_results_path, cf.origin_results_10_path)
    pass
