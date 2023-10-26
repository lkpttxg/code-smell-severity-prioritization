import pandas as pd
import os
import time
import numpy as np
import Configuration as cf

def gen_skesd():
    # sk_esd root path of the generated data.
    folder_path = cf.r_skesd_path
    # Location where the data for plotting is stored.
    target_path = cf.plot_data_path

    print("start time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    starttime = time.time()

    # # # Integrate the data processed for the heatmap.
    # Index columns of the data (four datasets).
    datasets = ["GodClass", "DataClass", "LongMethod", "FeatureEnvy"]
    # model_name_header = ["NB", "CART", "SVM", "KNN", "LogR", "RF", "XGB", "DF",
    #   "Bagging", "AdaBoost", "O-NB", "O-CART", "O-SVM", "O-KNN",
    #   "O-LogR", "O-RF", "O-XGB", "O-DF", "O-Bagging", "O-AdaBoost",
    #   "LR", "PR", "RFR", "DTR", "BRR", "NNR",
    #   "SVR", "KNR", "GBR", "GPR", "SDGR"]
    model_name_header = ["NB", "CART", "SVM", "KNN", "LogR", "RF", "XGB", "DF",
                         "Bagging", "AdaBoost", "O-NB", "O-CART", "O-SVM", "O-KNN",
                         "O-LogR", "O-RF", "O-XGB", "O-DF", "O-Bagging", "O-AdaBoost",
                         "LR", "PR", "RFR", "DTR", "BRR", "NNR", "SVR", "KNR", "GBR", "GPR", "SDGR", "RankingSVM",
                         "RankBoost",
                         "RankNet", "LambdaRank", "ListNet", "AdaRank", "CoordinateAscent", "EALTR"]

    # Integrate the data into DataFrame format.
    df = pd.DataFrame(index=datasets, columns=model_name_header)

    # Get the corresponding path, subfolder names, and file names for the dataset folder.
    for root, dirs, files in os.walk(folder_path):
        if root == folder_path:
            thisroot = root
            for dir in dirs:
                # Get the path to the subfolder of the dataset.
                dir_path = os.path.join(thisroot, dir)
                for root, dirs, files, in os.walk(dir_path):
                    for file in files:
                        row_name = file.split("_")[0]
                        if row_name == "dataclass":
                            row_name = "DataClass"
                        elif row_name == "featureenvy":
                            row_name = "FeatureEnvy"
                        elif row_name == "godclass":
                            row_name = "GodClass"
                        elif row_name == "longmethod":
                            row_name = "LongMethod"
                        file_path = os.path.join(dir_path, file)
                        # ranking result
                        res = []
                        # model names
                        name = []
                        # open .txt file
                        with open(file_path, 'r') as f:
                            content = f.readlines()
                            content.pop(0)
                            for line in content:
                                line = line.strip()
                                name.append(eval(line.split(' ')[0]).replace('.', '-'))
                                res.append(line.split(' ')[1])
                        res = pd.DataFrame(res)
                        res.index = name
                        res.columns = [row_name]
                        res = res.T
                        for model in model_name_header:
                            df.loc[row_name, model] = res.loc[row_name, model]

                    # Storage path
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    file_name1 = "SKESD_%s.xls" % (dir)
                    file_path = os.path.join(target_path, file_name1)
                    print(file_path)
                    df.to_excel(file_path)

    print("end time:", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    endtime = time.time()
    print('Time consumed:', endtime - starttime, 's')

if __name__ == '__main__':
    pass
    pass