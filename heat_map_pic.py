import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings
from pathlib import Path

import Configuration as cf
import os

warnings.filterwarnings('ignore')

plt.rc('font', family='Times New Roman')


def draw(path, save_path, filename, except_col=[]):
    df = pd.read_excel(path)
    row_name = df.iloc[:, 0].values # datasets
    col_name = df.columns.tolist()[1:]      # classifier
    col_name = [col for col in col_name if col not in except_col]

    df = df.fillna(df.mean())
    data = np.array(df.loc[:, col_name].values)    #datasets

    f, ax = plt.subplots(constrained_layout=True)

    sns.heatmap(data, cmap='Reds', square=True, annot=True, cbar=False, fmt='.2f',
                annot_kws={'size': 4, 'color': 'black', 'family': 'Times New Roman'})

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 8,
            }

    plt.xticks(fontproperties='Times New Roman', size=6)
    plt.yticks(fontproperties='Times New Roman', fontsize=6)

    n1 = []
    tmp = 0.5
    for i in range(len(col_name)):
        n1.append(i + tmp)
    ax.set_xticks(n1)
    ax.set_xticklabels(col_name, rotation=90)
    ax.set_xlabel('CodeSmell Severity Prioritization Algorithms', fontdict=font)

    n = []
    temp = 0.5
    for i in range(len(row_name)):
        n.append(i + temp)
    ax.set_yticks(n)
    ax.set_yticklabels(row_name, rotation=0)
    ax.set_ylabel('Testing Datasets', fontdict=font)
    print(save_path + '\\' + filename + '.pdf' + "created")
    # plt.show()
    f.savefig(save_path + '\\' + filename + '.pdf')


def do_draw(type="mean", except_model=[]):
    if type == "median":
        folder_path = cf.median_data_path
        save_path = cf.heat_map_median_path
    elif type == "mean":
        folder_path = cf.mean_data_path
        save_path = cf.heat_map_mean_path
    elif type == "skesd":
        folder_path = cf.plot_data_path
        save_path = cf.heat_map_skesd_path

    # except_model = ['O-NB', 'O-CART', 'O-SVM', 'O-KNN', 'O-LogR', 'O-RF', 'O-XGB', 'O-DF', 'O-Bagging', 'O-AdaBoost']
    except_model = ["RankingSVM", "RankBoost", "RankNet", "LambdaRank", "ListNet", "AdaRank", "CoordinateAscent", "EALTR"]
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(folder_path):
        if root == folder_path:
            for file in files:
                draw(os.path.join(folder_path, file), save_path, file.split('.xls')[0], except_model)
    pass

if __name__ == '__main__':
    pass


