import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
import os
import Configuration as cf
from pathlib import Path
import math
wine = load_wine()
data = wine.data
lables = wine.target
feaures = wine.feature_names
df = pd.DataFrame(data, columns=feaures)


def ShowGRAHeatMap(DataFrame, classifier):
    colormap = plt.cm.hsv
    ylabels = DataFrame.columns.values.tolist()
    file_name = classifier.split('.csv')[0]
    f, ax = plt.subplots(figsize=(15, 15))
    ax.set_title(file_name + ' Correlation', fontsize=18, weight='heavy', family='Times New Roman')


    with sns.axes_style("white"):
        sns.heatmap(DataFrame,
                    cmap="YlGnBu",
                    annot=True,
                    yticklabels=ylabels,
                    annot_kws = {'size': 20, 'family': 'Times New Roman'},
                    cbar_kws = {"orientation": "horizontal", "shrink": 1}
                    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=65, fontsize=15, family='Times New Roman', weight='heavy')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15, family='Times New Roman', weight='heavy')
    file_name = classifier.split('.csv')[0]
    Path(cf.heat_map_metrics).mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(cf.heat_map_metrics, file_name + '.pdf')
    print(file_path)
    f.savefig(file_path)

if __name__ == '__main__':
    # path = configuration_file().save_covar + '/1030/'
    path = cf.metircs_pearson_data_path
    for root, dirs, files in os.walk(path):
        for file in files:
            data = pd.read_csv(os.path.join(path, file))
            # print(data)
            ShowGRAHeatMap(data, file)
