"""
Storage paths for project configurations.
"""
import os

# project root path
root_path = os.getcwd()
# jar path
jar_path = os.path.join(root_path, "RankLib.jar")

datasets_path = os.path.join(root_path, "datasets")
godclass_path = os.path.join(datasets_path, "dataset-god-class-modified.csv")
dataclass_path = os.path.join(datasets_path, "dataset-data-class-modified.csv")
longmethod_path = os.path.join(datasets_path, "dataset-long-method-modified.csv")
featureenvy_path = os.path.join(datasets_path, "dataset-feature-envy-modified.csv")

datasets_location = {"GodClass": godclass_path, "DataClass": dataclass_path,
                     "LongMethod": longmethod_path, "FeatureEnvy": featureenvy_path}
# path of generated result
prediction_path = os.path.join(root_path, "prediction")

empirical_results_root = "./empirical_results"

origin_results_path = os.path.join(empirical_results_root, "origin-results")
# The number 10 refers to recording the final results of only 10 iterations of 10-fold cross-validation out of a total of 100 iterations.
origin_results_10_path = os.path.join(empirical_results_root, "origin-results-10")

# -----------------compared results------------------
# Root path of the converted files, compared based on metrics.
transform_path = os.path.join(empirical_results_root, "compared-results")
transform_path_10 = os.path.join(empirical_results_root, "compared-results-10")
transform_path_include_mean = os.path.join(empirical_results_root, "compared-results-include-mean")

# ------------------ skesd -> heat map------------------
# The ranking data saved after executing in R language.
r_skesd_path = os.path.join(empirical_results_root, "first-skesd")
# Data for plotting the heatmap after the first execution of SK-ESD.
plot_data_path = os.path.join(empirical_results_root, "plot-data-results")
plot_data_path_10 = os.path.join(empirical_results_root, "plot-data-results-10")
# Path for plotting the heatmap after the first execution of SK-ESD.
heat_map_skesd_path = os.path.join(empirical_results_root, "heat-map-skesd")
heat_map_skesd_path_10 = os.path.join(empirical_results_root, "heat-map-skesd-10")

# ------------mean & median -> heat map-----------------
median_data_path = os.path.join(empirical_results_root, "median-data-results")
mean_data_path = os.path.join(empirical_results_root, "mean-data-results")
heat_map_median_path = os.path.join(empirical_results_root, "heat-map-median")
heat_map_mean_path = os.path.join(empirical_results_root, "heat-map-mean")

# ------- Path for storing the classifiers & effect size enhancement (O-Wilcoxon) results. --------
wilcoxon_path = os.path.join(empirical_results_root, "wilcoxon_results")

datasets_metrics_path = os.path.join(empirical_results_root, "datasets_metrics_results")


method_index = [2, 1, 5, 7, 3, 4]
method_name = ["RankBoost", "RankNet", "LambdaRank", "ListNet", "AdaRank", "CoordinateAscent"]

