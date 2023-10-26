import time
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

import Configuration as cf
from sklearn.model_selection import StratifiedKFold
import PerformanceMeasure as measure
# different model
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression, BayesianRidge, SGDRegressor
import sklearn.pipeline as pl
import sklearn.linear_model as lm
import sklearn.preprocessing as sp
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
import CalculateResultMean as pr
from LinearLTR import LTR
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from deepforest import CascadeForestClassifier
from RankSVM import RankSVM

from SimpleReinforcementMethod import SimpleOrdinalClassifier
from rankboostYin import *
from TransformResultToComparsion import transform_result
from GetMedianComparisonResults import gen_heat_result
from heat_map_pic import do_draw
from SKESD_test import gen_skesd
from CalculateResultMean import cal_all

warnings.filterwarnings('ignore')


def get_origin_xy(path):
    """
    Perform basic preprocessing on the raw data:
    (1) Convert non-numeric values to one-hot encoding.
    (2) Separate features and labels.
    (3) Normalize the data.
    :param path: the location of the dataset
    :return: x——features， y——label
    """
    data = pd.read_csv(path)
    # Perform one-hot encoding on "modifier_type" and "visibility_type".
    data = pd.get_dummies(data)
    data_x = data.iloc[:, 1:]
    # Preprocess the features.
    data_x_values = MinMaxScaler().fit_transform(data_x.values)
    data_x = pd.DataFrame(data_x_values, columns=data_x.columns)
    data_y = data.iloc[:, 0]
    return data_x, data_y


def process_data(path, data_type, process_type):
    """
    Train on different models for each dataset to obtain the model performance.
    :param path: Location of the dataset files.
    :param data_type: Type of the dataset (e.g., GodClass, FeatureEnvy, etc.).
    :param process_type: Processing method (e.g., meta-learning).
    :return:
    """
    # 1. Retrieve features and targets from the original dataset.
    data_x, data_y = get_origin_xy(path)
    data_x = data_x.values
    data_y = data_y.values

    # 2. 10 times 10-fold cross
    for i in range(10):
        # Perform ten-fold cross-validation.
        kFold = StratifiedKFold(10, shuffle=True, random_state=np.random.RandomState(i))

        num = -1
        for train, test in kFold.split(data_x, data_y):
            num += 1
            # (1) get train_x and test_x
            train_x = data_x[train]
            test_x = data_x[test]
            # (2) get train_y and test_y
            train_y = data_y[train]
            test_y = data_y[test]

            # (3) Train different models.
            '''
             #[Classification]
            '''
            # CART
            model_name = "CART"
            dtree = DecisionTreeClassifier(criterion="gini")
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(dtree, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # O-CART [SimpleOrdinalClassifier enhanced]
            model_name = "O-CART"
            dtree = DecisionTreeClassifier(criterion="gini")
            model = SimpleOrdinalClassifier(dtree)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # RandomForest
            model_name = "RF"
            rfModel = RandomForestClassifier()
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(rfModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # O-RF
            model_name = "O-RF"
            rfModel = RandomForestClassifier()
            model = SimpleOrdinalClassifier(rfModel)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # NB
            model_name = "NB"
            gsModel = GaussianNB()
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(gsModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # O-NB[Ordinal version]
            model_name = "O-NB"
            gsModel = GaussianNB()
            model = SimpleOrdinalClassifier(gsModel)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)


            # LogR
            model_name = "LogR"
            linearModel = LogisticRegression()
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(linearModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # O-LogR
            model_name = "O-LogR"
            linearModel = LogisticRegression()
            model = SimpleOrdinalClassifier(linearModel)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # KNN
            model_name = "KNN"
            knnModel = KNeighborsClassifier(n_neighbors=9)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(knnModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # O-KNN
            model_name = "O-KNN"
            knnModel = KNeighborsClassifier(n_neighbors=9)
            model = SimpleOrdinalClassifier(knnModel)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # Bagging
            model_name = "Bagging"
            bgModel = BaggingClassifier(base_estimator=DecisionTreeClassifier())
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(bgModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # O-Bagging
            model_name = "O-Bagging"
            bgModel = BaggingClassifier(base_estimator=DecisionTreeClassifier())
            model = SimpleOrdinalClassifier(bgModel)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # AdaBoost
            model_name = "AdaBoost"
            adaModel = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(adaModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # O-AdaBoost
            model_name = "O-AdaBoost"
            adaModel = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
            model = SimpleOrdinalClassifier(adaModel)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # XGBoost
            model_name = "XGB"
            xgmodel = XGBClassifier()
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(xgmodel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # O-XGBoost
            model_name = "O-XGB"
            xgmodel = XGBClassifier()
            model = SimpleOrdinalClassifier(xgmodel)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # DeepForest【DF】
            model_name = "DF"
            dfModel = CascadeForestClassifier()
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(dfModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # O-DF
            model_name = "O-DF"
            dfModel = CascadeForestClassifier()
            model = SimpleOrdinalClassifier(dfModel)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            '''
                # [Classification model]
            '''

            # Linear Regression
            model_name = "LR"
            model = LinearRegression()
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # Polynomial Regression
            model_name = "PR"
            polyModel = pl.make_pipeline(
                sp.PolynomialFeatures(3),
                lm.LinearRegression()
            )
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(polyModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # RandomForest Regression
            model_name = "RFR"
            rfModel = RandomForestRegressor(100, max_depth=3, max_leaf_nodes=4)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(rfModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # Decision Tree Regression
            model_name = "DTR"
            dtModel = DecisionTreeRegressor()
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(dtModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # 6.2 Bayesian Ridge Regression
            model_name = "BRR"
            nbModel = BayesianRidge()
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(nbModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # 6.3 Neural Network Regression
            model_name = "NNR"
            nnrModel = MLPRegressor(hidden_layer_sizes=64)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(nnrModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # 6.4 Support Vector Regression
            model_name = "SVR"
            svrModel = SVR()
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(svrModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # 6.5 K-nearest Neighbors Regression
            model_name = "KNR"
            knnrModel = KNeighborsRegressor(n_neighbors=9)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(knnrModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # 6.6 Gradient Boosting Regression
            model_name = "GBR"
            gbrModel = GradientBoostingRegressor()
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(gbrModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # 6.7 Gaussian Process Regression
            model_name = "GPR"
            gprModel = GaussianProcessRegressor(n_restarts_optimizer=4)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(gprModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # 6.8 Stochastic Gradient Descent Regression
            model_name = "SDGR"
            sgdrModel = SGDRegressor()
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(sgdrModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            '''
                # [pairwise + pairlist]
            '''
            # RankingSVM
            model_name = "RankingSVM"
            rsvm_model = RankSVM()
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(rsvm_model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # 2: RankBoost 1: RankNet 5: LambdaRank 7: ListNet 3: AdaRank 4: CoordinateAscent
            method_index = [2, 1, 5, 7, 3, 4]
            method_name = ["RankBoost", "RankNet", "LambdaRank", "ListNet", "AdaRank", "CoordinateAscent"]

            ## Convert to the data format required for model training and testing in RankLib.
            train_dat_path = create_datYin(f"train", f"{data_type}_{i}{num}", train_x, train_y)
            test_dat_path = create_datYin(f"test", f"{data_type}_{i}{num}", test_x, test_y)

            for j in range(len(method_index)):
                print("The current ranker is set to be：", method_index[j])
                model_name = method_name[j]
                # Evaluate the model.
                start_time = time.time()
                measure.evaluate_model(method_index[j], model_name, data_type, train_dat_path, train_y, test_dat_path,
                                       test_y, start_time)

            # EALTR
            model_name = "EALTR"
            model = LTR(x=train_x, y=train_y, data_type=data_type, optimized_metric="relserverity@20%")
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)


            model_name = "SVM"
            smoModel = SVC(kernel="rbf", decision_function_shape="ovo")
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(smoModel, model_name, data_type, train_x, train_y, test_x, test_y, start_time)

            # O-SVM
            model_name = "O-SVM"
            smoModel = SVC(kernel="rbf", decision_function_shape="ovo", probability=True)
            model = SimpleOrdinalClassifier(smoModel)
            # Evaluate the model.
            start_time = time.time()
            measure.evaluate_model(model, model_name, data_type, train_x, train_y, test_x, test_y, start_time)
    pr.cal_all(cf.prediction_path)



def do_run():
    # Calculate the average value and replace NaN values.
    cal_all(cf.origin_results_path, "median")
    # 1. Convert the generated data into comparative data.
    transform_result(cf.origin_results_path, cf.transform_path)
    # 2. Convert the data into median and mean values for heat.
    gen_heat_result("mean")
    gen_heat_result("median")
    # 3. Generate a PDF for the heat.
    do_draw("mean")
    do_draw("median")
    # 4. Execute the generated comparative data in R language.
    # 5. Generate SK-ESD results.
    gen_skesd()
    # # 6. Generate a PDF for SK-ESD.
    do_draw("skesd")
    pass


if __name__ == "__main__":
    do_run()
    pass
