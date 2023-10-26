# coding=utf-8
import os
import subprocess
import Configuration as cf
import xlrd

path = cf.root_path
jar_path = cf.jar_path


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")

    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False


def create_datYin(trainOrtestOrVali, filename, X, Y):
    # for each_group in os.listdir(path+"data"):
    each_group = filename
    x_data, y_data = X, Y
    dat_name = str(each_group) + "+" + trainOrtestOrVali + ".dat"
    dat_dir_path = os.path.join(path, "dat_file")
    mkdir(dat_dir_path)
    dat_path = os.path.join(dat_dir_path, dat_name)
    if not os.path.exists(dat_path):
        f = open(dat_path, "a", encoding="utf-8")
    else:
        f = open(dat_path, "w+", encoding="utf-8")
    for i in range(len(x_data)):
        f.write(str(y_data[i]) + " ")
        f.write(str(each_group.split(".")[0].replace("-", ":")) + " ")
        for j in range(1, len(x_data[i]) + 1):
            if j == len(x_data[i]):
                f.write(str(j) + ":" + str(x_data[i][j - 1]))
            else:
                f.write(str(j) + ":" + str(x_data[i][j - 1]) + " ")
        f.write("\n")
    return dat_path
    pass


def train_modelYin(ranker, train_dat_path):
    print("ranker:", ranker)
    print("train file path:{0}，yes?:{1}".format(train_dat_path, os.path.exists(train_dat_path)))
    dat_file = str(os.path.split(train_dat_path)[-1])
    save_model_path = os.path.join(path, "model")
    model_name = str(dat_file.split("+")[0]) + "_" + cf.method_name[cf.method_index.index(ranker)] +".dat"
    mkdir(save_model_path)
    model_path = os.path.join(save_model_path, model_name)
    command = "java -jar " + jar_path + " -train " + train_dat_path + " -metric2t ERR@20 -ranker " + str(
                ranker) + " -save " + model_path
    try:
        res = subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as exc:
        print(exc.returncode)
        print("results:" + str(exc.returncode))
    finally:
        return model_path
        pass
    pass


def pred_test_dataYin(test_dat_path, model_path):
    print("test_dat_path:{0}, yes:{1}".format(test_dat_path, os.path.exists(test_dat_path)))
    print("model path:{0}, yes?:{1}".format(model_path, os.path.exists(model_path)))
    test_dat_name = str(os.path.split(test_dat_path)[-1])
    save_result_data_dir = os.path.join(path, "test_result")
    mkdir(save_result_data_dir)
    result_dat_name = "result_" + str(os.path.split(model_path)[-1])
    result_dat_path = os.path.join(save_result_data_dir, result_dat_name)
    command = "java -jar " + jar_path + " -load " + model_path + " -rank " + test_dat_path + " -score " + result_dat_path
    # commandResult = os.system(command)
    try:
        res = subprocess.check_call(command, shell=True)
    except subprocess.CalledProcessError as exc:
        print(exc.returncode)
        print("results:" + str(exc.returncode))
    finally:
        return result_dat_path
        pass
    pass


def get_pred_bug(result_dat_path):
    print("result_dat_path：{0}, yes?:{1}".format(result_dat_path, os.path.exists(result_dat_path)))
    pred_bug = []
    lines = open(result_dat_path, "r").readlines()
    print('lines',lines)

    for each_line in lines:
        pred_bug.append(float(each_line.replace("\n", "")))
    return pred_bug


def RankalgorithmCreatedata(trainX, trainY, testX, testY, filename):
    train_dat_path = create_datYin("train", filename, trainX, trainY)
    test_dat_path = create_datYin("test", filename, testX, testY)
    return train_dat_path, test_dat_path


def RankalgorithmTrainandtest(ranker, filename, train_dat_path, test_dat_path):
    model_path = train_modelYin(ranker, train_dat_path)
    result_dat_path = pred_test_dataYin(test_dat_path, model_path)
    return get_pred_bug(result_dat_path)
