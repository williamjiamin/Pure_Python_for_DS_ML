# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号：乐学FinTech

# 1.Load our csv data
# 1.读取csv文件
from csv import reader
from math import sqrt
from random import randrange, seed


def load_csv(data_file):
    data_set = list()
    with open(data_file, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data_set.append(row)
    return data_set


# print(load_csv('insurance.csv'))
# 发现数据均为字符串，我们接下来对字符串进行转化
# 最终要转换成float
# 2.数据类型转换
def string_converter(data_set, column):
    for row in data_set:
        """
        Return a copy of the string with leading and trailing whitespace removed.
        因为很可能出现这样的数据" 99.9 "
        If chars is given and not None, remove characters in chars instead.
        """
        row[column] = float(row[column].strip())


# a = "       99.9 ".strip()
#
# print(a)


# 模型预测的准确性基本判断方法RMSE（衡量模型的标尺）
# 3.root mean squared error

def calculate_RMSE(actual_data, predicted_data):
    sum_error = 0.0
    for i in range(len(actual_data)):
        predicted_error = predicted_data[i] - actual_data[i]
        sum_error += (predicted_error ** 2)
    mean_error = sum_error / float(len((actual_data)))
    return sqrt(mean_error)


# 4.测试集、训练集切分train/test split

def train_test_split(data_set, split):
    train = list()
    train_size = split * len(data_set)
    data_set_copy = list(data_set)
    while len(train) < train_size:
        index = randrange(len(data_set_copy))
        train.append(data_set_copy.pop(index))
    return train, data_set_copy


# 5.模型到底如何（train/test split）通过在训练集，测试集切分后，用RMSE进行衡量模型好坏

def how_good_is_our_algo(data_set, algo, split, *args):
    train, test = train_test_split(data_set, split)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)

    # 伪代码思想，先用algo统一代替具体算法
    predicted = algo(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = calculate_RMSE(actual, predicted)
    return rmse


# 6.为了实现简单线形回归的小玩意儿

def mean(values):
    return sum(values) / float(len(values))


def covariance(x, the_mean_of_x, y, the_mean_of_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - the_mean_of_x) * (y[i] - the_mean_of_y)
    return covar


def variance(values, mean):
    return sum([(x - mean) ** 2 for x in values])


def coefficients(data_set):
    x = [row[0] for row in data_set]
    y = [row[1] for row in data_set]
    the_mean_of_x = mean(x)
    the_mean_of_y = mean(y)

    # y=b1*x+b0
    b1 = covariance(x, the_mean_of_x, y, the_mean_of_y) / variance(x, the_mean_of_x)
    b0 = the_mean_of_y - b1 * the_mean_of_x

    return [b0, b1]


# 7.这里写简单线性回归的具体预测

def using_simple_linear_regression(train, test):
    # 套路：先弄一个空的容器出来，然后逐一处理放入
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        y_hat = b1 * row[0] + b0
        predictions.append(y_hat)
    return predictions


# 8.带入真实数据
# 可调项，避免"魔法数字"
seed(4)
split = 0.6

# 读取数据
data_set = load_csv('insurance.csv')
# 数据准备
for i in range(len(data_set[0])):
    string_converter(data_set, i)

rmse = how_good_is_our_algo(data_set, using_simple_linear_regression, split)

print("RMSE of our algo is : %.3f" % (rmse))
