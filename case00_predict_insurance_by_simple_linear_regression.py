# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号：乐学FinTech

# Load our csv data
from csv import reader
from math import sqrt


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


# 模型预测的准确性基本判断方法RMSE
# root mean squared error

def calculate_RMSE(actual_data, predicted_data):
    sum_error = 0.0
    for i in range(len(actual_data)):
        predicted_error = predicted_data[i] - actual_data[i]
        sum_error += (predicted_error ** 2)
    mean_error = sum_error / float(len((actual_data)))
    return sqrt(mean_error)
