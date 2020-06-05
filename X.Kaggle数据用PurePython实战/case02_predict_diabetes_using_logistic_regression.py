# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号：乐学FinTech

from random import seed
from random import randrange
from csv import reader
from math import exp


# Load our data using csv reader

def load_data_from_csv_file(file_name):
    # 通过一个list容器去装数据
    dataset = list()
    with open(file_name, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# print(load_data_from_csv_file('diabetes.csv'))


# convert string in list of lists to float(data type change)


def change_string_to_float(dataset, column):
    for row in dataset:
        # 把前后对空格去掉
        row[column] = float(row[column].strip())


# ret_data = load_data_from_csv_file('diabetes.csv')
#
# change_string_to_float(ret_data, 1)
#
# print(ret_data)


# Find the min and max value of our data

def find_the_min_and_max_of_our_data(dataset):
    min_max_list = list()
    for i in range(len(dataset[0])):
        values_in_every_column = [row[i] for row in dataset]
        the_min_value = min(values_in_every_column)
        the_max_value = max(values_in_every_column)
        min_max_list.append([the_min_value, the_max_value])
    return min_max_list
