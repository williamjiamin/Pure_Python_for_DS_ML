# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号1.乐学偶得（lexueoude）2.乐学FinTech (LoveShareFinTech)

from random import seed
from random import randrange
from csv import reader
from math import sqrt


# 1.读取数据 csv

def read_our_csv_file(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# dataset = read_our_csv_file('abalone.csv')
# print(dataset)

# 2.数据类型转换（str to float）


def change_string_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# 3.数据类型转换（str to int）

def change_string_to_int(dataset, column):
    class_value = [row[column] for row in dataset]
    find_the_unique_class = set(class_value)
    lookup = dict()
    for i, value in enumerate(find_the_unique_class):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# 4.正则化
def find_the_min_and_max_of_our_data(dataset):
    min_and_max_list = list()
    for i in range(len(dataset[0])):
        column_value = [row[i] for row in dataset]
        the_min_value = min(column_value)
        the_max_value = max(column_value)
        min_and_max_list.append([the_min_value, the_max_value])
    return min_and_max_list


def normalize_our_data(dataset, min_and_max_list):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - min_and_max_list[i][0]) / (min_and_max_list[i][1] - min_and_max_list[i][0])


# 5. k fold切分数据，
# 注意：不改变原数据
def k_fold_cross_validation(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    every_fold_size = (len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < every_fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# 6.判断准确性（accuracy）
def calculate_our_model_accuracy(actual, predicted):
    correct_counter = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct_counter += 1
    return correct_counter / float(len(actual)) ** 100.0
