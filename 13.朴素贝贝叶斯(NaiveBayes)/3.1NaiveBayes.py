# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号1.乐学偶得（lexueoude）2.乐学FinTech (LoveShareFinTech)

from math import sqrt
from math import pi
from math import exp


def split_our_data_by_class(dataset):
    splited_data = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in splited_data):
            splited_data[class_value] = list()
        splited_data[class_value].append(vector)
    return splited_data


def calculate_the_mean(a_list_of_num):
    mean = sum(a_list_of_num) / float(len(a_list_of_num))
    return mean


def calculate_the_standard_deviation(a_list_of_num):
    the_mean = calculate_the_mean(a_list_of_num)
    the_variance = sum([(x - the_mean) ** 2 for x in a_list_of_num]) / float(len(a_list_of_num) - 1)
    std = sqrt(the_variance)
    return std


def describe_our_data(dataset):
    description = [(calculate_the_mean(column),
                    calculate_the_standard_deviation(column),
                    len(column)) for column in zip(*dataset)]
    del (description[-1])
    return description


def describe_our_data_by_class(dataset):
    splited_data = split_our_data_by_class(dataset)
    data_description = dict()
    for class_value, rows in splited_data.items():
        data_description[class_value] = describe_our_data(rows)
    return data_description


def calculate_the_probability(x, mean, stdev):
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    result = (1 / (sqrt(2 * pi) * stdev)) * exponent
    return result


def calculate_class_probability(description, row):
    total_rows = sum([description[label][0][2] for label in description])
    probabilities = dict()
    for class_value, class_description in description.items():
        probabilities[class_value] = description[class_value][0][2] / float(total_rows)
        for i in range(len(class_description)):
            mean, stdev, count = class_description[i]
            probabilities[class_value] *= calculate_the_probability(row[i], mean, stdev)
    return probabilities


dataset = [[0.8, 2.3, 0],
           [2.1, 1.6, 0],
           [2.0, 3.6, 0],
           [3.1, 2.5, 0],
           [3.8, 4.7, 0],
           [6.1, 4.4, 1],
           [8.6, 0.3, 1],
           [7.9, 5.3, 1],
           [9.1, 2.5, 1],
           [6.8, 2.7, 1]]

description = describe_our_data_by_class(dataset)

probability = calculate_class_probability(description, dataset[0])

print(probability)
