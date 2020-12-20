# Created by william from lexueoude.com. 更多正版技术视频讲解，
# 公众号1.乐学偶得（lexueoude）2.乐学FinTech (LoveShareFinTech)


from csv import reader
from random import randrange
from math import sqrt
from math import exp
from math import pi


# csv reader

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# dataset = load_csv('iris.csv')
# print(dataset)

def convert_str_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def convert_str_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique_value = set(class_values)
    look_up = dict()
    for i, value in enumerate(unique_value):
        look_up[value] = i
    for row in dataset:
        row[column] = look_up[row[column]]
    return look_up


def n_fold_cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def calculate_our_model_accuracy(actual, predicted):
    correct_count = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct_count += 1
    return correct_count / float(len(actual)) * 100.0


def whether_our_model_is_good_or_not(dataset, algo, n_folds, *args):
    folds = n_fold_cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])

        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None

        predicted = algo(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = calculate_our_model_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


def split_our_data_by_class(dataset):
    splited = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in splited):
            splited[class_value] = list()
        splited[class_value].append(vector)
    return splited


def mean(a_list_of_numbers):
    return sum(a_list_of_numbers) / float(len(a_list_of_numbers))


def stdev(a_list_of_numbers):
    the_mean_of_a_list_numbers = mean(a_list_of_numbers)
    variance = sum([(x - the_mean_of_a_list_numbers) ** 2 for x in a_list_of_numbers]) / float(
        len(a_list_of_numbers) - 1)

    return sqrt(variance)


def describe_our_data(dataset):
    description = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del (description[-1])
    return description


def describe_our_data_by_class(dataset):
    data_split = split_our_data_by_class(dataset)
    description = dict()
    for class_value, rows in data_split.items():
        description[class_value] = describe_our_data(rows)
    return description
