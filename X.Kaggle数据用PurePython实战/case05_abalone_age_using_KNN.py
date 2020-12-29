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
    every_fold_size = int(len(dataset) / n_folds)
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
    return correct_counter / float(len(actual)) * 100.0


# 7.给我们的算法进行评估（打分）

def how_good_is_our_algo(dataset, algo, n_folds, *args):
    folds = k_fold_cross_validation(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_dataset = list(folds)
        train_dataset.remove(fold)
        train_dataset = sum(train_dataset, [])
        test_dataset = list()
        for row in fold:
            row_copy = list(row)
            test_dataset.append(row_copy)
            row_copy[-1] = None
        predicted = algo(train_dataset, test_dataset, *args)
        actual = [row[-1] for row in fold]
        accuracy = calculate_our_model_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


# 8.计算欧几里德举例

def calculate_euclidiean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# 9.找到最近的k个点
def get_our_neighbors(train_dataset, test_row, num_of_neighbors):
    distances = list()
    for train_dataset_row in train_dataset:
        dist = calculate_euclidiean_distance(test_row, train_dataset_row)
        distances.append((train_dataset_row, dist))
    distances.sort(key=lambda every_tuple: every_tuple[1])
    neighbors = list()
    for i in range(num_of_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# 10.做预测
def make_prediction(train_dataset, test_row, num_of_neighbors):
    neighbors = get_our_neighbors(train_dataset, test_row, num_of_neighbors)
    output = [row[-1] for row in neighbors]
    our_prediction = max(set(output), key=output.count)
    return our_prediction


# 11.运用KNN算法
def get_our_prediction_using_knn_algo(train_dataset, test_dataset, num_of_neighbors):
    predictions = list()
    for test_row in test_dataset:
        our_prediction = make_prediction(train_dataset, test_row, num_of_neighbors)
        predictions.append(our_prediction)
    return predictions


seed(1)
dataset = read_our_csv_file('abalone.csv')
for i in range(1, len(dataset[0])):
    change_string_to_float(dataset, i)

change_string_to_int(dataset, 0)

# print(dataset)

n_folds = 10

num_neighbors = 7
scores = how_good_is_our_algo(dataset, get_our_prediction_using_knn_algo, n_folds, num_neighbors)

print('Our model\'s scores are : %s' % scores)
print('The mean accuracy is :%.3f%%' % (sum(scores) / float(len(scores))))
