# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号1.乐学偶得（lexueoude）2.乐学FinTech (LoveShareFinTech)


# 1.导入必备库
from random import seed
from random import randrange
from csv import reader
from math import sqrt


# 2.读取csv数据

def read_our_csv_file(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# dataset=read_our_csv_file("ionosphere.csv")
#
# print(dataset)

# 3.数据类型转换（string to float）
def convert_string_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# 4.类别数据转换为int数据（class string to int）

def convert_string_to_int(dataset, column):
    class_value = [row[column] for row in dataset]
    unique_class = set(class_value)
    look_up = dict()
    for i, value in enumerate(unique_class):
        look_up[value] = i
    for row in dataset:
        row[column] = look_up[row[column]]
    return look_up


# 5.为了让预测更加稳定，要对数据进行处理

def k_fold_cross_validation_split(dataset, n_folds):
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


# 6.计算一下准确性

def calculate_our_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# 7.评估算法

def whether_our_algo_is_good_or_not(dataset, algo, n_folds, *args):
    folds = k_fold_cross_validation_split(dataset, n_folds)
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

        accuracy = calculate_our_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


# 8.计算欧氏距离
def calculate_euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# 9.计算BMU

def calculate_BMU(codebooks, test_row):
    distances = list()
    for codebook in codebooks:
        dist = calculate_euclidean_distance(codebook, test_row)
        distances.append((codebook, dist))
    distances.sort(key=lambda every_tuple: every_tuple[1])
    return distances[0][0]


# 10.预测以及生成随机codebook
def predict(codebooks, test_row):
    bmu = calculate_BMU(codebooks, test_row)
    return bmu[-1]


def random_codebook(train):
    train_index = len(train)
    n_features = len(train[0])
    codebook = [train[randrange(train_index)][i] for i in range(n_features)]
    return codebook


# 11.对codebook进行训练

def train_our_codebooks(train, n_codebooks, learning_rate, epochs):
    codebooks = [random_codebook(train) for i in range(n_codebooks)]
    for epoch in range(epochs):
        rate = learning_rate * (1.0 - (epoch / float(epochs)))
        for row in train:
            bmu = calculate_BMU(codebooks, row)
            for i in range(len(row) - 1):
                error = row[i] - bmu[i]
                if bmu[-1] == row[-1]:
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error
    return codebooks


# 12.LVQ

def LVQ(train, test, n_codebooks, learning_rate, epochs):
    codebooks = train_our_codebooks(train, n_codebooks, learning_rate, epochs)
    predictions = list()
    for row in test:
        output = predict(codebooks, row)
        predictions.append(output)
    return predictions


seed(1)

dataset = read_our_csv_file("ionosphere.csv")
for i in range(len(dataset[0]) - 1):
    convert_string_to_float(dataset, i)

convert_string_to_int(dataset, len(dataset[0]) - 1)

#
# print(dataset)

n_folds = 5
learning_rate = 0.3
n_epochs = 50
n_codebooks = 50

scores = whether_our_algo_is_good_or_not(dataset, LVQ, n_folds, n_codebooks, learning_rate, n_epochs)

print("Our LVQ algo's scores are : %s " % scores)

print("Our LVQ's mean accracy is %.3f%%" % (sum(scores) / float(len(scores))))
