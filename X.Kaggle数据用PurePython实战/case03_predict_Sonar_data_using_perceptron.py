# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号：乐学FinTech

# 1. import lib
from random import seed
from random import randrange
from csv import reader


# 2.write a csv/data reader
def read_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# 3.change string datatype
def change_string_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# filename = 'sonar.all-data.csv'
# dataset = read_csv(filename)
# for i in range(len(dataset[0]) - 1):
#     change_string_to_float(dataset, i)
# print(dataset)


# 4.change string column(class) to int
# 重点关注，核心思想在于通过i进行每一列的数据类型转换（i）

def change_str_column_to_int(dataset, column):
    class_value = [row[column] for row in dataset]
    unique_value = set(class_value)

    search_tool = dict()

    for i, value in enumerate(unique_value):
        search_tool[value] = i

    for row in dataset:
        row[column] = search_tool[row[column]]
    return search_tool


# 5. using k_folds cross validation

def k_folds_cross_validation(dataset, n_folds):
    dataset_split = list()
    # 对数据进行操作的时候，最好不要损坏原数据
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)

    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# 6.calculate the accuracy of our model
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# 7. whether the algo is good or not ?

def whether_the_algo_is_good_or_not(dataset, algo, n_folds, *args):
    folds = k_folds_cross_validation(dataset, n_folds)
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
        accuracy = calculate_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


# 8.make prediction

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0


# 9.using sgd(stochastic gradient descent) method to estimate weights

def estimate_our_weight_using_sgd_method(training_data, learning_rate, n_epoch):
    weights = [0.0 for i in range(len(training_data[0]))]
    for epoch in range(n_epoch):
        for row in training_data:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] = weights[0] + learning_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + learning_rate * error * row[i]
    return weights


# 10. using sgd method to make perceptron algo's prediction

def perceptron(training_data, testing_data, learning_rate, n_epoch):
    predictions = list()
    weights = estimate_our_weight_using_sgd_method(training_data, learning_rate, n_epoch)
    for row in testing_data:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions


# 11.using real sonar dataset
seed(1)

filename = 'sonar.all-data.csv'
dataset = read_csv(filename)
for i in range(len(dataset[0]) - 1):
    change_string_to_float(dataset, i)

change_str_column_to_int(dataset, len(dataset[0]) - 1)

n_folds = 3
learning_rate = 0.01
n_epoch = 500
scores = whether_the_algo_is_good_or_not(dataset, perceptron, n_folds, learning_rate, n_epoch)

print("The score of our model is : %s " % scores)
print("The average accuracy is : %3.f%% ， The baseline is 50%%" % (sum(scores) / float(len(scores))))
