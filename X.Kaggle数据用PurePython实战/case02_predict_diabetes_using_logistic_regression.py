# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号：乐学FinTech
# github.com/williamjiamin

from random import seed
from random import randrange
from csv import reader
from math import exp


# 1.Load our data using csv reader

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


# 2. convert string in list of lists to float(data type change)


def change_string_to_float(dataset, column):
    for row in dataset:
        # 把前后对空格去掉
        row[column] = float(row[column].strip())


# ret_data = load_data_from_csv_file('diabetes.csv')
#
# change_string_to_float(ret_data, 1)
#
# print(ret_data)


# 3.Find the min and max value of our data

def find_the_min_and_max_of_our_data(dataset):
    min_max_list = list()
    for i in range(len(dataset[0])):
        values_in_every_column = [row[i] for row in dataset]
        the_min_value = min(values_in_every_column)
        the_max_value = max(values_in_every_column)
        min_max_list.append([the_min_value, the_max_value])
    return min_max_list


# 4.rescale our data so it fits to range 0 ~ 1

def rescale_our_data(dataset, min_max_list):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - min_max_list[i][0]) / (min_max_list[i][1] - min_max_list[i][0])


# 5.k fold train and test split

def k_fold_cross_validation(dataset, how_many_fold_do_you_want):
    splited_dataset = list()
    # 对原数据进行处理的时候，尽量不要改动源数据（可以通过创建copy的方式对copy数据进行处理）
    copy_dataset = list(dataset)
    how_big_is_every_fold = int(len(dataset) / how_many_fold_do_you_want)
    # 创建一个空的盒子，然后逐一随机选取数据放入盒子中
    for i in range(how_many_fold_do_you_want):
        box_for_my_fold = list()
        while len(box_for_my_fold) < how_big_is_every_fold:
            some_random_index_in_the_fold = randrange(len(copy_dataset))
            box_for_my_fold.append(copy_dataset.pop(some_random_index_in_the_fold))
        splited_dataset.append(box_for_my_fold)
    return splited_dataset


# 6.Calculate the accuracy of our model

def calculate_the_accuracy_of_our_model(actual_data, predicted_data):
    counter_of_correct_prediction = 0
    for i in range(len(actual_data)):
        if actual_data[i] == predicted_data[i]:
            counter_of_correct_prediction += 1
    return counter_of_correct_prediction / float(len(actual_data)) * 100.0


# 7. how good is our algo ?

def how_good_is_our_algo(dataset, algo, how_many_fold_do_you_want, *args):
    folds = k_fold_cross_validation(dataset, how_many_fold_do_you_want)
    scores = list()
    for fold in folds:
        training_data_set = list(folds)
        training_data_set.remove(fold)
        training_data_set = sum(training_data_set, [])
        testing_data_set = list()

        # 保险操作，去除真实数据，避免影响模型的学习结果
        for row in fold:
            row_copy = list(row)
            testing_data_set.append(row_copy)
            row_copy[-1] = None
        predicted = algo(training_data_set, testing_data_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = calculate_the_accuracy_of_our_model(actual, predicted)
        scores.append(accuracy)
    return scores


# 8. make prediction by using the coef

def prediction(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return 1 / (1.0 + exp(-yhat))


# 9. using stochastic gradient descent to estimate our coef of logistic regression


def estimate_coef_lr_using_sgd_method(training_data, learning_rate, how_many_epoch_do_you_want):
    coef = [0.0 for i in range(len(training_data[0]))]
    for epoch in range(how_many_epoch_do_you_want):
        for row in training_data:
            yhat = prediction(row, coef)
            error = row[-1] - yhat
            coef[0] = coef[0] + learning_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] + learning_rate * error * yhat * (1.0 - yhat) * row[i]
    return coef


# 10. Logistic Regression's prediction in a our func

def logistic_regression(training_data, testing_data, learning_rate, how_many_epoch_do_you_want):
    predictions = list()
    coef = estimate_coef_lr_using_sgd_method(training_data, learning_rate, how_many_epoch_do_you_want)
    for row in testing_data:
        yhat = prediction(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return predictions


# Using Kaggle diabetes dataset to test our model

seed(1)

dataset = load_data_from_csv_file('diabetes.csv')

for i in range(len(dataset[0])):
    change_string_to_float(dataset, i)

min_max_value = find_the_min_and_max_of_our_data(dataset)

# make data range form 0 to 1
rescale_our_data(dataset, min_max_value)

how_many_fold_do_you_want = 10

learning_rate = 0.1
how_many_epoch_do_you_want = 1000

scores = how_good_is_our_algo(dataset, logistic_regression, how_many_fold_do_you_want, learning_rate,
                              how_many_epoch_do_you_want)

print("The scores of our model are %s" % scores)
print("The average accuracy of our model is %.3f" % (sum(scores) / float(len(scores))))

# 77.63