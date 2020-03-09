# 1.load csv
# 2.convert string to float
# 3.normalization
# 4.cross validation
# 5.evaluate our algo(RMSE)

# 1 . Import standard Lib
from csv import reader
from math import sqrt
from random import randrange
from random import seed


# 2. Load our csv file

def csv_loader(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


dataset_list = csv_loader('winequality-white.csv')

print(dataset_list)


# 3.Convert our datatype

def string_to_float_converter(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# 4.find the min and max of our dataset
def find_the_min_and_max_of_our_dataset(dateset):
    min_max_list = list()
    for i in range(len(dateset[0])):
        col_value = [row[i] for row in dateset]
        max_value = max(col_value)
        min_value = min(col_value)
        min_max_list.append([min_value, max_value])
    return min_max_list


# 5.normalization our data

def normalization(dataset, min_max_list):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - min_max_list[i][0]) / (min_max_list[i][1] - min_max_list[i][0])


# 6.splitting our data

def k_fold_cross_validation_split(dataset, n_folds):
    split_dataset = list()
    copy_dataset = list(dataset)
    every_fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < every_fold_size:
            index = randrange(len(copy_dataset))
            fold.append(copy_dataset.pop(index))
        split_dataset.append(fold)
    return split_dataset


# 7.using root mean squared error method to calculate our model

def rmse_method(actual_data, predicted_data):
    sum_of_error = 0.0
    for i in range(len(actual_data)):
        predicted_error = predicted_data[i] - actual_data[i]
        sum_of_error += (predicted_error ** 2)
    mean_error = sum_of_error / float(len(actual_data))
    rmse = sqrt(mean_error)
    return rmse


# 8. how good is our algo by using cross validation

def how_good_is_our_algo(dataset, algo, n_folds, *args):
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
        rmse = rmse_method(actual, predicted)
        scores.append(rmse)
    return scores
