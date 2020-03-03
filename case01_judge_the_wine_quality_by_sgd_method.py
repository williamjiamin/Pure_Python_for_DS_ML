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


# print(dataset_list)

# 3.Convert our datatype

def string_to_float_converter(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())
