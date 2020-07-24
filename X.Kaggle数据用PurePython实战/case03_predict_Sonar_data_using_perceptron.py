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


filename = 'sonar.all-data.csv'
dataset = read_csv(filename)
for i in range(len(dataset[0]) - 1):
    change_string_to_float(dataset, i)
print(dataset)
