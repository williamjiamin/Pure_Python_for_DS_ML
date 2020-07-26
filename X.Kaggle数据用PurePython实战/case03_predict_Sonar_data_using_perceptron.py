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


# 4.change string column to int
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
