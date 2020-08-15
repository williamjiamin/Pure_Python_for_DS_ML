# Created by william from lexueoude.com. 更多正版技术视频讲解，
# 公众号1.乐学偶得（lexueoude）2.乐学FinTech (LoveShareFinTech)

# 1. 导入相关库
# http://archive.ics.uci.edu/ml/datasets/banknote+authentication
from random import seed
from random import randrange
from csv import reader


# 2.读取数据
def csv_loader(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


print(csv_loader('data_banknote_authentication.csv'))


# 3.数据类型转换
def str_to_float_converter(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# 4. 交叉检验

def k_fold_cross_validation_and_split(dataset, n_folds):
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


# 5.计算模型的准确性

def calculate_the_accuracy(actual, predicted):
    correct_num = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct_num += 1
    return correct_num / float(len(actual)) * 100.0


# 6.通过应用k-fold交叉检验检验模型的准确性

def how_good_is_our_algo(dataset, algo, n_folds, *args):
    folds = k_fold_cross_validation_and_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            # 为了让数据"纯净"，把真正的结果直接删除掉（但是注意实在row_copy中删除的，并不影响原数据）
            row_copy[-1] = None
        predicted = algo(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = calculate_the_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores
