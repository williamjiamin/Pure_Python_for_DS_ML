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


# print(csv_loader('data_banknote_authentication.csv'))


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

def how_good_is_our_algo(dataset, algorithm, n_folds, *args):
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
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = calculate_the_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


# 7. 把数据切分成左边与右边（split the data to left and right）

def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# 8.计算gini index
def gini_index(groups, classes):
    # 计算有多少个实例（有多少个样本在split的）
    n_instances = float(sum([len(group) for group in groups]))
    # 计算每一组（group）的gini
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # 避免size为0的情况
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # 计算gini并做加权调整
        gini += (1.0 - score) * (size / n_instances)
    return gini


# 9.利用gini index算出最好的split
def get_split(dataset):
    class_value = list(set(row[-1] for row in dataset))
    posi_index, posi_value, posi_score, posi_groups = 888, 888, 888, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_value)
            if gini < posi_score:
                posi_index, posi_value, posi_score, posi_groups = index, row[index], gini, groups
    return {'index': posi_index, 'value': posi_value, 'groups': posi_groups}


# 10.直接到决策树末端结束

def determine_the_terminal_node(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# 11.去创建决策树

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del (node['groups'])
    # 查看是否已经没有切分了
    if not left or not right:
        node['left'] = node['right'] = determine_the_terminal_node(left + right)
        return
    # 查看是否目前的分类深度超过最大深度
    if depth >= max_depth:
        node['left'], node['right'] = determine_the_terminal_node(left), determine_the_terminal_node(right)
        return
    # 左边的分类
    if len(left) <= min_size:
        node['left'] = determine_the_terminal_node(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
    # 右边的分类
    if len(right) <= min_size:
        node['right'] = determine_the_terminal_node(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# 12.形成tree

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# 13.实现预测功能

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# 14.Classification And Regression Tree
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions


# 16.实际数据演练:数据读取与预处理
seed(1)
filename = 'data_banknote_authentication.csv'
dataset = csv_loader(filename)

for i in range(len(dataset[0])):
    str_to_float_converter(dataset, i)

# print(dataset)

n_folds = 10
max_depth = 10
min_size = 5
scores = how_good_is_our_algo(dataset, decision_tree, n_folds, max_depth, min_size)
print('Score is %s' % scores)
print('Average Accuracy is : %.3f%%' % (sum(scores) / float(len(scores))))
