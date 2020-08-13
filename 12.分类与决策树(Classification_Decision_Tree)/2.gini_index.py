# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号1.乐学偶得（lexueoude）2.乐学FinTech (LoveShareFinTech)

def calculate_the_gini_index(groups, classes):
    # 计算有多少实例
    n_instances = float(sum([len(groups) for group in groups]))
    # 把每一个group里面的加权gini计算出来
    gini = 0.0
    for group in groups:
        size = float(len(groups))
        # *注意，这里不能除以0，所以我们要考虑到分母为0的情况
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # 这个做了一个加权处理
        gini += (1 - score) * (size / n_instances)

    return gini


worst_case_for_two_classes = [[[1, 1], [1, 0]], [[1, 1], [1, 0]]]

print(calculate_the_gini_index(worst_case_for_two_classes, [0, 1]))

best_case_for_two_classes = [[[1, 0], [1, 0]], [[1, 1], [1, 1]]]

print(calculate_the_gini_index(best_case_for_two_classes, [0, 1]))


def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# index,value,groups数据较多，所以选用dict
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    posi_index, posi_value, posi_score, posi_groups = 888, 888, 888, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = calculate_the_gini_index(groups, class_values)
            if gini < posi_score:
                posi_index, posi_value, posi_score, posi_groups = index, row[index], gini, groups
    return {'The Best Index is': posi_index, 'The Best Value is': posi_value, 'The Best Groups is': posi_groups}


dataset = [[2.1, 1.1, 0],
           [3.4, 2.5, 0],
           [1.3, 5.8, 0],
           [1.9, 8.6, 0],
           [3.7, 6.2, 0],
           [8.8, 1.1, 1],
           [9.6, 3.4, 1],
           [10.2, 7.4, 1],
           [7.7, 8.8, 1],
           [9.7, 6.9, 1]]

split = get_split(dataset)
print(split)
