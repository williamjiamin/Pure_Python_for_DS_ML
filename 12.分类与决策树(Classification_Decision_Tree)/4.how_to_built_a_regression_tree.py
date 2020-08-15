# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号1.乐学偶得（lexueoude）2.乐学FinTech (LoveShareFinTech)

# 1.root node
# 2.recursive split
# 3.terminal node (为了解决over-fitting的问题，减少整个tree的深度/高度，以及必须规定最小切分单位)
# 4.finish building the tree

def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


def calculate_the_gini_index(groups, classes):
    # 计算有多少实例
    n_instances = float(sum([len(group) for group in groups]))
    # 把每一个group里面的加权gini计算出来
    gini = 0.0
    for group in groups:
        size = float(len(group))
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


def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    posi_index, posi_value, posi_score, posi_groups = 888, 888, 888, None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = calculate_the_gini_index(groups, class_values)
            print("X%d < %.3f Gini=%.3f" % ((index + 1), row[index], gini))
            if gini < posi_score:
                posi_index, posi_value, posi_score, posi_groups = index, row[index], gini, groups
    return {'index': posi_index, 'value': posi_value, 'groups': posi_groups}


def determine_the_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# 1.把数据进行切分（分为左边与右边），原数据删除掉
# 2.检查非空以及满足我们的我们设置的条件（深度/最小切分单位/非空）
# 3.一直重复类似寻找root node的操作，一直到最末端


def split(node, max_depth, min_size, depth):
    # 做切分，并删除掉原数据
    left, right = node['groups']
    del (node['groups'])
    # 查看非空
    if not left or not right:
        node['left'] = node['right'] = determine_the_terminal(left + right)
        return

    # 检查最大深度是否超过
    if depth >= max_depth:
        node['left'], node['right'] = determine_the_terminal(left), determine_the_terminal(right)
        return
        # 最小分类判断与左侧继续向下分类
    if len(left) <= min_size:
        node['left'] = determine_the_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)
        # 最小分类判断与右侧继续向下分类
    if len(right) <= min_size:
        node['right'] = determine_the_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# 最终建立决策树

def build_the_regression_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# 通过CLI可视化的呈现类树状结构便于感性认知
def print_our_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth * '-', (node['index'] + 1), node['value'])))
        print_our_tree(node['left'], depth + 1)
        print_our_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * '-', node)))


def make_prediction(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return make_prediction(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return make_prediction(node['right'], row)
        else:
            return node['right']


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

tree = build_the_regression_tree(dataset, 3, 1)
print_our_tree(tree)

decision_tree_stump = {'index': 0, 'right': 1, 'value': 9.3, 'left': 0}
for row in dataset:
    prediction = make_prediction(decision_tree_stump, row)
    print("What is expected data : %d , Your prediction is %d " % (row[-1], prediction))
