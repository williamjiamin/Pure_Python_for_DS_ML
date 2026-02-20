# Pure Python for Data Science & Machine Learning
# Author: William
"""Build a CART classification tree from scratch using the Gini index.

Covers the full pipeline:
  1. Binary split helper
  2. Gini impurity calculation
  3. Greedy best-split search
  4. Terminal node (majority vote)
  5. Recursive splitting with max-depth / min-size stopping criteria
  6. Tree construction, printing, and prediction
"""


# --- Step 1: Binary split helper -------------------------------------------

def test_split(index, value, dataset):
    """Split a dataset into left/right groups on a feature threshold.

    Parameters
    ----------
    index : int
        Feature column index to split on.
    value : float
        Threshold; rows with feature < value go left.
    dataset : list[list[float]]
        The dataset to split.

    Returns
    -------
    tuple[list, list]
        (left_group, right_group).
    """
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# --- Step 2: Gini index calculation ----------------------------------------

def calculate_gini_index(groups, classes):
    """Compute the weighted Gini impurity for a candidate split.

    Parameters
    ----------
    groups : tuple[list, list]
        Two groups (left, right) produced by a binary split.
    classes : list
        Unique class labels present in the dataset.

    Returns
    -------
    float
        Weighted Gini impurity (0.0 = pure).
    """
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1 - score) * (size / n_instances)
    return gini


# --- Step 3: Greedy best-split search --------------------------------------

def get_best_split(dataset):
    """Find the best split point by exhaustive Gini-based search.

    Parameters
    ----------
    dataset : list[list[float]]
        Rows of [feature_1, ..., feature_n, class_label].

    Returns
    -------
    dict
        Keys: 'index', 'value', 'groups'.
    """
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = (
        None, None, float('inf'), None
    )
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = calculate_gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = (
                    index, row[index], gini, groups
                )
    return {'index': best_index, 'value': best_value, 'groups': best_groups}


# --- Step 4: Terminal node --------------------------------------------------

def determine_terminal(group):
    """Return the majority class label in a group (leaf node prediction).

    Parameters
    ----------
    group : list[list[float]]
        Data rows belonging to one node.

    Returns
    -------
    int or float
        The most common class label in the group.
    """
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# --- Step 5: Recursive splitting -------------------------------------------

def recursive_split(node, max_depth, min_size, depth):
    """Recursively split tree nodes until stopping criteria are met.

    Stopping criteria: empty child, max depth reached, or child size
    is at or below min_size.

    Parameters
    ----------
    node : dict
        Current node containing 'groups', 'index', 'value'.
    max_depth : int
        Maximum allowed tree depth.
    min_size : int
        Minimum number of samples required to split a node further.
    depth : int
        Current depth in the tree.
    """
    left, right = node['groups']
    del node['groups']

    if not left or not right:
        node['left'] = node['right'] = determine_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'] = determine_terminal(left)
        node['right'] = determine_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = determine_terminal(left)
    else:
        node['left'] = get_best_split(left)
        recursive_split(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right'] = determine_terminal(right)
    else:
        node['right'] = get_best_split(right)
        recursive_split(node['right'], max_depth, min_size, depth + 1)


# --- Step 6: Build the classification tree ---------------------------------

def build_classification_tree(train, max_depth, min_size):
    """Build a CART classification tree.

    Parameters
    ----------
    train : list[list[float]]
        Training data rows of [feature_1, ..., feature_n, class_label].
    max_depth : int
        Maximum tree depth.
    min_size : int
        Minimum samples per node to allow further splitting.

    Returns
    -------
    dict
        The root node of the trained decision tree.
    """
    root = get_best_split(train)
    recursive_split(root, max_depth, min_size, 1)
    return root


# --- Step 7: Print the tree ------------------------------------------------

def print_tree(node, depth=0):
    """Print a text representation of the decision tree.

    Parameters
    ----------
    node : dict or int/float
        A tree node (dict) or a terminal leaf value.
    depth : int
        Current depth used for indentation.
    """
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % (depth * '-', node['index'] + 1, node['value']))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % (depth * '-', node))


# --- Step 8: Prediction ----------------------------------------------------

def make_prediction(node, row):
    """Predict the class label for a single row using the trained tree.

    Parameters
    ----------
    node : dict
        The root (or sub-root) of a decision tree.
    row : list[float]
        A data row to classify.

    Returns
    -------
    int or float
        The predicted class label.
    """
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


# --- Main execution --------------------------------------------------------

if __name__ == '__main__':
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

    tree = build_classification_tree(dataset, 3, 1)
    print_tree(tree)

    decision_tree_stump = {'index': 0, 'right': 1, 'value': 9.3, 'left': 0}
    for row in dataset:
        pred = make_prediction(decision_tree_stump, row)
        print("Expected: %d, Predicted: %d" % (row[-1], pred))
