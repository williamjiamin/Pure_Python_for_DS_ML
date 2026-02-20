# Pure Python for Data Science & Machine Learning
# Author: William
"""Banknote authentication using a CART (Classification and Regression Tree).

Loads the banknote authentication dataset, builds a decision tree using
Gini index for splitting, and evaluates with k-fold cross-validation.
Data source: http://archive.ics.uci.edu/ml/datasets/banknote+authentication
"""

from random import seed
from random import randrange
from csv import reader


# --- Step 1: Load CSV Data ---

def csv_loader(filename):
    """Load a CSV file into a list of lists.

    Args:
        filename: Path to the CSV file.

    Returns:
        A list of rows, where each row is a list of string values.
    """
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# --- Step 2: Data Type Conversion ---

def str_to_float_converter(dataset, column):
    """Convert a column of string values to float in place.

    Args:
        dataset: The dataset (list of rows).
        column: Column index to convert.
    """
    for row in dataset:
        row[column] = float(row[column].strip())


# --- Step 3: K-Fold Cross-Validation Split ---

def k_fold_cross_validation_and_split(dataset, n_folds):
    """Split dataset into k folds for cross-validation.

    Args:
        dataset: The dataset to split.
        n_folds: Number of folds.

    Returns:
        A list of folds, each fold is a list of rows.
    """
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


# --- Step 4: Accuracy Metric ---

def calculate_the_accuracy(actual, predicted):
    """Calculate classification accuracy as a percentage.

    Args:
        actual: List of actual class labels.
        predicted: List of predicted class labels.

    Returns:
        Accuracy percentage (float).
    """
    correct_num = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct_num += 1
    return correct_num / float(len(actual)) * 100.0


# --- Step 5: Algorithm Evaluation with K-Fold Cross-Validation ---

def how_good_is_our_algo(dataset, algorithm, n_folds, *args):
    """Evaluate an algorithm using k-fold cross-validation.

    Args:
        dataset: The full dataset.
        algorithm: The prediction algorithm function.
        n_folds: Number of cross-validation folds.
        *args: Additional arguments passed to the algorithm.

    Returns:
        A list of accuracy scores, one per fold.
    """
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


# --- Step 6: Binary Split on a Feature ---

def test_split(index, value, dataset):
    """Split a dataset into left and right groups based on a feature threshold.

    Args:
        index: Feature column index to split on.
        value: Threshold value for the split.
        dataset: The dataset to split.

    Returns:
        A tuple (left, right) of row lists.
    """
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# --- Step 7: Gini Index Calculation ---

def gini_index(groups, classes):
    """Calculate the Gini index for a given split.

    Args:
        groups: A list of groups (left and right), each a list of rows.
        classes: List of unique class values.

    Returns:
        The weighted Gini index (float). Lower is better.
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
        gini += (1.0 - score) * (size / n_instances)
    return gini


# --- Step 8: Find the Best Split ---

def get_split(dataset):
    """Find the best split point for the dataset using Gini index.

    Args:
        dataset: The dataset to find the best split for.

    Returns:
        A dict with 'index', 'value', and 'groups' for the best split.
    """
    class_value = list(set(row[-1] for row in dataset))
    posi_index, posi_value, posi_score, posi_groups = float('inf'), float('inf'), float('inf'), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_value)
            if gini < posi_score:
                posi_index, posi_value, posi_score, posi_groups = index, row[index], gini, groups
    return {'index': posi_index, 'value': posi_value, 'groups': posi_groups}


# --- Step 9: Terminal Node ---

def determine_the_terminal_node(group):
    """Determine the class label for a terminal (leaf) node by majority vote.

    Args:
        group: List of rows in this node.

    Returns:
        The most common class value.
    """
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


# --- Step 10: Recursive Tree Splitting ---

def split(node, max_depth, min_size, depth):
    """Recursively build the decision tree by splitting nodes.

    Args:
        node: Current node dict with 'groups' to split.
        max_depth: Maximum allowed tree depth.
        min_size: Minimum number of samples required to split.
        depth: Current depth in the tree.
    """
    left, right = node['groups']
    del node['groups']

    if not left or not right:
        node['left'] = node['right'] = determine_the_terminal_node(left + right)
        return

    if depth >= max_depth:
        node['left'] = determine_the_terminal_node(left)
        node['right'] = determine_the_terminal_node(right)
        return

    if len(left) <= min_size:
        node['left'] = determine_the_terminal_node(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right'] = determine_the_terminal_node(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)


# --- Step 11: Build the Decision Tree ---

def build_tree(train, max_depth, min_size):
    """Build a decision tree from the training data.

    Args:
        train: Training dataset.
        max_depth: Maximum allowed tree depth.
        min_size: Minimum number of samples to allow a split.

    Returns:
        The root node of the decision tree (dict).
    """
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root


# --- Step 12: Make a Prediction ---

def predict(node, row):
    """Predict the class for a single row by traversing the decision tree.

    Args:
        node: Current tree node (dict or terminal value).
        row: The data row to classify.

    Returns:
        The predicted class label.
    """
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


# --- Step 13: CART Algorithm ---

def decision_tree(train, test, max_depth, min_size):
    """Classification and Regression Tree (CART) algorithm.

    Args:
        train: Training dataset.
        test: Testing dataset.
        max_depth: Maximum tree depth.
        min_size: Minimum samples for a split.

    Returns:
        A list of predicted class labels for the test set.
    """
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions


# --- Step 14: Run on Banknote Authentication Data ---

if __name__ == '__main__':
    seed(1)
    filename = 'data_banknote_authentication.csv'
    dataset = csv_loader(filename)

    for i in range(len(dataset[0])):
        str_to_float_converter(dataset, i)

    n_folds = 10
    max_depth = 10
    min_size = 5
    scores = how_good_is_our_algo(dataset, decision_tree, n_folds, max_depth, min_size)
    print('Score is %s' % scores)
    print('Average Accuracy is: %.3f%%' % (sum(scores) / float(len(scores))))
