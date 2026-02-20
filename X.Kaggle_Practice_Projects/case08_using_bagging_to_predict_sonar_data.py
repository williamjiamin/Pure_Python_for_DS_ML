# Pure Python for Data Science & Machine Learning
# Author: William
"""Predict sonar data using bagged decision trees.

Loads the sonar dataset, builds multiple CART decision trees on
bootstrap subsamples (bagging), and evaluates with k-fold
cross-validation. Tests different numbers of trees.
"""

from random import seed
from random import randrange
from random import random
from csv import reader


# --- Step 1: Load CSV Data ---

def load_csv(filename):
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

def str_to_float(dataset, column):
    """Convert a column of string values to float in place.

    Args:
        dataset: The dataset (list of rows).
        column: Column index to convert.
    """
    for row in dataset:
        row[column] = float(row[column].strip())


def str_to_int(dataset, column):
    """Convert a column of string class labels to integer codes.

    Args:
        dataset: The dataset (modified in place).
        column: Column index containing class labels.

    Returns:
        A dict mapping original string values to integer codes.
    """
    class_value = [row[column] for row in dataset]
    unique = set(class_value)
    look_up = dict()
    for i, value in enumerate(unique):
        look_up[value] = i
    for row in dataset:
        row[column] = look_up[row[column]]
    return look_up


# --- Step 3: K-Fold Cross-Validation Split ---

def cross_validation_split(dataset, n_folds):
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

def calculate_accuracy(actual, predicted):
    """Calculate classification accuracy as a percentage.

    Args:
        actual: List of actual class labels.
        predicted: List of predicted class labels.

    Returns:
        Accuracy percentage (float).
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# --- Step 5: Algorithm Evaluation with Cross-Validation ---

def evaluate_our_algo(dataset, algo, n_folds, *args):
    """Evaluate an algorithm using k-fold cross-validation.

    Args:
        dataset: The full dataset.
        algo: The prediction algorithm function.
        n_folds: Number of cross-validation folds.
        *args: Additional arguments passed to the algorithm.

    Returns:
        A list of accuracy scores, one per fold.
    """
    folds = cross_validation_split(dataset, n_folds)
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
        accuracy = calculate_accuracy(actual, predicted)
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
        gini += (1 - score) * (size / n_instances)
    return gini


# --- Step 8: Find the Best Split ---

def get_split(dataset):
    """Find the best split point for the dataset using Gini index.

    Args:
        dataset: The dataset to find the best split for.

    Returns:
        A dict with 'index', 'value', and 'groups' for the best split.
    """
    class_values = list(set(row[-1] for row in dataset))
    posi_index, posi_value, posi_score, posi_groups = float('inf'), float('inf'), float('inf'), None
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < posi_score:
                posi_index, posi_value, posi_score, posi_groups = index, row[index], gini, groups
    return {'index': posi_index, 'value': posi_value, 'groups': posi_groups}


# --- Step 9: Terminal Node ---

def determine_the_terminal(group):
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
        node['left'] = node['right'] = determine_the_terminal(left + right)
        return

    if depth >= max_depth:
        node['left'] = determine_the_terminal(left)
        node['right'] = determine_the_terminal(right)
        return

    if len(left) <= min_size:
        node['left'] = determine_the_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)

    if len(right) <= min_size:
        node['right'] = determine_the_terminal(right)
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


# --- Step 13: Bootstrap Subsample ---

def subsample(dataset, ratio):
    """Create a bootstrap subsample from the dataset with replacement.

    Args:
        dataset: The original dataset.
        ratio: Fraction of the dataset to sample.

    Returns:
        A list containing the sampled rows.
    """
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


# --- Step 14: Bagging Prediction ---

def bagging_predict(trees, row):
    """Predict a class label by majority vote across all trees.

    Args:
        trees: List of trained decision trees.
        row: The data row to classify.

    Returns:
        The most common prediction across all trees.
    """
    predictions = [predict(tree, row) for tree in trees]
    return max(set(predictions), key=predictions.count)


# --- Step 15: Bagging Algorithm ---

def bagging(train, test, max_depth, min_size, sample_size, n_trees):
    """Bagged decision trees algorithm.

    Builds multiple trees on bootstrap subsamples and aggregates predictions.

    Args:
        train: Training dataset.
        test: Testing dataset.
        max_depth: Maximum tree depth.
        min_size: Minimum samples for a split.
        sample_size: Bootstrap sample ratio.
        n_trees: Number of trees to build.

    Returns:
        A list of predicted class labels for the test set.
    """
    trees = list()
    for i in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return predictions


# --- Step 16: Run on Sonar Dataset ---

if __name__ == '__main__':
    seed(1)
    dataset = load_csv('sonar.all-data.csv')

    for i in range(len(dataset[0]) - 1):
        str_to_float(dataset, i)

    str_to_int(dataset, len(dataset[0]) - 1)

    n_folds = 5
    max_depth = 6
    min_size = 2
    sample_size = 0.5

    for n_trees in [1, 5, 10, 50]:
        scores = evaluate_our_algo(dataset, bagging, n_folds, max_depth, min_size, sample_size, n_trees)
        print('We are using [%d] trees' % n_trees)
        print('The scores are: [%s]' % scores)
        print('The mean accuracy is [%.3f]' % (sum(scores) / float(len(scores))))
