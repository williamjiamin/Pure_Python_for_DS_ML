# Pure Python for Data Science & Machine Learning
# Author: William
"""Predict abalone age using K-Nearest Neighbors (KNN).

Loads the abalone dataset, normalizes features, and evaluates
a KNN classifier using k-fold cross-validation with accuracy
as the metric.
"""

from random import seed
from random import randrange
from csv import reader
from math import sqrt


# --- Step 1: Load CSV Data ---

def read_our_csv_file(filename):
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


# --- Step 2: Data Type Conversion (String to Float) ---

def change_string_to_float(dataset, column):
    """Convert a column of string values to float in place.

    Args:
        dataset: The dataset (list of rows).
        column: Column index to convert.
    """
    for row in dataset:
        row[column] = float(row[column].strip())


# --- Step 3: Class Label Conversion (String to Int) ---

def change_string_to_int(dataset, column):
    """Convert a column of string class labels to integer codes.

    Args:
        dataset: The dataset (modified in place).
        column: Column index containing class labels.

    Returns:
        A dict mapping original string values to integer codes.
    """
    class_value = [row[column] for row in dataset]
    find_the_unique_class = set(class_value)
    lookup = dict()
    for i, value in enumerate(find_the_unique_class):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# --- Step 4: Min-Max Normalization ---

def find_the_min_and_max_of_our_data(dataset):
    """Find the minimum and maximum value for each column.

    Args:
        dataset: The dataset (list of rows).

    Returns:
        A list of [min, max] pairs, one per column.
    """
    min_and_max_list = list()
    for i in range(len(dataset[0])):
        column_value = [row[i] for row in dataset]
        the_min_value = min(column_value)
        the_max_value = max(column_value)
        min_and_max_list.append([the_min_value, the_max_value])
    return min_and_max_list


def normalize_our_data(dataset, min_and_max_list):
    """Normalize dataset values to the range [0, 1] using min-max scaling.

    Args:
        dataset: The dataset (modified in place).
        min_and_max_list: List of [min, max] pairs per column.
    """
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - min_and_max_list[i][0]) / (min_and_max_list[i][1] - min_and_max_list[i][0])


# --- Step 5: K-Fold Cross-Validation Split ---

def k_fold_cross_validation(dataset, n_folds):
    """Split dataset into k folds for cross-validation.

    Works on a copy to preserve the original data.

    Args:
        dataset: The dataset to split.
        n_folds: Number of folds.

    Returns:
        A list of folds, each fold is a list of rows.
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    every_fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < every_fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# --- Step 6: Accuracy Metric ---

def calculate_our_model_accuracy(actual, predicted):
    """Calculate classification accuracy as a percentage.

    Args:
        actual: List of actual class labels.
        predicted: List of predicted class labels.

    Returns:
        Accuracy percentage (float).
    """
    correct_counter = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct_counter += 1
    return correct_counter / float(len(actual)) * 100.0


# --- Step 7: Algorithm Evaluation with Cross-Validation ---

def how_good_is_our_algo(dataset, algo, n_folds, *args):
    """Evaluate an algorithm using k-fold cross-validation.

    Args:
        dataset: The full dataset.
        algo: The prediction algorithm function.
        n_folds: Number of cross-validation folds.
        *args: Additional arguments passed to the algorithm.

    Returns:
        A list of accuracy scores, one per fold.
    """
    folds = k_fold_cross_validation(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_dataset = list(folds)
        train_dataset.remove(fold)
        train_dataset = sum(train_dataset, [])
        test_dataset = list()
        for row in fold:
            row_copy = list(row)
            test_dataset.append(row_copy)
            row_copy[-1] = None
        predicted = algo(train_dataset, test_dataset, *args)
        actual = [row[-1] for row in fold]
        accuracy = calculate_our_model_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


# --- Step 8: Euclidean Distance ---

def calculate_euclidean_distance(row1, row2):
    """Calculate the Euclidean distance between two rows.

    Args:
        row1: First data row.
        row2: Second data row.

    Returns:
        Euclidean distance (float), ignoring the last column (label).
    """
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# --- Step 9: Find K Nearest Neighbors ---

def get_our_neighbors(train_dataset, test_row, num_of_neighbors):
    """Find the k nearest neighbors of a test row.

    Args:
        train_dataset: The training dataset.
        test_row: The test row to find neighbors for.
        num_of_neighbors: Number of neighbors to return.

    Returns:
        A list of the k closest training rows.
    """
    distances = list()
    for train_dataset_row in train_dataset:
        dist = calculate_euclidean_distance(test_row, train_dataset_row)
        distances.append((train_dataset_row, dist))
    distances.sort(key=lambda every_tuple: every_tuple[1])
    neighbors = list()
    for i in range(num_of_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# --- Step 10: Make a Single Prediction ---

def make_prediction(train_dataset, test_row, num_of_neighbors):
    """Predict the class for a single test row via majority vote of neighbors.

    Args:
        train_dataset: The training dataset.
        test_row: The test row to classify.
        num_of_neighbors: Number of neighbors to consider.

    Returns:
        The predicted class label.
    """
    neighbors = get_our_neighbors(train_dataset, test_row, num_of_neighbors)
    output = [row[-1] for row in neighbors]
    return max(set(output), key=output.count)


# --- Step 11: KNN Algorithm ---

def get_our_prediction_using_knn_algo(train_dataset, test_dataset, num_of_neighbors):
    """Run KNN classification on the test dataset.

    Args:
        train_dataset: The training dataset.
        test_dataset: The testing dataset.
        num_of_neighbors: Number of neighbors (K).

    Returns:
        A list of predicted class labels.
    """
    predictions = list()
    for test_row in test_dataset:
        our_prediction = make_prediction(train_dataset, test_row, num_of_neighbors)
        predictions.append(our_prediction)
    return predictions


# --- Step 12: Run on Abalone Dataset ---

if __name__ == '__main__':
    seed(1)
    dataset = read_our_csv_file('abalone.csv')
    for i in range(1, len(dataset[0])):
        change_string_to_float(dataset, i)

    change_string_to_int(dataset, 0)

    n_folds = 10
    num_neighbors = 7
    scores = how_good_is_our_algo(dataset, get_our_prediction_using_knn_algo, n_folds, num_neighbors)

    print("Our model's scores are: %s" % scores)
    print("The mean accuracy is: %.3f%%" % (sum(scores) / float(len(scores))))
