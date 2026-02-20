# Pure Python for Data Science & Machine Learning
# Author: William
"""Predict sonar data (rock vs. mine) using a perceptron classifier.

Loads the sonar dataset, converts features and class labels,
and evaluates a perceptron trained via stochastic gradient descent
using k-fold cross-validation.
"""

from random import seed
from random import randrange
from csv import reader


# --- Step 1: Load CSV Data ---

def read_csv(filename):
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

def change_str_column_to_int(dataset, column):
    """Convert a column of string class labels to integer codes.

    Args:
        dataset: The dataset (modified in place).
        column: Column index containing class labels.

    Returns:
        A dict mapping original string values to integer codes.
    """
    class_value = [row[column] for row in dataset]
    unique_value = set(class_value)

    search_tool = dict()
    for i, value in enumerate(unique_value):
        search_tool[value] = i

    for row in dataset:
        row[column] = search_tool[row[column]]
    return search_tool


# --- Step 4: K-Fold Cross-Validation Split ---

def k_folds_cross_validation(dataset, n_folds):
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
    fold_size = int(len(dataset) / n_folds)

    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# --- Step 5: Accuracy Metric ---

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


# --- Step 6: Algorithm Evaluation with Cross-Validation ---

def whether_the_algo_is_good_or_not(dataset, algo, n_folds, *args):
    """Evaluate an algorithm using k-fold cross-validation.

    Args:
        dataset: The full dataset.
        algo: The prediction algorithm function.
        n_folds: Number of cross-validation folds.
        *args: Additional arguments passed to the algorithm.

    Returns:
        A list of accuracy scores, one per fold.
    """
    folds = k_folds_cross_validation(dataset, n_folds)
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


# --- Step 7: Perceptron Prediction ---

def predict(row, weights):
    """Make a binary prediction using the perceptron activation rule.

    Args:
        row: A single data row.
        weights: List of weights (bias at index 0).

    Returns:
        1.0 if activation >= 0, else 0.
    """
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0


# --- Step 8: Weight Estimation via SGD ---

def estimate_our_weight_using_sgd_method(training_data, learning_rate, n_epoch):
    """Estimate perceptron weights using stochastic gradient descent.

    Args:
        training_data: The training dataset.
        learning_rate: Step size for weight updates.
        n_epoch: Number of training epochs.

    Returns:
        A list of estimated weights.
    """
    weights = [0.0 for i in range(len(training_data[0]))]
    for epoch in range(n_epoch):
        for row in training_data:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            weights[0] = weights[0] + learning_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + learning_rate * error * row[i]
    return weights


# --- Step 9: Perceptron Algorithm ---

def perceptron(training_data, testing_data, learning_rate, n_epoch):
    """Train a perceptron and predict on test data.

    Args:
        training_data: The training dataset.
        testing_data: The testing dataset.
        learning_rate: Step size for weight updates.
        n_epoch: Number of training epochs.

    Returns:
        A list of predicted class labels.
    """
    predictions = list()
    weights = estimate_our_weight_using_sgd_method(training_data, learning_rate, n_epoch)
    for row in testing_data:
        prediction = predict(row, weights)
        predictions.append(prediction)
    return predictions


# --- Step 10: Run on Real Sonar Dataset ---

if __name__ == '__main__':
    seed(1)

    filename = 'sonar.all-data.csv'
    dataset = read_csv(filename)
    for i in range(len(dataset[0]) - 1):
        change_string_to_float(dataset, i)

    change_str_column_to_int(dataset, len(dataset[0]) - 1)

    n_folds = 3
    learning_rate = 0.01
    n_epoch = 500
    scores = whether_the_algo_is_good_or_not(dataset, perceptron, n_folds, learning_rate, n_epoch)

    print("The score of our model is: %s" % scores)
    print("The average accuracy is: %3.f%%, The baseline is 50%%" % (sum(scores) / float(len(scores))))
