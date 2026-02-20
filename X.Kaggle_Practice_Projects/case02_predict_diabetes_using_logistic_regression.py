# Pure Python for Data Science & Machine Learning
# Author: William
"""Predict diabetes using logistic regression with SGD optimization.

Loads the Pima Indians diabetes dataset, rescales features to [0, 1],
and evaluates a logistic regression model trained via stochastic gradient
descent using k-fold cross-validation with accuracy as the metric.
"""

from random import seed
from random import randrange
from csv import reader
from math import exp


# --- Step 1: Load CSV Data ---

def load_data_from_csv_file(file_name):
    """Load a CSV file into a list of lists.

    Args:
        file_name: Path to the CSV file.

    Returns:
        A list of rows, where each row is a list of string values.
    """
    dataset = list()
    with open(file_name, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# --- Step 2: Data Type Conversion ---

def change_string_to_float(dataset, column):
    """Convert a column of string values to float in place.

    Strips leading/trailing whitespace before conversion.

    Args:
        dataset: The dataset (list of rows).
        column: Column index to convert.
    """
    for row in dataset:
        row[column] = float(row[column].strip())


# --- Step 3: Min-Max Calculation ---

def find_the_min_and_max_of_our_data(dataset):
    """Find the minimum and maximum value for each column.

    Args:
        dataset: The dataset (list of rows).

    Returns:
        A list of [min, max] pairs, one per column.
    """
    min_max_list = list()
    for i in range(len(dataset[0])):
        values_in_every_column = [row[i] for row in dataset]
        the_min_value = min(values_in_every_column)
        the_max_value = max(values_in_every_column)
        min_max_list.append([the_min_value, the_max_value])
    return min_max_list


# --- Step 4: Data Rescaling ---

def rescale_our_data(dataset, min_max_list):
    """Rescale dataset values to the range [0, 1] using min-max scaling.

    Args:
        dataset: The dataset (modified in place).
        min_max_list: List of [min, max] pairs per column.
    """
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - min_max_list[i][0]) / (min_max_list[i][1] - min_max_list[i][0])


# --- Step 5: K-Fold Cross-Validation Split ---

def k_fold_cross_validation(dataset, n_folds):
    """Split dataset into k folds for cross-validation.

    Works on a copy of the dataset to preserve the original data.

    Args:
        dataset: The dataset to split.
        n_folds: Number of folds.

    Returns:
        A list of folds, each fold is a list of rows.
    """
    split_data = list()
    copy_dataset = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        box_for_my_fold = list()
        while len(box_for_my_fold) < fold_size:
            some_random_index = randrange(len(copy_dataset))
            box_for_my_fold.append(copy_dataset.pop(some_random_index))
        split_data.append(box_for_my_fold)
    return split_data


# --- Step 6: Accuracy Metric ---

def calculate_the_accuracy_of_our_model(actual_data, predicted_data):
    """Calculate classification accuracy as a percentage.

    Args:
        actual_data: List of actual class labels.
        predicted_data: List of predicted class labels.

    Returns:
        Accuracy percentage (float).
    """
    counter_of_correct_prediction = 0
    for i in range(len(actual_data)):
        if actual_data[i] == predicted_data[i]:
            counter_of_correct_prediction += 1
    return counter_of_correct_prediction / float(len(actual_data)) * 100.0


# --- Step 7: Algorithm Evaluation with Cross-Validation ---

def how_good_is_our_algo(dataset, algo, n_folds, *args):
    """Evaluate an algorithm using k-fold cross-validation.

    Strips target values from the test set to prevent data leakage.

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
        training_data_set = list(folds)
        training_data_set.remove(fold)
        training_data_set = sum(training_data_set, [])
        testing_data_set = list()

        for row in fold:
            row_copy = list(row)
            testing_data_set.append(row_copy)
            row_copy[-1] = None
        predicted = algo(training_data_set, testing_data_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = calculate_the_accuracy_of_our_model(actual, predicted)
        scores.append(accuracy)
    return scores


# --- Step 8: Logistic Prediction ---

def prediction(row, coefficients):
    """Predict probability using the logistic (sigmoid) function.

    Args:
        row: A single data row.
        coefficients: List of coefficients (intercept at index 0).

    Returns:
        Predicted probability in the range (0, 1).
    """
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return 1 / (1.0 + exp(-yhat))


# --- Step 9: SGD Coefficient Estimation for Logistic Regression ---

def estimate_coef_lr_using_sgd_method(training_data, learning_rate, n_epochs):
    """Estimate logistic regression coefficients using stochastic gradient descent.

    Args:
        training_data: The training dataset.
        learning_rate: Step size for weight updates.
        n_epochs: Number of training epochs.

    Returns:
        A list of estimated coefficients.
    """
    coef = [0.0 for i in range(len(training_data[0]))]
    for epoch in range(n_epochs):
        for row in training_data:
            yhat = prediction(row, coef)
            error = row[-1] - yhat
            coef[0] = coef[0] + learning_rate * error * yhat * (1.0 - yhat)
            for i in range(len(row) - 1):
                coef[i + 1] = coef[i + 1] + learning_rate * error * yhat * (1.0 - yhat) * row[i]
    return coef


# --- Step 10: Logistic Regression ---

def logistic_regression(training_data, testing_data, learning_rate, n_epochs):
    """Run logistic regression: train via SGD, then predict on test data.

    Args:
        training_data: The training dataset.
        testing_data: The testing dataset.
        learning_rate: Step size for weight updates.
        n_epochs: Number of training epochs.

    Returns:
        A list of predicted class labels (0 or 1).
    """
    predictions = list()
    coef = estimate_coef_lr_using_sgd_method(training_data, learning_rate, n_epochs)
    for row in testing_data:
        yhat = prediction(row, coef)
        yhat = round(yhat)
        predictions.append(yhat)
    return predictions


# --- Step 11: Run on Kaggle Diabetes Dataset ---

if __name__ == '__main__':
    seed(1)

    dataset = load_data_from_csv_file('diabetes.csv')

    for i in range(len(dataset[0])):
        change_string_to_float(dataset, i)

    min_max_value = find_the_min_and_max_of_our_data(dataset)
    rescale_our_data(dataset, min_max_value)

    n_folds = 10
    learning_rate = 0.1
    n_epochs = 1000

    scores = how_good_is_our_algo(dataset, logistic_regression, n_folds, learning_rate, n_epochs)

    print("The scores of our model are %s" % scores)
    print("The average accuracy of our model is %.3f" % (sum(scores) / float(len(scores))))
