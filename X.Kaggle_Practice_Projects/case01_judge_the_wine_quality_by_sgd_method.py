# Pure Python for Data Science & Machine Learning
# Author: William
"""Predict wine quality using stochastic gradient descent (SGD) linear regression.

Loads the white wine quality dataset, normalizes features, and evaluates
a linear regression model trained via SGD using k-fold cross-validation
with RMSE as the metric.
"""

from csv import reader
from math import sqrt
from random import randrange
from random import seed


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

def string_to_float_converter(dataset, column):
    """Convert a column of string values to float in place.

    Args:
        dataset: The dataset (list of rows).
        column: Column index to convert.
    """
    for row in dataset:
        row[column] = float(row[column].strip())


# --- Step 3: Min-Max Calculation ---

def find_the_min_and_max_of_our_dataset(dataset):
    """Find the minimum and maximum value for each column.

    Args:
        dataset: The dataset (list of rows).

    Returns:
        A list of [min, max] pairs, one per column.
    """
    min_max_list = list()
    for i in range(len(dataset[0])):
        col_value = [row[i] for row in dataset]
        max_value = max(col_value)
        min_value = min(col_value)
        min_max_list.append([min_value, max_value])
    return min_max_list


# --- Step 4: Data Normalization ---

def normalization(dataset, min_max_list):
    """Normalize dataset values to the range [0, 1] using min-max scaling.

    Args:
        dataset: The dataset (modified in place).
        min_max_list: List of [min, max] pairs per column.
    """
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - min_max_list[i][0]) / (min_max_list[i][1] - min_max_list[i][0])


# --- Step 5: K-Fold Cross-Validation Split ---

def k_fold_cross_validation_split(dataset, n_folds):
    """Split dataset into k folds for cross-validation.

    Args:
        dataset: The dataset to split.
        n_folds: Number of folds.

    Returns:
        A list of folds, each fold is a list of rows.
    """
    split_dataset = list()
    copy_dataset = list(dataset)
    every_fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < every_fold_size:
            index = randrange(len(copy_dataset))
            fold.append(copy_dataset.pop(index))
        split_dataset.append(fold)
    return split_dataset


# --- Step 6: RMSE Evaluation Metric ---

def rmse_method(actual_data, predicted_data):
    """Calculate root mean squared error between actual and predicted values.

    Args:
        actual_data: List of actual target values.
        predicted_data: List of predicted values.

    Returns:
        The RMSE value (float).
    """
    sum_of_error = 0.0
    for i in range(len(actual_data)):
        predicted_error = predicted_data[i] - actual_data[i]
        sum_of_error += predicted_error ** 2
    mean_error = sum_of_error / float(len(actual_data))
    return sqrt(mean_error)


# --- Step 7: Algorithm Evaluation with Cross-Validation ---

def how_good_is_our_algo(dataset, algo, n_folds, *args):
    """Evaluate an algorithm using k-fold cross-validation and RMSE.

    Args:
        dataset: The full dataset.
        algo: The prediction algorithm function.
        n_folds: Number of cross-validation folds.
        *args: Additional arguments passed to the algorithm.

    Returns:
        A list of RMSE scores, one per fold.
    """
    folds = k_fold_cross_validation_split(dataset, n_folds)
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
        rmse = rmse_method(actual, predicted)
        scores.append(rmse)
    return scores


# --- Step 8: Make Prediction ---

def predict(row, coefficients):
    """Predict a target value using linear coefficients.

    Args:
        row: A single data row.
        coefficients: List of coefficients (intercept at index 0).

    Returns:
        The predicted value (float).
    """
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return yhat


# --- Step 9: SGD Coefficient Estimation ---

def sgd_method_to_calculate_coefficient(training_data, learning_rate, n_epoch):
    """Estimate linear regression coefficients using stochastic gradient descent.

    Args:
        training_data: The training dataset.
        learning_rate: Step size for weight updates.
        n_epoch: Number of training epochs.

    Returns:
        A list of estimated coefficients.
    """
    coefficients_list = [0.0 for i in range(len(training_data[0]))]
    for epoch in range(n_epoch):
        for row in training_data:
            yhat = predict(row, coefficients_list)
            error = yhat - row[-1]
            coefficients_list[0] = coefficients_list[0] - learning_rate * error
            for i in range(len(row) - 1):
                coefficients_list[i + 1] = coefficients_list[i + 1] - learning_rate * error * row[i]
    return coefficients_list


# --- Step 10: SGD Linear Regression ---

def using_sgd_method_to_calculate_linear_regression(training_data, testing_data, learning_rate, n_epoch):
    """Run linear regression using SGD-estimated coefficients.

    Args:
        training_data: The training dataset.
        testing_data: The testing dataset.
        learning_rate: Step size for weight updates.
        n_epoch: Number of training epochs.

    Returns:
        A list of predicted values for the testing data.
    """
    predictions = list()
    coefficients_list = sgd_method_to_calculate_coefficient(training_data, learning_rate, n_epoch)
    for row in testing_data:
        yhat = predict(row, coefficients_list)
        predictions.append(yhat)
    return predictions


# --- Step 11: Run on Real Data ---

if __name__ == '__main__':
    seed(1)
    wine_quality_data_name = 'winequality-white.csv'
    dataset = csv_loader(wine_quality_data_name)
    for i in range(len(dataset[0])):
        string_to_float_converter(dataset, i)

    min_and_max = find_the_min_and_max_of_our_dataset(dataset)
    normalization(dataset, min_and_max)

    n_folds = 5
    learning_rate = 0.1
    n_epoch = 50

    algo_score = how_good_is_our_algo(
        dataset, using_sgd_method_to_calculate_linear_regression,
        n_folds, learning_rate, n_epoch
    )

    print("Our algo's score is %s" % algo_score)
    print("The mean of our algo's RMSE is %.3f" % (sum(algo_score) / float(len(algo_score))))
