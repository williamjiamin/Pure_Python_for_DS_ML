# Pure Python for Data Science & Machine Learning
# Author: William
"""Predict insurance charges using simple linear regression.

Implements simple linear regression from scratch, evaluates
the model with a train/test split, and reports RMSE.
"""

from csv import reader
from math import sqrt
from random import randrange, seed


# --- Step 1: Load CSV Data ---

def load_csv(data_file):
    """Load a CSV file into a list of lists.

    Args:
        data_file: Path to the CSV file.

    Returns:
        A list of rows, where each row is a list of string values.
    """
    data_set = list()
    with open(data_file, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data_set.append(row)
    return data_set


# --- Step 2: Data Type Conversion ---

def string_converter(data_set, column):
    """Convert a column of string values to float in place.

    Strips leading/trailing whitespace before conversion.

    Args:
        data_set: The dataset (list of rows).
        column: Column index to convert.
    """
    for row in data_set:
        row[column] = float(row[column].strip())


# --- Step 3: RMSE Evaluation Metric ---

def calculate_RMSE(actual_data, predicted_data):
    """Calculate root mean squared error between actual and predicted values.

    Args:
        actual_data: List of actual target values.
        predicted_data: List of predicted values.

    Returns:
        The RMSE value (float).
    """
    sum_error = 0.0
    for i in range(len(actual_data)):
        predicted_error = predicted_data[i] - actual_data[i]
        sum_error += predicted_error ** 2
    mean_error = sum_error / float(len(actual_data))
    return sqrt(mean_error)


# --- Step 4: Train/Test Split ---

def train_test_split(data_set, split):
    """Randomly split dataset into training and testing sets.

    Args:
        data_set: The full dataset.
        split: Fraction of data to use for training (0 to 1).

    Returns:
        A tuple of (train_set, test_set).
    """
    train = list()
    train_size = split * len(data_set)
    data_set_copy = list(data_set)
    while len(train) < train_size:
        index = randrange(len(data_set_copy))
        train.append(data_set_copy.pop(index))
    return train, data_set_copy


# --- Step 5: Algorithm Evaluation ---

def how_good_is_our_algo(data_set, algo, split, *args):
    """Evaluate an algorithm using a train/test split and RMSE.

    Args:
        data_set: The full dataset.
        algo: The prediction algorithm function.
        split: Fraction of data for training.
        *args: Additional arguments passed to the algorithm.

    Returns:
        The RMSE score on the test set.
    """
    train, test = train_test_split(data_set, split)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)

    predicted = algo(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = calculate_RMSE(actual, predicted)
    return rmse


# --- Step 6: Statistical Helpers ---

def mean(values):
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        values: List of numeric values.

    Returns:
        The mean value (float).
    """
    return sum(values) / float(len(values))


def covariance(x, the_mean_of_x, y, the_mean_of_y):
    """Calculate the covariance between two variables.

    Args:
        x: List of values for variable x.
        the_mean_of_x: Precomputed mean of x.
        y: List of values for variable y.
        the_mean_of_y: Precomputed mean of y.

    Returns:
        The covariance value (float).
    """
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - the_mean_of_x) * (y[i] - the_mean_of_y)
    return covar


def variance(values, mean_val):
    """Calculate the sum of squared deviations from the mean.

    Args:
        values: List of numeric values.
        mean_val: Precomputed mean of values.

    Returns:
        The variance sum (float).
    """
    return sum([(x - mean_val) ** 2 for x in values])


# --- Step 7: Compute Regression Coefficients ---

def coefficients(data_set):
    """Calculate the coefficients (b0, b1) for simple linear regression.

    Args:
        data_set: Dataset with two columns [x, y].

    Returns:
        A list [b0, b1] where y = b1 * x + b0.
    """
    x = [row[0] for row in data_set]
    y = [row[1] for row in data_set]
    the_mean_of_x = mean(x)
    the_mean_of_y = mean(y)

    b1 = covariance(x, the_mean_of_x, y, the_mean_of_y) / variance(x, the_mean_of_x)
    b0 = the_mean_of_y - b1 * the_mean_of_x

    return [b0, b1]


# --- Step 8: Simple Linear Regression Prediction ---

def using_simple_linear_regression(train, test):
    """Predict target values using simple linear regression.

    Args:
        train: Training dataset.
        test: Testing dataset.

    Returns:
        A list of predicted values.
    """
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        y_hat = b1 * row[0] + b0
        predictions.append(y_hat)
    return predictions


# --- Step 9: Run on Real Data ---

if __name__ == '__main__':
    seed(4)
    split = 0.6

    data_set = load_csv('../Y.Kaggle_Data/insurance.csv')
    for i in range(len(data_set[0])):
        string_converter(data_set, i)

    rmse = how_good_is_our_algo(data_set, using_simple_linear_regression, split)

    print("RMSE of our algo is: %.3f" % rmse)
