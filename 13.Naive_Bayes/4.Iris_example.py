# Pure Python for Data Science & Machine Learning
# Author: William
"""Evaluate a Gaussian Naive Bayes classifier on the Iris dataset.

Loads the Iris CSV, converts strings to numeric types, runs k-fold
cross-validation, and reports per-fold accuracy together with the
overall mean accuracy.
"""

from csv import reader
from random import randrange
from math import sqrt, exp, pi


# --- Step 1: Data loading and conversion ------------------------------------

def load_csv(filename):
    """Load a CSV file into a list of lists.

    Parameters
    ----------
    filename : str
        Path to the CSV file.

    Returns
    -------
    list[list[str]]
        Each inner list is one row of string values.
    """
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def convert_str_to_float(dataset, column):
    """Convert a column of string values to floats in-place.

    Parameters
    ----------
    dataset : list[list]
        The dataset (modified in-place).
    column : int
        Column index to convert.
    """
    for row in dataset:
        row[column] = float(row[column].strip())


def convert_str_to_int(dataset, column):
    """Map unique string labels to integer codes in-place.

    Parameters
    ----------
    dataset : list[list]
        The dataset (modified in-place).
    column : int
        Column index containing class labels.

    Returns
    -------
    dict
        Lookup mapping original label -> integer code.
    """
    class_values = [row[column] for row in dataset]
    unique_values = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique_values):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# --- Step 2: Cross-validation -----------------------------------------------

def k_fold_split(dataset, n_folds):
    """Split a dataset into k random folds.

    Parameters
    ----------
    dataset : list[list]
        The full dataset.
    n_folds : int
        Number of folds.

    Returns
    -------
    list[list[list]]
        A list of folds, each fold being a list of rows.
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


def calculate_accuracy(actual, predicted):
    """Compute classification accuracy as a percentage.

    Parameters
    ----------
    actual : list
        True class labels.
    predicted : list
        Predicted class labels.

    Returns
    -------
    float
        Accuracy percentage (0-100).
    """
    correct_count = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct_count += 1
    return correct_count / float(len(actual)) * 100.0


def evaluate_with_cv(dataset, algorithm, n_folds, *args):
    """Evaluate an algorithm using k-fold cross-validation.

    Parameters
    ----------
    dataset : list[list]
        Full dataset.
    algorithm : callable
        A function(train, test, *args) -> list of predictions.
    n_folds : int
        Number of cross-validation folds.
    *args
        Extra arguments forwarded to *algorithm*.

    Returns
    -------
    list[float]
        Accuracy score for each fold.
    """
    folds = k_fold_split(dataset, n_folds)
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
        accuracy = calculate_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


# --- Step 3: Naive Bayes implementation -------------------------------------

def split_data_by_class(dataset):
    """Group rows by class label.

    Parameters
    ----------
    dataset : list[list[float]]
        Rows of [features..., class_label].

    Returns
    -------
    dict
        class_label -> list of rows.
    """
    split_data = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in split_data:
            split_data[class_value] = list()
        split_data[class_value].append(vector)
    return split_data


def mean(numbers):
    """Return the arithmetic mean.

    Parameters
    ----------
    numbers : list[float]

    Returns
    -------
    float
    """
    return sum(numbers) / float(len(numbers))


def stdev(numbers):
    """Return the sample standard deviation.

    Parameters
    ----------
    numbers : list[float]

    Returns
    -------
    float
    """
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


def describe_data(dataset):
    """Compute (mean, stdev, count) per feature column, excluding the label.

    Parameters
    ----------
    dataset : list[list[float]]

    Returns
    -------
    list[tuple[float, float, int]]
    """
    description = [
        (mean(column), stdev(column), len(column))
        for column in zip(*dataset)
    ]
    del description[-1]
    return description


def describe_data_by_class(dataset):
    """Per-class feature statistics.

    Parameters
    ----------
    dataset : list[list[float]]

    Returns
    -------
    dict
        class_label -> list of (mean, stdev, count).
    """
    data_split = split_data_by_class(dataset)
    description = dict()
    for class_value, rows in data_split.items():
        description[class_value] = describe_data(rows)
    return description


def gaussian_probability(x, m, s):
    """Gaussian PDF value.

    Parameters
    ----------
    x : float
    m : float
        Mean.
    s : float
        Standard deviation.

    Returns
    -------
    float
    """
    exponent = exp(-((x - m) ** 2 / (2 * s ** 2)))
    return (1 / (sqrt(2 * pi) * s)) * exponent


def calculate_class_probabilities(description, row):
    """Posterior probability for each class (unnormalised).

    Parameters
    ----------
    description : dict
        From describe_data_by_class().
    row : list[float]

    Returns
    -------
    dict
        class_label -> probability.
    """
    total_rows = sum([description[label][0][2] for label in description])
    probabilities = dict()
    for class_value, class_desc in description.items():
        probabilities[class_value] = description[class_value][0][2] / float(total_rows)
        for i in range(len(class_desc)):
            m, s, count = class_desc[i]
            probabilities[class_value] *= gaussian_probability(row[i], m, s)
    return probabilities


def predict(description, row):
    """Predict the class label for a single row.

    Parameters
    ----------
    description : dict
        Per-class feature summaries.
    row : list[float]

    Returns
    -------
    int
        Predicted class label.
    """
    probabilities = calculate_class_probabilities(description, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


def naive_bayes(train, test):
    """Run Naive Bayes: fit on train, predict on test.

    Parameters
    ----------
    train : list[list[float]]
        Training data.
    test : list[list[float]]
        Test data (label column may be None).

    Returns
    -------
    list
        Predicted labels for each test row.
    """
    description = describe_data_by_class(train)
    predictions = list()
    for row in test:
        prediction = predict(description, row)
        predictions.append(prediction)
    return predictions


# --- Main execution --------------------------------------------------------

if __name__ == '__main__':
    dataset = load_csv('iris.csv')

    for i in range(len(dataset[0]) - 1):
        convert_str_to_float(dataset, i)
    convert_str_to_int(dataset, len(dataset[0]) - 1)

    n_folds = 5
    scores = evaluate_with_cv(dataset, naive_bayes, n_folds)

    print("Scores per fold: %s" % scores)
    print("Mean accuracy: %.6f%%" % (sum(scores) / float(len(scores))))
