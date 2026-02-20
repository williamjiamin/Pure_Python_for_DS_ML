# Pure Python for Data Science & Machine Learning
# Author: William
"""Gaussian Naive Bayes classifier built from scratch.

Implements the full pipeline:
  1. Split data by class label
  2. Compute per-feature statistics (mean, stdev, count)
  3. Calculate Gaussian probability density
  4. Combine class priors with likelihoods to obtain posterior probabilities
"""

from math import sqrt, pi, exp


# --- Step 1: Split data by class -------------------------------------------

def split_data_by_class(dataset):
    """Group dataset rows by their class label (last column).

    Parameters
    ----------
    dataset : list[list[float]]
        Rows of [feature_1, ..., feature_n, class_label].

    Returns
    -------
    dict
        Mapping from class_label -> list of rows.
    """
    split_data = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in split_data:
            split_data[class_value] = list()
        split_data[class_value].append(vector)
    return split_data


# --- Step 2: Descriptive statistics ----------------------------------------

def calculate_mean(numbers):
    """Return the arithmetic mean of a list of numbers.

    Parameters
    ----------
    numbers : list[float]
        Numeric values.

    Returns
    -------
    float
        The mean.
    """
    return sum(numbers) / float(len(numbers))


def calculate_stdev(numbers):
    """Return the sample standard deviation of a list of numbers.

    Parameters
    ----------
    numbers : list[float]
        Numeric values (length >= 2).

    Returns
    -------
    float
        Sample standard deviation.
    """
    mean = calculate_mean(numbers)
    variance = sum([(x - mean) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


def describe_data(dataset):
    """Compute (mean, stdev, count) for each feature column.

    Parameters
    ----------
    dataset : list[list[float]]
        Data rows; the last column (class label) is excluded.

    Returns
    -------
    list[tuple[float, float, int]]
        Per-feature summaries [(mean, stdev, count), ...].
    """
    description = [
        (calculate_mean(column), calculate_stdev(column), len(column))
        for column in zip(*dataset)
    ]
    del description[-1]
    return description


def describe_data_by_class(dataset):
    """Compute per-feature statistics grouped by class label.

    Parameters
    ----------
    dataset : list[list[float]]
        Rows of [feature_1, ..., feature_n, class_label].

    Returns
    -------
    dict
        Mapping from class_label -> list of (mean, stdev, count) tuples.
    """
    split_data = split_data_by_class(dataset)
    data_description = dict()
    for class_value, rows in split_data.items():
        data_description[class_value] = describe_data(rows)
    return data_description


# --- Step 3: Gaussian probability density -----------------------------------

def gaussian_probability(x, mean, stdev):
    """Compute the Gaussian probability density function value.

    Parameters
    ----------
    x : float
        The observed value.
    mean : float
        Distribution mean.
    stdev : float
        Distribution standard deviation.

    Returns
    -------
    float
        P(x | mean, stdev) under a Gaussian distribution.
    """
    exponent = exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


# --- Step 4: Class posterior probabilities ----------------------------------

def calculate_class_probabilities(description, row):
    """Calculate posterior probability for each class given a data row.

    Uses the Naive Bayes assumption that features are conditionally
    independent given the class.

    Parameters
    ----------
    description : dict
        Per-class feature summaries from describe_data_by_class().
    row : list[float]
        A single data row (features only; label ignored if present).

    Returns
    -------
    dict
        Mapping from class_label -> posterior probability (unnormalised).
    """
    total_rows = sum([description[label][0][2] for label in description])
    probabilities = dict()
    for class_value, class_description in description.items():
        probabilities[class_value] = description[class_value][0][2] / float(total_rows)
        for i in range(len(class_description)):
            mean, stdev, count = class_description[i]
            probabilities[class_value] *= gaussian_probability(row[i], mean, stdev)
    return probabilities


# --- Main execution --------------------------------------------------------

if __name__ == '__main__':
    dataset = [[0.8, 2.3, 0],
               [2.1, 1.6, 0],
               [2.0, 3.6, 0],
               [3.1, 2.5, 0],
               [3.8, 4.7, 0],
               [6.1, 4.4, 1],
               [8.6, 0.3, 1],
               [7.9, 5.3, 1],
               [9.1, 2.5, 1],
               [6.8, 2.7, 1]]

    description = describe_data_by_class(dataset)
    probability = calculate_class_probabilities(description, dataset[0])
    print(probability)
