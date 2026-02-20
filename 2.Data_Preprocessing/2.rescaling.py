# Pure Python for Data Science & Machine Learning
# Author: William
"""Demonstrate min-max normalization and z-score standardization.

Loads the diabetes dataset, applies min-max normalization to scale
values into [0, 1], then applies z-score standardization so each
feature has zero mean and unit variance.
"""

from math import sqrt
import data_reader


# --- Step 1: Load and Prepare Data ---

dataset = data_reader.read_csv('diabetes.csv')
for i in range(len(dataset[0])):
    data_reader.convert_string_to_float(dataset, i)
dataset = dataset[1:]


# --- Step 2: Min-Max Normalization ---

def find_min_max(dataset):
    """Find the maximum and minimum value for each column.

    Args:
        dataset: A list of lists (rows x columns) of numeric values.

    Returns:
        A list of [max, min] pairs, one per column.
    """
    min_max = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_max = max(col_values)
        value_min = min(col_values)
        min_max.append([value_max, value_min])
    return min_max


def min_max_normalize(dataset, min_max):
    """Normalize the dataset in place using min-max scaling to [0, 1].

    Args:
        dataset: A list of lists to normalize in place.
        min_max: A list of [max, min] pairs per column.
    """
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - min_max[i][1]) / (min_max[i][0] - min_max[i][1])


# --- Step 3: Column Mean Calculation ---

def find_column_means(dataset):
    """Compute the mean of each column by accumulating into a list.

    Args:
        dataset: A list of lists (rows x columns) of numeric values.

    Returns:
        A list of column means.
    """
    means = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        mean = sum(col_values) / float(len(dataset))
        means.append(mean)
    return means


def find_column_means_v2(dataset):
    """Compute the mean of each column using pre-allocated list.

    Args:
        dataset: A list of lists (rows x columns) of numeric values.

    Returns:
        A list of column means.
    """
    means = [0.0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means


# --- Step 4: Standard Deviation and Z-Score Standardization ---

def calculate_stdevs(dataset, means):
    """Compute the sample standard deviation of each column.

    Args:
        dataset: A list of lists (rows x columns) of numeric values.
        means:   A list of pre-computed column means.

    Returns:
        A list of column standard deviations.
    """
    stdevs = [0.0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i] - means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(element / float(len(dataset) - 1)) for element in stdevs]
    return stdevs


def standardize_dataset(dataset, means, stdevs):
    """Apply z-score standardization to the dataset in place.

    Each value is transformed as: (value - mean) / stdev.

    Args:
        dataset: A list of lists to standardize in place.
        means:   A list of column means.
        stdevs:  A list of column standard deviations.
    """
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - means[i]) / stdevs[i]


# --- Step 5: Main Execution ---

if __name__ == '__main__':
    min_max = find_min_max(dataset)
    min_max_normalize(dataset, min_max)

    means_list = find_column_means_v2(dataset)
    stdevs_list = calculate_stdevs(dataset, means_list)

    standardize_dataset(dataset, means_list, stdevs_list)

    print(dataset)
