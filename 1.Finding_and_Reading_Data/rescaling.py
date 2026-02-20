# Pure Python for Data Science & Machine Learning
# Author: William
"""Standardize (z-score normalize) a dataset.

Demonstrates computing column means and standard deviations, then
applying z-score standardization so each feature has zero mean and
unit variance.
"""

from math import sqrt


# --- Step 1: Column Statistics ---

def column_means(dataset):
    """Compute the mean of each column in the dataset.

    Args:
        dataset: A list of lists (rows x columns) of numeric values.

    Returns:
        A list of column means.
    """
    means = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        means[i] = sum(col_values) / float(len(dataset))
    return means


def column_stdevs(dataset, means):
    """Compute the sample standard deviation of each column.

    Args:
        dataset: A list of lists (rows x columns) of numeric values.
        means:   A list of pre-computed column means.

    Returns:
        A list of column standard deviations.
    """
    stdevs = [0 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance = [pow(row[i] - means[i], 2) for row in dataset]
        stdevs[i] = sum(variance)
    stdevs = [sqrt(x / float(len(dataset) - 1)) for x in stdevs]
    return stdevs


# --- Step 2: Standardization ---

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


# --- Step 3: Main Execution ---

if __name__ == '__main__':
    dataset = [[50, 30], [20, 90], [30, 50]]
    print(dataset)

    means = column_means(dataset)
    stdevs = column_stdevs(dataset, means)

    dataset_alias = dataset
    standardize_dataset(dataset, means, stdevs)

    print(dataset)
    print(dataset_alias)
