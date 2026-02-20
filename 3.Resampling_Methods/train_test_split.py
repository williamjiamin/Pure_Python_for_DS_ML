# Pure Python for Data Science & Machine Learning
# Author: William
"""Split a dataset into training and testing subsets.

Randomly selects a proportion of rows for training; the remaining
rows form the test set.
"""

from random import seed
from random import randrange


# --- Step 1: Train/Test Split ---

def train_test_split(dataset, train=0.60):
    """Split a dataset into random training and testing subsets.

    Args:
        dataset: The full dataset as a list of rows.
        train:   The proportion of data to use for training (default 0.60).

    Returns:
        A tuple (train_set, test_set).
    """
    train_set = list()
    train_size = train * len(dataset)
    dataset_copy = list(dataset)

    while len(train_set) < train_size:
        index = randrange(len(dataset_copy))
        train_set.append(dataset_copy.pop(index))

    return train_set, dataset_copy


# --- Step 2: Main Execution ---

if __name__ == '__main__':
    seed(3)
    dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

    train, test = train_test_split(dataset)

    print(train)
    print(test)
