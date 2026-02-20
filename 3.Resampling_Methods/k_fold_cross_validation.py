# Pure Python for Data Science & Machine Learning
# Author: William
"""K-fold cross-validation splitting utility.

Randomly partitions a dataset into *k* equally-sized folds that can
be used for cross-validated model evaluation.
"""

from random import seed
from random import randrange


# --- Step 1: K-Fold Split ---

def k_fold_cross_validation_split(dataset, folds=10):
    """Split a dataset into k random, non-overlapping folds.

    Args:
        dataset: The full dataset as a list of rows.
        folds:   The number of folds to create (default 10).

    Returns:
        A list of folds, where each fold is a list of rows.
    """
    split_data = list()
    fold_size = int(len(dataset) / folds)
    dataset_copy = list(dataset)

    for i in range(folds):
        chosen_fold = list()
        while len(chosen_fold) < fold_size:
            index = randrange(len(dataset_copy))
            chosen_fold.append(dataset_copy.pop(index))
        split_data.append(chosen_fold)

    return split_data


# --- Step 2: Main Execution ---

if __name__ == '__main__':
    seed(1)
    dataset = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]

    k_folds_split = k_fold_cross_validation_split(dataset, 5)

    print(k_folds_split)
