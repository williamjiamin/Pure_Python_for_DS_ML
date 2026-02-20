# Pure Python for Data Science & Machine Learning
# Author: William
"""Find the optimal binary split for a dataset using the Gini index.

Iterates over every feature and every value in the dataset, evaluates the
resulting Gini impurity, and returns the split with the lowest score.
"""


# --- Step 1: Binary split helper -------------------------------------------

def test_split(index, value, dataset):
    """Split a dataset into left/right groups on a feature threshold.

    Parameters
    ----------
    index : int
        Feature column index to split on.
    value : float
        Threshold; rows with feature < value go left.
    dataset : list[list[float]]
        The dataset to split.

    Returns
    -------
    tuple[list, list]
        (left_group, right_group).
    """
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# --- Step 2: Gini index calculation ----------------------------------------

def calculate_gini_index(groups, classes):
    """Compute the weighted Gini impurity for a candidate split.

    Parameters
    ----------
    groups : tuple[list, list]
        Two groups (left, right) produced by a binary split.
    classes : list
        Unique class labels present in the dataset.

    Returns
    -------
    float
        Weighted Gini impurity (0.0 = pure).
    """
    n_instances = float(sum([len(group) for group in groups]))
    gini = 0.0
    for group in groups:
        size = float(len(group))
        if size == 0:
            continue
        score = 0.0
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        gini += (1 - score) * (size / n_instances)
    return gini


# --- Step 3: Greedy best-split search --------------------------------------

def get_best_split(dataset):
    """Find the best split point by exhaustive Gini-based search.

    Parameters
    ----------
    dataset : list[list[float]]
        Rows of [feature_1, ..., feature_n, class_label].

    Returns
    -------
    dict
        Keys: 'index', 'value', 'groups'.
    """
    class_values = list(set(row[-1] for row in dataset))
    best_index, best_value, best_score, best_groups = (
        None, None, float('inf'), None
    )
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = calculate_gini_index(groups, class_values)
            print("X%d < %.3f Gini=%.3f" % ((index + 1), row[index], gini))
            if gini < best_score:
                best_index, best_value, best_score, best_groups = (
                    index, row[index], gini, groups
                )
    return {'index': best_index, 'value': best_value, 'groups': best_groups}


# --- Main execution --------------------------------------------------------

if __name__ == '__main__':
    dataset = [[2.1, 1.1, 0],
               [3.4, 2.5, 0],
               [1.3, 5.8, 0],
               [1.9, 8.6, 0],
               [3.7, 6.2, 0],
               [8.8, 1.1, 1],
               [9.6, 3.4, 1],
               [10.2, 7.4, 1],
               [7.7, 8.8, 1],
               [9.7, 6.9, 1]]

    split = get_best_split(dataset)
    print('Split: [X%d < %.3f]' % ((split['index'] + 1), split['value']))
