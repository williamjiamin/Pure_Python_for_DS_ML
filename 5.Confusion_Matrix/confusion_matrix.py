# Pure Python for Data Science & Machine Learning
# Author: William
"""Build and display a confusion matrix for multi-class classification.

Given lists of actual and predicted labels, constructs a confusion matrix
and pretty-prints it with class labels.
"""


# --- Step 1: Build Confusion Matrix ---

def confusion_matrix(actual, predicted):
    """Build a confusion matrix from actual and predicted labels.

    Args:
        actual:    A list of true class labels.
        predicted: A list of predicted class labels (same length as actual).

    Returns:
        A tuple (unique_classes, matrix) where unique_classes is a set of
        distinct labels and matrix is a 2-D list of counts indexed by
        [predicted_class][actual_class].
    """
    unique_classes = set(actual)
    matrix = [list() for x in range(len(unique_classes))]
    for i in range(len(unique_classes)):
        matrix[i] = [0 for x in range(len(unique_classes))]

    class_index = dict()
    for i, class_value in enumerate(unique_classes):
        class_index[class_value] = i

    for i in range(len(actual)):
        col = class_index[actual[i]]
        row = class_index[predicted[i]]
        matrix[row][col] += 1

    return unique_classes, matrix


# --- Step 2: Pretty-Print ---

def pretty_confusion_matrix(unique_classes, matrix):
    """Print the confusion matrix in a human-readable table format.

    Args:
        unique_classes: A set of distinct class labels.
        matrix:         The confusion matrix (2-D list of counts).
    """
    print('(Actual)   ' + ' '.join(str(x) for x in unique_classes))
    print('(Predicted)-------------------------')
    for i, x in enumerate(unique_classes):
        print("{}  |       {}".format(x, ' '.join(str(x) for x in matrix[i])))


# --- Step 3: Main Execution ---

if __name__ == '__main__':
    actual = [0, 2, 0, 8, 7, 8, 6, 5, 2, 1]
    predicted = [2, 0, 2, 0, 1, 0, 0, 7, 0, 1]

    unique_classes, matrix = confusion_matrix(actual, predicted)
    pretty_confusion_matrix(unique_classes, matrix)
