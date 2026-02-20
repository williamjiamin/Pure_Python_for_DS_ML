# Pure Python for Data Science & Machine Learning
# Author: William
"""Calculate classification accuracy.

Compares actual and predicted class labels element-wise and returns
the fraction of correct predictions.
"""


# --- Step 1: Accuracy Calculation ---

def calculate_accuracy(actual, predicted):
    """Compute the classification accuracy between actual and predicted labels.

    Args:
        actual:    A list of true class labels.
        predicted: A list of predicted class labels (same length as actual).

    Returns:
        The accuracy as a float between 0.0 and 1.0.
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual))


# --- Step 2: Main Execution ---

if __name__ == '__main__':
    actual = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    predicted = [1, 1, 1, 1, 1, 0, 0, 0, 0, 1]

    accuracy = calculate_accuracy(actual, predicted)
    print(accuracy)
