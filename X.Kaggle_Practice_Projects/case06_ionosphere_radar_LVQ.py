# Pure Python for Data Science & Machine Learning
# Author: William
"""Classify ionosphere radar data using Learning Vector Quantization (LVQ).

Loads the ionosphere dataset, converts features and class labels,
and evaluates an LVQ classifier using k-fold cross-validation.
"""

from random import seed
from random import randrange
from csv import reader
from math import sqrt


# --- Step 1: Load CSV Data ---

def read_our_csv_file(filename):
    """Load a CSV file into a list of lists.

    Args:
        filename: Path to the CSV file.

    Returns:
        A list of rows, where each row is a list of string values.
    """
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# --- Step 2: Data Type Conversion (String to Float) ---

def convert_string_to_float(dataset, column):
    """Convert a column of string values to float in place.

    Args:
        dataset: The dataset (list of rows).
        column: Column index to convert.
    """
    for row in dataset:
        row[column] = float(row[column].strip())


# --- Step 3: Class Label Conversion (String to Int) ---

def convert_string_to_int(dataset, column):
    """Convert a column of string class labels to integer codes.

    Args:
        dataset: The dataset (modified in place).
        column: Column index containing class labels.

    Returns:
        A dict mapping original string values to integer codes.
    """
    class_value = [row[column] for row in dataset]
    unique_class = set(class_value)
    look_up = dict()
    for i, value in enumerate(unique_class):
        look_up[value] = i
    for row in dataset:
        row[column] = look_up[row[column]]
    return look_up


# --- Step 4: K-Fold Cross-Validation Split ---

def k_fold_cross_validation_split(dataset, n_folds):
    """Split dataset into k folds for cross-validation.

    Args:
        dataset: The dataset to split.
        n_folds: Number of folds.

    Returns:
        A list of folds, each fold is a list of rows.
    """
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split


# --- Step 5: Accuracy Metric ---

def calculate_our_accuracy(actual, predicted):
    """Calculate classification accuracy as a percentage.

    Args:
        actual: List of actual class labels.
        predicted: List of predicted class labels.

    Returns:
        Accuracy percentage (float).
    """
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# --- Step 6: Algorithm Evaluation with Cross-Validation ---

def whether_our_algo_is_good_or_not(dataset, algo, n_folds, *args):
    """Evaluate an algorithm using k-fold cross-validation.

    Args:
        dataset: The full dataset.
        algo: The prediction algorithm function.
        n_folds: Number of cross-validation folds.
        *args: Additional arguments passed to the algorithm.

    Returns:
        A list of accuracy scores, one per fold.
    """
    folds = k_fold_cross_validation_split(dataset, n_folds)
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
        predicted = algo(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = calculate_our_accuracy(actual, predicted)
        scores.append(accuracy)
    return scores


# --- Step 7: Euclidean Distance ---

def calculate_euclidean_distance(row1, row2):
    """Calculate the Euclidean distance between two rows.

    Args:
        row1: First data row.
        row2: Second data row.

    Returns:
        Euclidean distance (float), ignoring the last column (label).
    """
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# --- Step 8: Best Matching Unit (BMU) ---

def calculate_BMU(codebooks, test_row):
    """Find the best matching unit (closest codebook) to a test row.

    Args:
        codebooks: List of codebook vectors.
        test_row: The test row to match against.

    Returns:
        The codebook vector closest to the test row.
    """
    distances = list()
    for codebook in codebooks:
        dist = calculate_euclidean_distance(codebook, test_row)
        distances.append((codebook, dist))
    distances.sort(key=lambda every_tuple: every_tuple[1])
    return distances[0][0]


# --- Step 9: Prediction and Random Codebook Generation ---

def predict(codebooks, test_row):
    """Predict the class label using the BMU's class.

    Args:
        codebooks: List of codebook vectors.
        test_row: The test row to classify.

    Returns:
        The class label of the best matching unit.
    """
    bmu = calculate_BMU(codebooks, test_row)
    return bmu[-1]


def random_codebook(train):
    """Generate a random codebook vector from the training data.

    Each feature is randomly sampled from the corresponding column.

    Args:
        train: The training dataset.

    Returns:
        A codebook vector (list of feature values).
    """
    train_index = len(train)
    n_features = len(train[0])
    codebook = [train[randrange(train_index)][i] for i in range(n_features)]
    return codebook


# --- Step 10: Train Codebooks ---

def train_our_codebooks(train, n_codebooks, learning_rate, epochs):
    """Train codebook vectors using the LVQ learning rule.

    Codebooks are moved closer to matching-class samples and
    further from non-matching-class samples.

    Args:
        train: The training dataset.
        n_codebooks: Number of codebook vectors.
        learning_rate: Initial learning rate (decays over epochs).
        epochs: Number of training epochs.

    Returns:
        A list of trained codebook vectors.
    """
    codebooks = [random_codebook(train) for i in range(n_codebooks)]
    for epoch in range(epochs):
        rate = learning_rate * (1.0 - (epoch / float(epochs)))
        for row in train:
            bmu = calculate_BMU(codebooks, row)
            for i in range(len(row) - 1):
                error = row[i] - bmu[i]
                if bmu[-1] == row[-1]:
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error
    return codebooks


# --- Step 11: LVQ Algorithm ---

def LVQ(train, test, n_codebooks, learning_rate, epochs):
    """Learning Vector Quantization classification algorithm.

    Args:
        train: Training dataset.
        test: Testing dataset.
        n_codebooks: Number of codebook vectors.
        learning_rate: Initial learning rate.
        epochs: Number of training epochs.

    Returns:
        A list of predicted class labels.
    """
    codebooks = train_our_codebooks(train, n_codebooks, learning_rate, epochs)
    predictions = list()
    for row in test:
        output = predict(codebooks, row)
        predictions.append(output)
    return predictions


# --- Step 12: Run on Ionosphere Dataset ---

if __name__ == '__main__':
    seed(1)

    dataset = read_our_csv_file("ionosphere.csv")
    for i in range(len(dataset[0]) - 1):
        convert_string_to_float(dataset, i)

    convert_string_to_int(dataset, len(dataset[0]) - 1)

    n_folds = 5
    learning_rate = 0.3
    n_epochs = 50
    n_codebooks = 50

    scores = whether_our_algo_is_good_or_not(dataset, LVQ, n_folds, n_codebooks, learning_rate, n_epochs)

    print("Our LVQ algo's scores are: %s" % scores)
    print("Our LVQ's mean accuracy is %.3f%%" % (sum(scores) / float(len(scores))))
