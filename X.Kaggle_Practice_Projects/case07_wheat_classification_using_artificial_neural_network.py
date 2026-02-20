# Pure Python for Data Science & Machine Learning
# Author: William
"""Classify wheat seeds using an artificial neural network with backpropagation.

Loads the wheat seeds dataset, normalizes features, and trains a
single-hidden-layer neural network using backpropagation and SGD.
Evaluates with k-fold cross-validation.
"""

from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp


# --- Step 1: Load CSV Data ---

def load_csv(filename):
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


# --- Step 2: Data Type Conversion ---

def str_to_float(dataset, column):
    """Convert a column of string values to float in place.

    Args:
        dataset: The dataset (list of rows).
        column: Column index to convert.
    """
    for row in dataset:
        row[column] = float(row[column].strip())


# --- Step 3: Class Label Conversion (String to Int) ---

def str_column_to_int(dataset, column):
    """Convert a column of string class labels to integer codes.

    Args:
        dataset: The dataset (modified in place).
        column: Column index containing class labels.

    Returns:
        A dict mapping original string values to integer codes.
    """
    class_value = [row[column] for row in dataset]
    unique = set(class_value)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# --- Step 4: Min-Max Normalization ---

def dataset_minmax(dataset):
    """Compute the min and max for each feature column.

    Args:
        dataset: The dataset (list of rows).

    Returns:
        A list of [min, max] pairs, one per column.
    """
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


def normalize_dataset(dataset, minmax):
    """Normalize feature columns to the range [0, 1].

    The last column (class label) is left unchanged.

    Args:
        dataset: The dataset (modified in place).
        minmax: List of [min, max] pairs per column.
    """
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# --- Step 5: K-Fold Cross-Validation Split ---

def cross_validation_split(dataset, n_folds):
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


# --- Step 6: Accuracy Metric ---

def accuracy_metric(actual, predicted):
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


# --- Step 7: Algorithm Evaluation with Cross-Validation ---

def evaluate_our_algorithm(dataset, algorithm, n_folds, *args):
    """Evaluate an algorithm using k-fold cross-validation.

    Args:
        dataset: The full dataset.
        algorithm: The prediction algorithm function.
        n_folds: Number of cross-validation folds.
        *args: Additional arguments passed to the algorithm.

    Returns:
        A list of accuracy scores, one per fold.
    """
    folds = cross_validation_split(dataset, n_folds)
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
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


# --- Step 8: Neuron Activation ---

def activate(weights, inputs):
    """Compute the weighted sum of inputs plus bias.

    Args:
        weights: List of weights; the last element is the bias.
        inputs: List of input values.

    Returns:
        The weighted activation value (float).
    """
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# --- Step 9: Sigmoid Transfer ---

def transfer(activation):
    """Apply the sigmoid activation function.

    Args:
        activation: The raw activation value.

    Returns:
        Sigmoid output in the range (0, 1).
    """
    return 1.0 / (1.0 + exp(-activation))


# --- Step 10: Forward Propagation ---

def forward_propagate(network, row):
    """Propagate an input row forward through the network.

    Args:
        network: List of layers, each layer is a list of neuron dicts.
        row: Input data row.

    Returns:
        List of output values from the final layer.
    """
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# --- Step 11: Transfer Derivative ---

def transfer_derivative(output):
    """Compute the derivative of the sigmoid function.

    Args:
        output: The sigmoid output value.

    Returns:
        The derivative: output * (1.0 - output).
    """
    return output * (1.0 - output)


# --- Step 12: Backward Propagation of Error ---

def backward_propagate_error(network, expected):
    """Propagate error backward through the network and compute deltas.

    Args:
        network: List of layers (modified in place with 'delta' keys).
        expected: List of expected output values.
    """
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += neuron['weights'][j] * neuron['delta']
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# --- Step 13: Update Weights ---

def update_weights(network, row, learning_rate):
    """Update network weights using the computed deltas.

    Args:
        network: The neural network (modified in place).
        row: The input row used in the forward pass.
        learning_rate: Step size for weight updates.
    """
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['delta']


# --- Step 14: Train the Network ---

def train_network(network, train, learning_rate, n_epoch, n_outputs):
    """Train the neural network over multiple epochs.

    Args:
        network: The neural network to train.
        train: Training dataset.
        learning_rate: Step size for weight updates.
        n_epoch: Number of training epochs.
        n_outputs: Number of output classes.
    """
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, learning_rate)


# --- Step 15: Network Initialization ---

def initialize_network(n_inputs, n_hidden, n_outputs):
    """Initialize a neural network with random weights.

    Creates a single hidden layer network.

    Args:
        n_inputs: Number of input features.
        n_hidden: Number of neurons in the hidden layer.
        n_outputs: Number of output neurons (classes).

    Returns:
        The initialized network (list of layers).
    """
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# --- Step 16: Make a Prediction ---

def predict(network, row):
    """Predict the class for a single row using the trained network.

    Args:
        network: The trained neural network.
        row: The data row to classify.

    Returns:
        The index of the output neuron with the highest activation.
    """
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# --- Step 17: Backpropagation Algorithm ---

def back_propagation(train, test, learning_rate, n_epoch, n_hidden):
    """Train a neural network with backpropagation and predict on test data.

    Args:
        train: Training dataset.
        test: Testing dataset.
        learning_rate: Step size for weight updates.
        n_epoch: Number of training epochs.
        n_hidden: Number of hidden neurons.

    Returns:
        A list of predicted class labels for the test set.
    """
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, learning_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return predictions


# --- Step 18: Run on Wheat Seeds Dataset ---

if __name__ == '__main__':
    seed(1)
    dataset = load_csv('seeds_dataset.csv')
    for i in range(len(dataset[0]) - 1):
        str_to_float(dataset, i)

    str_column_to_int(dataset, len(dataset[0]) - 1)

    minmax = dataset_minmax(dataset)
    normalize_dataset(dataset, minmax)

    n_folds = 5
    learning_rate = 0.1
    n_epoch = 1000
    n_hidden = 5
    scores = evaluate_our_algorithm(dataset, back_propagation, n_folds, learning_rate, n_epoch, n_hidden)

    print("Our algo's score is [%s]" % scores)
    results = sum(scores) / float(len(scores))
    print("The mean accuracy is [%.3f%%]" % results)
