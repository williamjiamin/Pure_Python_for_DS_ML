# Created by william from lexueoude.com. 更多正版技术视频讲解，
# 公众号1.乐学偶得（lexueoude）2.乐学FinTech (LoveShareFinTech)


# 1.import lib
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp


# 2. read our data
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# a= load_csv('seeds_dataset.csv')
# print(a)

# 3.change string date type to float

def str_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# 4.change class string to int

def str_column_to_int(dataset, column):
    class_value = [row[column] for row in dataset]
    # using set to make int unique
    unique = set(class_value)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# 5.normalize our data
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats


def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# 6. k_fold cross-validation

def cross_validation_split(dataset, n_folds):
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


# 7.calculate accuracy

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# 8.score our algo

def evaluate_our_algorithm(dataset, algorithm, n_folds, *args):
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


# 9. activation

def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# 10. transfer our neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# 11.forward propagate our network to get our output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# 12.calculate the derivative of a output
def transfer_derivative(output):
    return output * (1.0 - output)


# 13.calculate backpropagation error
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# 14. update our weight

def update_weights(network, row, learning_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += learning_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += learning_rate * neuron['delta']


# 15. epoch

def train_network(network, train, learning_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            backward_propagate_error(network, expected)
            update_weights(network, row, learning_rate)


# 16. initialization

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# 17. make prediction

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))


# 18. using backpropagation&SGD

def back_propagation(train, test, learning_rate, n_epoch, n_hidden):
    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, learning_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return (predictions)


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
print("The mean accuracy is [%.3f%%] " % results)
