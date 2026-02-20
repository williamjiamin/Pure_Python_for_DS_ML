# Pure Python for Data Science & Machine Learning
# Author: William
"""Backpropagation error test for a simple neural network.

Demonstrates forward propagation and backward error propagation
on a small two-layer network with hardcoded weights and outputs.
"""

from math import exp


# --- Step 1: Neuron Activation ---

def neuron_activation(weights, inputs):
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


# --- Step 2: Sigmoid Transfer ---

def neuron_transfer(activation):
    """Apply the sigmoid activation function.

    Args:
        activation: The raw activation value.

    Returns:
        Sigmoid output in the range (0, 1).
    """
    return 1.0 / (1.0 + exp(-activation))


# --- Step 3: Forward Propagation ---

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
            activation = neuron_activation(neuron['weights'], inputs)
            neuron['output'] = neuron_transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# --- Step 4: Transfer Derivative ---

def neuron_transfer_derivative(output):
    """Compute the derivative of the sigmoid function.

    Args:
        output: The sigmoid output value.

    Returns:
        The derivative: output * (1 - output).
    """
    return output * (1 - output)


# --- Step 5: Backward Propagation of Error ---

def backward_propagate_error(network, expected):
    """Propagate error backward through the network and compute deltas.

    For hidden layers:
        error = sum(weight_k * delta_j) for each downstream neuron
    For the output layer:
        error = expected - output

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
            neuron['delta'] = errors[j] * neuron_transfer_derivative(neuron['output'])


# --- Step 6: Run Test ---

if __name__ == '__main__':
    network = [
        [{'output': 0.71, 'weights': [0.13436424411240122,
                                       0.8474337369372327,
                                       0.763774618976614]}],
        [{'output': 0.62, 'weights': [0.2550690257394217,
                                       0.49543508709194095]},
         {'output': 0.65, 'weights': [0.4494910647887381,
                                       0.651592972722763]}]
    ]
    expected = [0, 1]
    backward_propagate_error(network, expected)

    for layer in network:
        print(layer)
