# Created by william from lexueoude.com. 更多正版技术视频讲解，
# 公众号1.乐学偶得（lexueoude）2.乐学FinTech (LoveShareFinTech)

from math import exp


def neuron_activation(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


def neuron_transfer(activation):
    result = 1.0 / (1.0 + exp(-activation))
    return result


def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = neuron_activation(neuron['weights'], inputs)
            neuron['output'] = neuron_transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


def neuron_transfer_derivative(output):
    derivative = output * (1 - output)
    return derivative


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()

        #除了最后的output的layer，其他全部按照hidden layer的方式计算：
        # error = (weight_k * error_j) * transfer_derivative(output)
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        # 执行的是：error = (expected - output) * transfer_derivative(output)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])

        # *transfer_derivative(output)
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * neuron_transfer_derivative(neuron['output'])


network = [[{'output': 0.71, 'weights': [0.13436424411240122,
                                         0.8474337369372327,
                                         0.763774618976614]}],
           [{'output': 0.62, 'weights': [0.2550690257394217,
                                         0.49543508709194095]},
            {'output': 0.65, 'weights': [0.4494910647887381,
                                         0.651592972722763]}]]
expected = [0, 1]
backward_propagate_error(network, expected)

for layer in network:
    print(layer)
