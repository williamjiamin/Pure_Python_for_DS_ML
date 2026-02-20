# Pure Python for Data Science & Machine Learning
# Author: William
"""Train a single-layer Perceptron for binary classification.

Implements the perceptron learning rule: for each misclassified sample the
weights are nudged in the direction that reduces the error, using a fixed
learning rate over a given number of epochs.
"""


# --- Step 1: Prediction ----------------------------------------------------

def predict(row, weights):
    """Predict binary class (0 or 1) using the perceptron activation rule.

    Parameters
    ----------
    row : list[float]
        A data row where all elements except the last are features.
    weights : list[float]
        Perceptron weights: [bias, w1, w2, ...].

    Returns
    -------
    float
        1.0 if the weighted activation >= 0, else 0.0.
    """
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


# --- Step 2: Weight optimisation -------------------------------------------

def optimize_weights(train, learning_rate, n_epochs):
    """Train perceptron weights on the given dataset.

    Parameters
    ----------
    train : list[list[float]]
        Training rows of [feature_1, ..., feature_n, label].
    learning_rate : float
        Step size for weight updates.
    n_epochs : int
        Number of full passes over the training data.

    Returns
    -------
    list[float]
        Learned weights [bias, w1, w2, ...].
    """
    weights = [0.0 for _ in range(len(train[0]))]
    for epoch in range(n_epochs):
        sum_error = 0.0
        for row in train:
            pred = predict(row, weights)
            error = row[-1] - pred
            sum_error += error ** 2
            weights[0] = weights[0] + learning_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + learning_rate * error * row[i]
        print("Epoch: %d, learning_rate: %.4f, error: %.4f"
              % (epoch, learning_rate, sum_error))
    return weights


# --- Main execution --------------------------------------------------------

if __name__ == '__main__':
    dataset = [[2.78, 2.55, 0],
               [1.47, 2.36, 0],
               [1.39, 1.85, 0],
               [3.06, 3.01, 0],
               [7.63, 2.76, 0],
               [5.33, 2.09, 1],
               [6.93, 1.76, 1],
               [8.76, -0.77, 1],
               [7.66, 2.46, 1]]

    learning_rate = 0.1
    n_epochs = 200

    weights = optimize_weights(dataset, learning_rate, n_epochs)
    print(weights)
