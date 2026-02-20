# Pure Python for Data Science & Machine Learning
# Author: William
"""Learn linear regression coefficients via Stochastic Gradient Descent.

Implements SGD from scratch: iterates over training rows, computes the
prediction error, and updates the intercept and weights using a fixed
learning rate until convergence.
"""


# --- Step 1: Prediction ----------------------------------------------------

def make_prediction(input_row, coefficients):
    """Compute a linear prediction for a single data row.

    Parameters
    ----------
    input_row : list[float]
        A data row where all elements except the last are features.
    coefficients : list[float]
        Model coefficients: [intercept, weight_1, weight_2, ...].

    Returns
    -------
    float
        The predicted value y_hat = b0 + b1*x1 + b2*x2 + ...
    """
    y_hat = coefficients[0]
    for i in range(len(input_row) - 1):
        y_hat += coefficients[i + 1] * input_row[i]
    return y_hat


# --- Step 2: SGD coefficient estimation ------------------------------------

def sgd_coefficients(training_dataset, learning_rate, n_epochs):
    """Estimate linear regression coefficients using SGD.

    Parameters
    ----------
    training_dataset : list[list[float]]
        Rows of [feature_1, ..., feature_n, target].
    learning_rate : float
        Step size for each gradient update.
    n_epochs : int
        Number of full passes over the training data.

    Returns
    -------
    list[float]
        Learned coefficients [intercept, w1, w2, ...].
    """
    coefficients = [0.0 for _ in range(len(training_dataset[0]))]
    for epoch in range(n_epochs):
        sum_error = 0
        for row in training_dataset:
            y_hat = make_prediction(row, coefficients)
            error = y_hat - row[-1]
            sum_error += error ** 2
            coefficients[0] = coefficients[0] - learning_rate * error
            for i in range(len(row) - 1):
                coefficients[i + 1] = coefficients[i + 1] - learning_rate * error * row[i]
        print("Epoch [%d], learning rate [%.3f], error [%.3f]"
              % (epoch, learning_rate, sum_error))
    return coefficients


# --- Main execution --------------------------------------------------------

if __name__ == '__main__':
    your_training_dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
    your_model_learning_rate = 0.001
    your_n_epoch = 1000000
    your_coefficients = sgd_coefficients(
        your_training_dataset, your_model_learning_rate, your_n_epoch
    )
    print(your_coefficients)
