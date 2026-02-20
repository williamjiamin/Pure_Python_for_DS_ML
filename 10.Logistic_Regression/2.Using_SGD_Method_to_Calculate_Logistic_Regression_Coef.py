# Pure Python for Data Science & Machine Learning
# Author: William
"""Learn logistic regression coefficients via Stochastic Gradient Descent.

Uses the sigmoid function for binary classification and updates weights with
SGD, incorporating the derivative of the sigmoid (y_hat * (1 - y_hat)) in
the gradient step.
"""

from math import exp


# --- Step 1: Prediction (sigmoid) ------------------------------------------

def prediction(row, coefficients):
    """Predict the probability of class 1 using logistic (sigmoid) function.

    Parameters
    ----------
    row : list[float]
        A data row where all elements except the last are features.
    coefficients : list[float]
        Model coefficients: [intercept, weight_1, weight_2, ...].

    Returns
    -------
    float
        Predicted probability in (0, 1).
    """
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return 1 / (1 + exp(-yhat))


# --- Step 2: SGD coefficient estimation ------------------------------------

def sgd_logistic_coefficients(training_dataset, learning_rate, n_epochs):
    """Estimate logistic regression coefficients using SGD.

    Parameters
    ----------
    training_dataset : list[list[float]]
        Rows of [feature_1, ..., feature_n, label] where label is 0 or 1.
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
            y_hat = prediction(row, coefficients)
            error = row[-1] - y_hat
            sum_error += error ** 2
            coefficients[0] = coefficients[0] + learning_rate * error * y_hat * (1.0 - y_hat)
            for i in range(len(row) - 1):
                coefficients[i + 1] = (coefficients[i + 1]
                                       + learning_rate * error * y_hat * (1.0 - y_hat) * row[i])
        print("Epoch [%d], learning rate [%.3f], error [%.3f]"
              % (epoch, learning_rate, sum_error))
    return coefficients


# --- Main execution --------------------------------------------------------

if __name__ == '__main__':
    dataset = [[2, 2, 0],
               [2, 4, 0],
               [3, 3, 0],
               [4, 5, 0],
               [8, 1, 1],
               [8.5, 3.5, 1],
               [9, 1, 1],
               [10, 4, 1]]

    learning_rate = 0.1
    n_epochs = 1000

    coef = sgd_logistic_coefficients(dataset, learning_rate, n_epochs)
    print(coef)
