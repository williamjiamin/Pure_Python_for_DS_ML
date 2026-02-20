# Pure Python for Data Science & Machine Learning
# Author: William
"""Demonstrate simple linear prediction using pre-set coefficients.

Shows how a linear model computes y_hat = b0 + b1*x1 given a small dataset
and fixed coefficients, comparing predictions against expected values.
"""


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


# --- Main execution --------------------------------------------------------

if __name__ == '__main__':
    test_dataset = [[1, 1],
                    [2, 3],
                    [4, 3],
                    [3, 2],
                    [5, 5]]
    test_coefficients = [0.4, 0.8]

    for row in test_dataset:
        y_hat = make_prediction(row, test_coefficients)
        print("Expected = %.3f, Our_Prediction = %.3f" % (row[-1], y_hat))
