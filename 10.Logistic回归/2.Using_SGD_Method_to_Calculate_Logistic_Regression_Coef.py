# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号：乐学FinTech

from math import exp


def prediction(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row) - 1):
        yhat += coefficients[i + 1] * row[i]
    return 1 / (1 + exp(-yhat))


def using_sgd_method_to_calculate_coefficients(training_dataset, learning_rate, n_times_epoch):
    coefficients = [0.0 for i in range(len(training_dataset[0]))]
    for epoch in range(n_times_epoch):
        the_sum_of_error = 0
        for row in training_dataset:
            y_hat = prediction(row, coefficients)
            error = row[-1] - y_hat
            the_sum_of_error += error ** 2
            coefficients[0] = coefficients[0] + learning_rate * error * y_hat * (1.0 - y_hat)
            for i in range(len(row) - 1):
                coefficients[i + 1] = coefficients[i + 1] + learning_rate * error * y_hat * (1.0 - y_hat) * row[i]
        print("This is epoch 【%d】,the learning rate we are using is 【%.3f】,the error is 【%.3f】" % (
            epoch, learning_rate, the_sum_of_error))
    return coefficients


dataset = [[2, 2, 0],
           [2, 4, 0],
           [3, 3, 0],
           [4, 5, 0],
           [8, 1, 1],
           [8.5, 3.5, 1],
           [9, 1, 1],
           [10, 4, 1]]

learning_rate = 0.1

n_times_epoch = 1000

coef = using_sgd_method_to_calculate_coefficients(dataset, learning_rate, n_times_epoch)

print(coef)
