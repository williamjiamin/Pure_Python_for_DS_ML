# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号：乐学FinTech


def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0


def opt_weights(train, learning_rate, how_many_epoch):
    weights = [0.0 for i in range(len(train[0]))]
    for epoch in range(how_many_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error ** 2
            weights[0] = weights[0] + learning_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] = weights[i + 1] + learning_rate * error * row[i]
        print('This is epoch: %d, our learning_rate is : %.4f, the error is : %.4f' % (epoch, learning_rate, sum_error))
    return weights


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

how_many_epoch = 200

weights = opt_weights(dataset, learning_rate, how_many_epoch)

print(weights)
