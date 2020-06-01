# Created by william from lexueoude.com. 更多正版技术视频讲解，公众号：乐学FinTech
def make_prediction(input_row, coefficients):
    out_put_y_hat = coefficients[0]
    for i in range(len(row) - 1):
        out_put_y_hat += coefficients[i + 1] * input_row[i]
    return out_put_y_hat


test_dataset = [[1, 1],
                [2, 3],
                [4, 3],
                [3, 2],
                [5, 5]]
test_coefficients = [0.4, 0.8]

for row in test_dataset:
    y_hat = make_prediction(row, test_coefficients)
    print("Expected = %.3f, Our_Prediction = %.3f" % (row[-1], y_hat))
