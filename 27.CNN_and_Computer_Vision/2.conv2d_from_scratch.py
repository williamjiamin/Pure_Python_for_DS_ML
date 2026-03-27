from __future__ import annotations

from typing import List


Matrix = List[List[float]]


def conv2d_valid(image: Matrix, kernel: Matrix) -> Matrix:
    h, w = len(image), len(image[0])
    kh, kw = len(kernel), len(kernel[0])
    out_h, out_w = h - kh + 1, w - kw + 1
    out = [[0.0 for _ in range(out_w)] for _ in range(out_h)]

    for i in range(out_h):
        for j in range(out_w):
            s = 0.0
            for ki in range(kh):
                for kj in range(kw):
                    s += image[i + ki][j + kj] * kernel[ki][kj]
            out[i][j] = s
    return out


def relu(m: Matrix) -> Matrix:
    return [[max(0.0, x) for x in row] for row in m]


if __name__ == "__main__":
    image = [
        [1, 2, 3, 2, 1],
        [4, 5, 6, 5, 4],
        [7, 8, 9, 8, 7],
        [4, 5, 6, 5, 4],
        [1, 2, 3, 2, 1],
    ]
    sobel_x = [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ]

    y = conv2d_valid(image, sobel_x)
    y = relu(y)
    print("Feature map:")
    for row in y:
        print([round(v, 1) for v in row])
