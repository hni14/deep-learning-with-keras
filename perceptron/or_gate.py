# coding: utf-8
import numpy as np


def OR(x1, x2):
    # 入寮
    x = np.array([x1, x2])

    # 重み
    w = np.array([0.5, 0.5])
    
    # バイアス
    b = -0.2

    # 加重和
    tmp = np.sum(w*x) + b

    # 活性化
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = OR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))