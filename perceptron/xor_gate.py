# coding: utf-8
from and_gate import AND
from or_gate import OR
from nand_gate import NAND


def XOR(x1, x2):
    # 単純パーセプトロン
    s1 = NAND(x1, x2)

    # 単純パーセプトロン
    s2 = OR(x1, x2)

    # ３つの単純パーセプトロン(NAND, OR, AND)を組み合わせる
    y = AND(s1, s2)
    return y

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))