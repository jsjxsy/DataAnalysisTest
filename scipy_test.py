# encoding=utf-8
import numpy as np

from scipy.integrate import quad, dblquad, nquad  # dbquad 是二元积分，nquad是n维积分


def main():
    print(quad(lambda x: np.exp(-x), 0, np.inf))
    # 在0 - 正无穷的积分
    print(dblquad(lambda t, x: np.exp(-x * t) / t ** 3, 0, np.inf, lambda x: 1, lambda x: np.inf))
    # 先定义 t 的范围再定义 x 的范围，他实际上是 t 的函数
    pass

if __name__ == '__main__':
        main()
