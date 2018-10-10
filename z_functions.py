# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com
import numpy as np


class Contours:
    def __init__(self):
        self.boundary  = [[-30, 30], [-30, 30]]
        self.criterion = -2630

    @staticmethod
    def zf(state):
        x = state[0] * 0.1
        y = state[1] * 0.1
        r = np.sqrt(x ** 2 + y ** 2)
        z = np.sin(x ** 2 + 3 * y ** 2) / (0.1 + r ** 2) + (x ** 2 + 5 * y ** 2) * np.exp(1 - r ** 2) / 2
        return -z * 1000


class X2Y2:
    def __init__(self):
        self.boundary  = [[-20, 20], [-20, 20]]
        self.criterion = -1e4

    @staticmethod
    def zf(state):
        return state[0] ** 2 + state[1] ** 2


class Hartman:
    def __init__(self):
        self.boundary  = [[0, 1e3], [0, 1e3], [0, 1e3], [0, 1e3], [0, 1e3], [0, 1e3]]
        self.criterion = 1e8

    @staticmethod
    def zf(state):
        state_zf = state*1e-3
        alpha = np.array([1.0, 1.2, 3.0, 3.2])[:, np.newaxis]
        a_cap = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]])
        p_cap = 1e-4 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ])
        result = 0
        for i in range(4):
            temp = 0
            for j in range(6):
                temp = a_cap[i, j] * (state_zf[j] - p_cap[i, j]) ** 2
            temp *= -1
            result += alpha[i] * np.exp(temp)
        result *= -1
        return result * 1000
