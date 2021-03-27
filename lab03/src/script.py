from random import *
from pprint import pprint
import numpy as np
from math import sqrt
from scipy import linalg
from scipy.stats import t as t_criterium
from scipy.stats import f as f_criterium

class Experiment:
    m, N, d = 3, 4, 4
    mat_1X = [[1, -1, -1, -1],
              [1, -1, 1, 1],
              [1, 1, -1, 1],
              [1, 1, 1, -1]]
    show_info = False

    def __init__(self, x_list):
        """

        :param x_list: список иксов
        """

        """матрица иксов"""
        self.mat_sX = x_list
        self.mat_X = [[self.mat_sX[0][0], self.mat_sX[1][0], self.mat_sX[2][0]],
                      [self.mat_sX[0][0], self.mat_sX[1][1], self.mat_sX[2][1]],
                      [self.mat_sX[0][1], self.mat_sX[1][0], self.mat_sX[2][1]],
                      [self.mat_sX[0][1], self.mat_sX[1][1], self.mat_sX[2][0]]]

        """Т-матрица иксов"""
        self.trans_sx = [list(i) for i in zip(*self.mat_1X)]
        self.trans_x = [list(i) for i in zip(*self.mat_X)]

        self.x_min_max_av = [sum(self.mat_sX[i][k] for i in range(3)) / 3 for k in range(2)]
        self.y_min_max = [int(200 + self.x_min_max_av[i]) for i in range(2)]

        """матрица игреков"""
        self.mat_Y = [[randint(self.y_min_max[0], self.y_min_max[1]) for _ in range(self.m)] for _ in range(self.N)]

    def get_average_y(self):
        """середні значення функцій"""
        return [sum(self.mat_Y[k1]) / self.m for k1 in range(self.N)]

    def get_dispersion(self):
        return [sum([((k1 - self.get_average_y()[j]) ** 2) for k1 in self.mat_Y[j]]) / self.m for j in range(self.N)]

    def show_dets(self, check, b_list):
        print('Матрица Х')
        pprint(self.mat_X)
        print('Матрица У')
        pprint(self.mat_Y)

        print(f"\nУравнение регресии\ny = {b_list[0]} + {b_list[1]}*x1 + {b_list[2]}*x2 + {b_list[3]}*x3")
        print("Дисперсии:")
        print(self.get_dispersion())

        print('Среднее У')
        print(self.get_average_y())
        print("Проверка сравнением со средним Y:\n", check)
        print()

    def get_mx_my(self):
        """Знайдемо коефіцієнти рівняння регресії"""
        mx = [sum(self.mat_X[i][k] for i in range(self.N)) / self.N for k in range(self.m)]
        my = sum(self.get_average_y()) / self.N
        return mx, my

    def get_ai_aii(self):
        ai = [sum(self.trans_x[k][i] * self.get_average_y()[i] for i in range(self.N)) / self.N for k in range(self.m)]
        aii = [sum(self.trans_x[k][i] ** 2 for i in range(self.N)) / self.N for k in range(self.m)]
        return ai, aii

    def find_cf(self):
        tran = self.trans_x
        mx, my = self.get_mx_my()
        ai, aii = self.get_ai_aii()

        a12 = a21 = (tran[0][0] * tran[1][0] + tran[0][1] * tran[1][1] + tran[0][2] * tran[1][2] + tran[0][3] * tran[1][
            3]) / self.N
        a13 = a31 = (tran[0][0] * tran[2][0] + tran[0][1] * tran[2][1] + tran[0][2] * tran[2][2] + tran[0][3] * tran[2][
            3]) / self.N
        a23 = a32 = (tran[1][0] * tran[2][0] + tran[1][1] * tran[2][1] + tran[1][2] * tran[2][2] + tran[1][3] * tran[2][
            3]) / self.N

        # Використовую об'єкт linalg та його метод det з бібліотеки scipy для знаходження к-ів рівняння

        znamen = linalg.det(
            [[1, mx[0], mx[1], mx[2]],
             [mx[0], aii[0], a12, a13],
             [mx[1], a12, aii[1], a32],
             [mx[2], a13, a23, aii[2]]])

        b0 = linalg.det([[my, mx[0], mx[1], mx[2]],
                         [ai[0], aii[0], a12, a13],
                         [ai[1], a12, aii[1], a32],
                         [ai[2], a13, a23, aii[2]]]) / znamen

        b1 = linalg.det([[1, my, mx[1], mx[2]],
                         [mx[0], ai[0], a12, a13],
                         [mx[1], ai[1], aii[1], a32],
                         [mx[2], ai[2], a23, aii[2]]]) / znamen

        b2 = linalg.det([[1, mx[0], my, mx[2]],
                         [mx[0], aii[0], ai[0], a13],
                         [mx[1], a12, ai[1], a32],
                         [mx[2], a13, ai[2], aii[2]]]) / znamen

        b3 = linalg.det([[1, mx[0], mx[1], my],
                         [mx[0], aii[0], a12, ai[0]],
                         [mx[1], a12, aii[1], ai[1]],
                         [mx[2], a13, a23, ai[2]]]) / znamen

        check = [b0 + b1 * tran[0][i] + b2 * tran[1][i] + b3 * tran[2][i] for i in range(4)]
        b_list = [b0, b1, b2, b3]
        return check, b_list

    def make_experiment(self):
        f1 = self.m-1
        f2 = self.N
        f3 = f1*f2
        f4 = self.N - self.d

        mat_disY = self.get_dispersion()
        check, b_list = self.find_cf()
        if self.show_info:
            self.show_dets(check, b_list)

        print('\nПроверка однородности за Кохрена:')
        if max(mat_disY) / sum(mat_disY) < 0.7679:
            print('Дисперсия однородная - ', max(mat_disY) / sum(mat_disY))
        else:
            print('Дисперсия неоднородная - ', max(mat_disY) / sum(mat_disY))

        print('\nПроверка значимости:\n')
        S2b = sum(mat_disY) / self.N
        S2bs = S2b / (self.m * self.N)
        Sbs = sqrt(S2bs)
        bb = [sum(self.get_average_y()[k] * self.trans_sx[i][k] for k in range(self.N)) / self.N for i in range(self.N)]
        t = [abs(bb[i]) / Sbs for i in range(self.N)]
        for i in range(self.N):
            if t[i] < t_criterium.ppf(q=0.975, df=f3):
                print('Незначительный ', b_list[i])
                b_list[i] = 0
                self.d -= 1

        y_reg = [b_list[0] + b_list[1] * self.mat_X[i][0] + b_list[2] * self.mat_X[i][1] + b_list[3] * self.mat_X[i][2]
                 for i in range(self.N)]
        print('Значения у:\n')
        [print(
            f"{b_list[0]} + {b_list[1]}*x1 + {b_list[2]}*x2 + {b_list[3]}*x3 = {b_list[0] + b_list[1] * self.mat_X[i][0] + b_list[2] * self.mat_X[i][1] + b_list[3] * self.mat_X[i][2]}")
            for i in range(self.N)]

        print('\nПроверка адекватности за Фишера:\n')
        Sad = (self.m / (self.N - self.d)) * int(sum(y_reg[i] - self.get_average_y()[i] for i in range(self.N)) ** 2)
        Fp = Sad / S2b
        q = 0.05
        F_table = f_criterium.ppf(q=1-q, dfn=f4, dfd=f3)
        print('FP  =', Fp)
        if Fp > F_table:
            print('Неадекватно при 0.05')
        else:
            print('Адекватно при 0.05')
