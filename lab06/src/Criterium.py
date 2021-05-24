import time
from scipy.stats import f


def time_of_function(function):
    """"Функция-декоратор для замера времени"""
    def wrapped(*args):
        start_time = time.perf_counter()
        res = function(*args)
        time_of_work = time.perf_counter() - start_time
        print(f"Час виконання - {time_of_work}")
        return res

    return wrapped


class Criteria:
    def __init__(self, x, y, n, m):
        self.x = x
        self.y = y

        self.n = n
        self.m = m
        self.f1 = self.m - 1
        self.f2 = self.n
        self.f3 = self.f1 * self.f2
        self.q = 0.05
        self.q1 = self.q / self.f1

    def s_kv(self, y_aver):
        res = []
        for i in range(self.n):
            s = sum([(y_aver[i] - self.y[i][j]) ** 2 for j in range(self.m)]) / self.m
            res.append(round(s, 3))
        return res

    @time_of_function
    def criteria_cochrana(self, y_aver):
        S_kv = self.s_kv(y_aver)
        Gp = max(S_kv) / sum(S_kv)
        return Gp

    def cohren(self):
        fisher_value = f.ppf(q=1 - self.q1, dfn=self.f2, dfd=(self.f1 - 1) * self.f2)
        return fisher_value / (fisher_value + self.f1 - 1)

    def bs(self, x, y_aver):
        res = [sum(y_aver) / self.n]

        for i in range(len(x[0])):
            b = sum(j[0] * j[1] for j in zip(x[:, i], y_aver)) / self.n
            res.append(b)

        return res

    @time_of_function
    def criteria_studenta(self, x, y_aver):
        S_kv = self.s_kv(y_aver)
        s_kv_aver = sum(S_kv) / self.n

        s_Bs = (s_kv_aver / self.n / self.m) ** 0.5
        Bs = self.bs(x, y_aver)
        ts = [round(abs(B) / s_Bs, 3) for B in Bs]

        return ts

    @time_of_function
    def criteria_fishera(self, y_aver, y_new, d):
        S_ad = self.m / (self.n - d) * sum([(y_new[i] - y_aver[i]) ** 2 for i in range(len(self.y))])
        S_kv = self.s_kv(y_aver)
        S_kv_aver = sum(S_kv) / self.n

        return S_ad / S_kv_aver
