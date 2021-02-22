import time
from random import *
from copy import deepcopy


def time_of_function(function):
    """"Функция-декоратор для замера времени"""
    def wrapped(*args):
        start_time = time.time()
        res = function(*args)
        time_of_work = time.time() - start_time
        print(f"Час виконання - {time_of_work} при діапазоні рандому - {args[0]}")
        return res

    return wrapped


@time_of_function
def main(top_edge):
    # добавил маленькую задержку что-бы можно было разницу увидеть, а то слишком уж быстро даже при больших числах)
    time.sleep(1)
    mat = [[randint(1, top_edge) for _ in range(3)] for _ in range(8)]
    a_mat = [randint(1, top_edge) for _ in range(4)]
    Y = [(a_mat[0] + a_mat[1] * mat[i][0] + a_mat[2] * mat[i][1] + a_mat[3] * mat[i][2]) for i in range(8)]
    x1_mat = [mat[i][0] for i in range(8)]
    x01 = (max(x1_mat) + min(x1_mat)) / 2
    x2_mat = [mat[i][1] for i in range(8)]
    x02 = (max(x2_mat) + min(x2_mat)) / 2
    x3_mat = [mat[i][2] for i in range(8)]
    x03 = (max(x3_mat) + min(x3_mat)) / 2
    x0_mat = [x01, x02, x03, '', '', '', '', '']
    dx1 = x01 - min(x1_mat)
    dx2 = x02 - min(x2_mat)
    dx3 = x03 - min(x3_mat)
    dx_mat = [dx1, dx2, dx3, '', '', '', '', '']
    Norm_mat = [[round((mat[i][j] - x0_mat[j]) / dx_mat[j], 2) for j in range(3)] for i in range(8)]
    Y_et = round(a_mat[0] + a_mat[1] * x01 + a_mat[2] * x02 + a_mat[3] * x03, 2)

    def nearest(lst, target):
        return min(lst, key=lambda x: abs(x - target))

    y_shuk = nearest([y for y in Y if y < Y_et], Y_et)
    x_for_y_shuk = mat[Y.index(y_shuk)]

    all_table = deepcopy(mat)

    for i, y in enumerate(Y):
        all_table[i].append(y)
        all_table[i].append('')

    for i, norm in enumerate(Norm_mat):
        all_table[i].extend(norm)

    all_table.append(x0_mat)
    all_table.append(dx_mat)
    return all_table, y_shuk, x_for_y_shuk


if __name__ == '__main__':
    main(20)
