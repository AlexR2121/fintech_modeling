import numpy as np
from itertools import permutations
from copy import deepcopy

A = np.array([[1, 1, 0, 3, 4],
              [0, 5, 0, 7, 8],
              [0, 9, 1, 11, 12]])
b = np.array([1, 2, 3])


def kek(A=A, b=b):
    indep_cols = []
    base_rows = []
    for i in range(A.shape[1]):
        if (A[:, i] != 0).sum() == 1:
            br = np.where(np.array(A[:, i]) != 0)[0][0]
            if br not in base_rows:
                base_rows.append(br)
                A[br] = A[br] / A[br, i]
                indep_cols.append(i)
        if len(indep_cols) == A.shape[0]:
            return indep_cols, base_rows
    if len(indep_cols) == 0:
        indep_cols = [A.shape[1] - 1]
    cols = deepcopy(indep_cols)

    r = len(indep_cols)

    for col in range(A.shape[-1] - 1, -1, -1):
        if col not in cols:
            cols.append(col)
            r_new = np.linalg.matrix_rank(A[:, cols])
            if r_new == r + 1:
                indep_cols.append(col)
                r = r_new
            if r == A.shape[0]:
                break
        else:
            print('{eq')

    A_new = deepcopy(A).astype(float)
    b_new = deepcopy(b).astype(float)

    cols_perm = permutations(indep_cols, len(indep_cols))

    for cols_order in cols_perm:
        next_comb = False
        # for i, j in enumerate(cols_order):
        #     base_col = A_new[:, j]
        #     if base_col[i] == 0:
        #         next_comb = True
        #         break
        for k, row in enumerate(A_new):
            print(k)
            if row[cols_order[k]] != 0:
                rep_row = np.repeat(row.reshape(1, -1), A_new.shape[0], axis=0)
                rep_row[k] = 0
                bb = np.array([b_new[k]] * len(b_new))
                bb[k] = 0
                print(rep_row)
                print(A_new[:, cols_order[k]])
                print(row[cols_order[k]])
                mul = (A_new[:, cols_order[k]] / row[cols_order[k]])
                A_new -= rep_row * mul.reshape(-1, 1)
                A_new[k] /= row[cols_order[k]]

                b_new -= bb * mul
                b_new[k] /= row[cols_order[k]]

                print(A_new)
                print(b_new)
            else:
                next_comb = True
                break
        if (b_new < 0).any() or next_comb:
            continue
        else:
            break
    else:
        print("No solutions")
    def transform_to_base(self):

        indep_cols = []
        base_rows = []

        # смотрим, есть ли столбцы с единицей
        for i in range(self.A.shape[1]):
            if (self.A[:, i] != 0).sum() == 1:
                br = np.where(np.array(self.A[:, i]) != 0)[0][0]
                # проверяем, что новый столбец не содержит выделенную раньше переменную и
                # что коэффициент при переменной больше нуля
                if br not in base_rows and self.A[br, i] > 0:
                    base_rows.append(br)
                    self.A[br] = self.A[br] / self.A[br, i]
                    self.b[br] = self.b[br] / self.A[br, i]
                    indep_cols.append(i)
            # проверяем, совпадает ли количество выделенных столбцов
            # с количеством строк в А
            if len(indep_cols) == self.A.shape[0]:
                for j, i in zip(indep_cols, base_rows):
                    self.obj_coefs -= (self.obj_coefs[j] * self.A[i]).astype(float)
                    self.free_coef -= self.obj_coefs[j] * self.b[i]
                return indep_cols, base_rows
        # если не выделилось ни одного столбца с единицей
        if len(indep_cols) == 0:
            indep_cols = [self.A.shape[1] - 1]
        cols = deepcopy(indep_cols)

        r = len(indep_cols)

        # стакая столбцы, находим m независимых
        for col in range(self.A.shape[-1] - 1, -1, -1):
            if col not in cols:
                cols.append(col)
                r_new = np.linalg.matrix_rank(self.A[:, cols])
                if r_new == r + 1:
                    indep_cols.append(col)
                    r = r_new
                if r == self.A.shape[0]:
                    break

        # выделяем единичный базис
        cols_perm = permutations(indep_cols, len(indep_cols))

        for cols_order in cols_perm:
            A_new = deepcopy(self.A).astype(float)
            b_new = deepcopy(self.b).astype(float)

            next_comb = False

            for k, row in enumerate(A_new): # шаг метода Гаусса
                if row[cols_order[k]] != 0: # в диагонали стоит не 0

                    rep_row = np.repeat(row.reshape(1, -1), A_new.shape[0], axis=0)
                    rep_row[k] = 0
                    print(f'rep_row {rep_row}')

                    bb = np.array([b_new[k]] * len(b_new))
                    bb[k] = 0

                    mul = deepcopy((A_new[:, cols_order[k]] / row[cols_order[k]]))
                    mul_k = deepcopy(row[cols_order[k]])

                    A_new[k] /= mul_k
                    A_new -= rep_row * mul.reshape(-1, 1)

                    b_new[k] /= mul_k
                    b_new -= bb * mul

                else:
                    next_comb = True
                    break
            if (b_new < 0).any() or next_comb: # проверяем, не получилось ли отрицательных b
                continue
            else: # если все в порядке
                indep_cols = cols_order
                base_rows = tuple(range(self.A.shape[0]))
                break
        else:

            return 0

        self.A = A_new
        self.b = b_new
        print(self.A)
        print(self.b)
        # Transform obj_fun
        for j, i in zip(indep_cols, base_rows):
            self.obj_coefs -= (self.obj_coefs[j] * self.A[i]).astype(float)
            self.free_coef -= self.obj_coefs[j] * self.b[i]
        print(self.obj_coefs)
        print(self.free_coef)
        return indep_cols, np.array(base_rows)

# if __name__ == '__main__':
#     print(np.linalg.matrix_rank(A))
#     kek()
