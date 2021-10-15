import numpy as np
from copy import deepcopy
from itertools import permutations


class Simplex:
    def __init__(self, obj_coefs, free_coef=0,
                 type_of_optimization='min',
                 A_uneq=None, b_uneq=None,
                 uneq_types=None, A_eq=None,
                 b_eq=None, vars_constraints=None,
                 name='Unnamed'):

        assert type_of_optimization in ['min', 'max'], "\nInvalid type of optimization!\nMust be 'min' or 'max'."
        assert A_uneq is not None or A_eq is not None, 'Please, enter uneq or eq coefficients'
        assert A_uneq is None or len(A_uneq) == len(b_uneq) == len(uneq_types)
        assert A_eq is None or len(A_eq) == len(b_eq)

        self.name = name
        self.obj_coefs = np.array(obj_coefs)
        self.free_coef = free_coef

        # canonized form requires min
        if type_of_optimization == 'max':
            self.obj_coefs *= -1
            self.free_coef *= -1

        if A_uneq is not None:
            if isinstance(A_uneq[0], int) or isinstance(A_uneq[0], float):
                A_uneq = [A_uneq]
            self.A_uneq = np.array(A_uneq)
            self.b_uneq = np.array(b_uneq)
            self.uneq_types = uneq_types
        else:
            self.A_uneq = None

        if A_eq is not None:
            self.A_eq = np.array(A_eq)
            self.b_eq = np.array(b_eq)
        else:
            self.A_eq = None
        self.vars_constraints = vars_constraints

    def _deal_with_arbitrary_vars(self):
        # canonized form requires non-negative vars
        self.arbitrary_vars_map = dict()
        k = len(self.vars_constraints)
        for i, var_constr in self.vars_constraints:
            if var_constr == '<=':
                self.arbitrary_vars_map[i] = [i]
                if self.A_eq is not None:
                    self.A_eq[:, i] *= -1
                if self.A_uneq is not None:
                    self.A_uneq[:, i] *= -1
                self.obj_coefs[i] *= -1
            elif var_constr == 'arb':
                self.arbitrary_vars_map[i] = [i, k]
                if self.A_eq is not None:
                    self.A_eq = np.hstack((self.A_eq, -self.A_eq[:, i]))
                if self.A_uneq is not None:
                    self.A_uneq = np.hstack((self.A_uneq, -self.A_uneq[:, i]))
                self.obj_coefs = np.hstack((self.obj_coefs, -self.obj_coefs[i]))
                k += 1
        return self

    def _transform_uneq_to_eq(self):
        # canonized form requires eqs
        new_coefs = np.diag(np.where(np.array(self.uneq_types) == '<=', 1, -1))
        self.num_base_init = len(new_coefs)
        self.A_uneq = np.hstack((self.A_uneq, new_coefs))
        if self.A_eq is not None:
            self.A_eq = np.hstack((self.A_eq, np.zeros(len(self.A_eq), new_coefs.shape[1])))
        self.obj_coefs = np.hstack((self.obj_coefs, np.zeros((new_coefs.shape[1],))))
        if self.A_eq is not None:
            self.A = np.vstack((self.A_uneq, self.A_eq))
            self.b = np.vstack((self.b_uneq, self.b_eq))
        else:
            self.A = self.A_uneq
            self.b = self.b_uneq
        return self

    def _deal_with_neg_b(self):
        # canonized form requires positive right part of eqs
        mask = self.b < 0
        self.b = np.where(self.b >= 0, self.b, -self.b).ravel()
        self.A[mask] *= -1
        return self

    def _canonize(self):
        if self.vars_constraints is not None:
            self._deal_with_arbitrary_vars()

        if self.A_uneq is not None:
            self._transform_uneq_to_eq()
        else:
            self.A = self.A_eq
            self.b = self.b_eq

        if (self.b < 0).any():
            self._deal_with_neg_b()
        self.var_inds = np.arange(self.A.shape[1])
        return self

    @staticmethod
    def simplex_step(i_col, j_row, A, p_row, b_col, Q0):
        pivot_col = np.argmin(p_row)

        if p_row[pivot_col] < 0:
            if np.max(A[:, pivot_col]) < 0:
                return 0
            else:
                # pos_a_ij = A[A[:, pivot_col] > 0, pivot_col]
                b_a = np.argsort(b_col / pivot_col)
                for i in b_a:
                    if A[i, pivot_col] > 0:
                        pivot_row = i
                        break
                else:
                    print('В разрешающем столбце нет положительных элементов')
                    return 0
                # pivot_row = np.where(np.isclose(A[:, pivot_col], pivot_a))[0][0]
                pivot_a = A[pivot_row, pivot_col]
                new_i_col = deepcopy(i_col)
                new_j_row = deepcopy(j_row)

                # 1 step
                new_i_col[pivot_row] = j_row[pivot_col]
                new_j_row[pivot_col] = i_col[pivot_row]

                # 2 step
                pivot_a_hat = 1 / pivot_a
                a_col_pivot_hat = -A[:, pivot_col] * pivot_a_hat
                pivot_p_hat = - p_row[pivot_col] * pivot_a_hat

                # 3 step
                a_row_pivot_hat = A[pivot_row] * pivot_a_hat
                pivot_b_hat = b_col[pivot_row] * pivot_a_hat

                # 4 step
                new_A = A - A[:, pivot_col].reshape(-1, 1).dot(a_row_pivot_hat.reshape(1, -1))
                new_A[:, pivot_col] = a_col_pivot_hat
                new_A[pivot_row] = a_row_pivot_hat
                new_A[pivot_row, pivot_col] = pivot_a_hat

                # 5 step
                new_p_row = p_row - a_row_pivot_hat * p_row[pivot_col]
                new_p_row[pivot_col] = pivot_p_hat

                new_b_col = b_col - pivot_b_hat * A[:, pivot_col]
                new_b_col[pivot_row] = pivot_b_hat

                new_Q0 = Q0 - pivot_b_hat * p_row[pivot_col]

                return new_i_col, new_j_row, new_A, new_p_row, new_b_col, new_Q0
        else:
            return 1

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

        A_new = deepcopy(self.A).astype(float)
        b_new = deepcopy(self.b).astype(float)

        # выделяем единичный базис
        cols_perm = permutations(indep_cols, len(indep_cols))

        for cols_order in cols_perm:
            next_comb = False
            for k, row in enumerate(A_new): # шаг метода Гаусса
                if row[cols_order[k]] != 0: # в диагонали стоит не 0
                    rep_row = np.repeat(row.reshape(1, -1), A_new.shape[0], axis=0)
                    rep_row[k] = 0

                    bb = np.array([b_new[k]] * len(b_new))
                    bb[k] = 0

                    mul = (A_new[:, cols_order[k]] / row[cols_order[k]])

                    A_new -= rep_row * mul.reshape(-1, 1)
                    A_new[k] /= row[cols_order[k]]

                    b_new -= bb * mul
                    b_new[k] /= row[cols_order[k]]
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
            print("Can't extract basis")
            return 0

        self.A = A_new
        self.b = b_new

        # Transform obj_fun
        for j, i in zip(indep_cols, base_rows):
            self.obj_coefs -= self.obj_coefs[j] * self.A[i]
            self.free_coef += self.b[i]

        return indep_cols, base_rows

    def solve(self, num_iterations=100):

        self._canonize()
        rank_A = np.linalg.matrix_rank(self.A, tol=1e-12)
        diff = self.A.shape[0] - rank_A

        if diff == 0:
            # Basis can be extracted
            i_col, base_rows = self.transform_to_base()

            j_row = np.array([j for j in range(self.A.shape[1]) if j not in i_col]).ravel()
            A = self.A[:, j_row][base_rows]
            i_col = np.array(i_col)
            p_row = self.obj_coefs[j_row]
            print(self.b)
            print(base_rows)
            b_col = self.b[base_rows]
            Q0 = self.free_coef
            print(f'i_col: {i_col + 1}')
            print(f'j_row: {j_row + 1}')
            print(f'A: {A}')
            print(f'p_row: {p_row}')
            print(f'b_col: {b_col}')
            print(f'Q0: {Q0}')
            for i in range(num_iterations):
                print(f'Iteration {i}')
                res = self.simplex_step(i_col, j_row, A, p_row, b_col, Q0)
                if res == 0:
                    print("No solutions")
                    break
                elif res == 1:
                    break
                else:
                    i_col, j_row, A, p_row, b_col, Q0 = res
                    print(f'i_col: {i_col + 1}')
                    print(f'j_row: {j_row + 1}')
                    print(f'A: {A}')
                    print(f'p_row: {p_row}')
                    print(f'b_col: {b_col}')
                    print(f'Q0: {Q0}')
        else:
            print('Пошел нахуй')
            print(diff)
            print(rank_A)
            print(self.A)
        return self
