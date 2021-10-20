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
        self.obj_coefs = np.array(obj_coefs).astype(float)
        self.free_coef = free_coef
        self.type_of_optimization = type_of_optimization

        # canonized form requires min
        if self.type_of_optimization == 'max':
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
        self.initial_vars = list(range(len(obj_coefs)))
        self.arbitrary_vars_map = dict()

    def _deal_with_arbitrary_vars(self):
        # canonized form requires non-negative vars
        k = len(self.vars_constraints)
        for i, var_constr in enumerate(self.vars_constraints):
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
                    self.A_eq = np.hstack((self.A_eq, -self.A_eq[:, i].reshape(-1, 1)))
                if self.A_uneq is not None:
                    self.A_uneq = np.hstack((self.A_uneq, -self.A_uneq[:, i].reshape(-1, 1)))
                new_obj = np.empty((len(self.obj_coefs)+1,))
                new_obj[:len(self.obj_coefs)] = self.obj_coefs
                new_obj[-1] = -self.obj_coefs[i]
                self.obj_coefs = new_obj
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

    def _deal_with_arts_in_basis(self, i_col, A, b_col, art):
        arts_in_basis = np.in1d(i_col, art)
        all_zeros = np.all(A == 0, axis=0)
        z_mask = np.bitwise_and(arts_in_basis, all_zeros)
        if z_mask.any():
            A = A[z_mask]
            b_col = b_col[z_mask]
            i_col = i_col[z_mask]
        if np.in1d(i_col, art).any():
            for i, row in enumerate(A):
                if not np.in1d(i_col, art)[i]:
                    continue
                else:
                    art_pivot_coords = (i, np.arange(len(row))[row!=0][0])
        else:
            art_pivot_coords = 0
        return A, b_col, i_col, art_pivot_coords

    def simplex_step(self, i_col, j_row, A, p_row, b_col, Q0, art = None, pivot_elem = None):

        pivot_col = np.argmin(p_row)

        if p_row[pivot_col] < 0 or pivot_elem is not None: # проверяем закончено ли обновление или случай IIб
            if np.max(A[:, pivot_col]) < 0 and pivot_elem is None:
                return i_col, j_row, A, p_row, b_col, Q0, 0, None
            else:
                if pivot_elem is None: # проверяем не случай ли IIб
                    b_a = np.argsort(b_col / A[:, pivot_col])
                    for i in b_a:
                        # if b_col[i] == 0: #вырожденное решение
                        #     pivot_row = i
                        #     break
                        if A[i, pivot_col] > 0:
                            pivot_row = i
                            break
                    else:
                        print('В разрешающем столбце нет положительных элементов')
                        return i_col, j_row, A, p_row, b_col, Q0, 0, None
                else:
                    pivot_row, pivot_col = pivot_elem

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

                if art is not None:
                    mask = ~np.in1d(new_j_row, art)
                    new_A = new_A[:, mask]
                    new_p_row = new_p_row[mask]
                    new_j_row = new_j_row[mask]
                    if np.isclose(new_Q0, 0) and not np.in1d(new_i_col, art).any():
                        return new_i_col, new_j_row, new_A, new_p_row, new_b_col, new_Q0, 1, None
                    elif np.isclose(new_Q0, 0) and np.in1d(new_i_col, art).any():
                        new_A, new_b_col, new_i_col, art_pivot_coords = self._deal_with_arts_in_basis(i_col, A, b_col, art)
                        return new_i_col, new_j_row, new_A, new_p_row, new_b_col, new_Q0, None, art_pivot_coords
                return new_i_col, new_j_row, new_A, new_p_row, new_b_col, new_Q0, None, None
        elif p_row[pivot_col] >= 0 and art is not None:
            if Q0 < 0:
                return i_col, j_row, A, p_row, b_col, Q0, 0, None
        else:
            return i_col, j_row, A, p_row, b_col, Q0, 1, None

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
                    self.b[br] = self.b[br] / self.A[br, i]
                    self.A[br] /= self.A[br, i]
                    indep_cols.append(i)
            # проверяем, совпадает ли количество выделенных столбцов
            # с количеством строк в А
            if len(indep_cols) == self.A.shape[0]:
                for j, i in zip(indep_cols, base_rows):
                    self.free_coef += self.obj_coefs[j] * self.b[i]
                    self.obj_coefs -= (self.obj_coefs[j] * self.A[i]).astype(float)

                return indep_cols, base_rows
        else:
            print("Нельзя сходу выделить единичный базис")
            return 0, 0

    def debug_print(self, i_col, j_row, A, p_row, b_col, Q0):
        print(f'i_col: {i_col}')
        print(f'j_row: {j_row}')
        print(f'A: {A}')
        print(f'p_row: {p_row}')
        print(f'b_col: {b_col}')
        print(f'-Q0: {Q0}')
        print('\n')

    def add_artificial_vars(self):
        A_art = np.hstack((self.A, np.eye(self.A.shape[0])))
        g_fun = -np.sum(self.A, axis=0)
        g0 = -self.b.sum()
        return A_art, g_fun, g0

    def solve(self, num_iterations=100):

        self._canonize()
        self.A = self.A.astype(float)
        self.b = self.b.astype(float)
        i_col, base_rows = self.transform_to_base()

        if i_col == 0:
            print('Решаем вспомогательную задачу\n')
            A_art, g, g0 = self.add_artificial_vars()
            i_col = np.arange(self.A.shape[1], A_art.shape[1])
            j_row = np.arange(self.A.shape[1])
            A = self.A
            p_row = g
            b_col = self.b
            Q0 = g0
            arts = np.arange(self.A.shape[0])+self.A.shape[1]
            art_piv_elem = None
            print('Начальные данные:')
            self.debug_print(i_col, j_row, A, p_row, b_col, Q0)
            for i in range(num_iterations):
                print(f'Iteration {i}')
                res = self.simplex_step(i_col, j_row, A, p_row, b_col, Q0, arts, art_piv_elem)
                i_col, j_row, A, p_row, b_col, Q0, out, art_piv_elem = res
                self.debug_print(i_col, j_row, A, p_row, b_col, Q0)
                if out == 0:
                    return "No solutions"
                elif out == 1:
                    b_count = 0
                    new_A = np.empty(self.A.shape)
                    new_A[:, j_row] = A
                    eye = np.eye(self.A.shape[0])
                    for col in range(self.A.shape[1]):
                        if col in i_col:
                            new_A[:, col] = eye[:, np.where(i_col==col)[0][0]]
                            b_count+=1
                    for i, j in enumerate(i_col):
                        self.free_coef += self.obj_coefs[j] * b_col[i]
                        self.obj_coefs -= (self.obj_coefs[j] * new_A[i]).astype(float)
                    break

        else:
            j_row = np.array([j for j in range(self.A.shape[1]) if j not in i_col]).ravel()
            A = self.A[:, j_row][base_rows]
            i_col = np.array(i_col)
            b_col = self.b[base_rows]
        print('Решаем основную задачу')
        p_row = self.obj_coefs[j_row]
        Q0 = -self.free_coef
        print('Начальные данные:')
        self.debug_print(i_col, j_row, A, p_row, b_col, Q0)
        for i in range(num_iterations):
            print(f'Iteration {i}')
            res = self.simplex_step(i_col, j_row, A, p_row, b_col, Q0)
            i_col, j_row, A, p_row, b_col, Q0, out, art_piv_elem = res
            self.debug_print(i_col, j_row, A, p_row, b_col, Q0)
            if res[-2] == 0:
                return "Нет решений"
                break
            elif res[-2] == 1:
                solution_dict = dict()
                for j in j_row:
                    solution_dict[f'x{j}'] = 0
                for b, i in enumerate(i_col):
                    solution_dict[f'x{i}'] = round(b_col[b], 2)
                if self.arbitrary_vars_map:
                    for k, v in self.arbitrary_vars_map.items():
                        if len(v) == 1:
                            solution_dict[f'x{k}'] = -solution_dict[f'x{k}']
                        else:
                            solution_dict[f'x{v[0]}'] -= round(solution_dict[f'x{v[1]}'],2)
                fin_vars = deepcopy(list(solution_dict.keys()))
                for k in fin_vars:
                    num = int(k[1:])
                    if num not in self.initial_vars:
                        del solution_dict[f'x{num}']

                solution_dict =  dict(sorted(solution_dict.items(), key = lambda x: int(x[0][1:])))
                if self.type_of_optimization == 'max':
                    solution_dict['Q_opt'] = round(Q0, 2)
                else:
                    solution_dict['Q_opt'] = -round(Q0, 2)
                return solution_dict

        return "Количество итераций превысило максимум"
