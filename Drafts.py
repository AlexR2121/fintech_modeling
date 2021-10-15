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


if __name__ == '__main__':
    print(np.linalg.matrix_rank(A))
    kek()
