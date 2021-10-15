from Simplex import Simplex
import numpy as np

if __name__ == '__main__':
    # sym = Simplex(obj_coefs=np.array([4, 0, -6, 9, 0, 0, 0, 0]),
    #               free_coef=-66,
    #               type_of_optimization='min',
    #               A_uneq=None,
    #               b_uneq=None,
    #               uneq_types=None,
    #               A_eq=np.array([[2, 0, -1, 3, 0, 0, 0, 1],
    #                              [0, 1, 0, 3, 0, 0, 0, 0],
    #                              [2, 0, 0, 0, 0, 0, 1, 0],
    #                              [0, 0, 1, 0, 0, 1, 0, 0],
    #                              [-2, 0, 1, -3, 1, 0, 0, 0]]),
    #               b_eq=np.array([1, 2, 6, 6, 2]),
    #               vars_constraints=None,
    #               name='Unnamed')
    sym = Simplex(obj_coefs=np.array([5, 4]),
                  free_coef=0,
                  type_of_optimization='max',
                  A_uneq=np.array([[6, 4],
                                   [1, 2],
                                   [-1, 1],
                                   [0, 1]]),
                  b_uneq=np.array([24, 6, 1, 2]),
                  uneq_types=['<='] * 4,
                  A_eq=None,
                  b_eq=None,
                  vars_constraints=None,
                  name='Reddy Mikkis')
    sym.solve()
