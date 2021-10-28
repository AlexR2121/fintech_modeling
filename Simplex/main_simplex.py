from simplex import Simplex
import numpy as np

if __name__ == '__main__':
    sym = Simplex(obj_coefs=np.array([4, 0, -6, 9, 0, 0, 0, 0]),
                  free_coef=66,
                  type_of_optimization='min',
                  A_uneq=None,
                  b_uneq=None,
                  uneq_types=None,
                  A_eq=np.array([[2, 0, -1, 3, 0, 0, 0, 1],
                                 [0, 1, 0, 3, 0, 0, 0, 0],
                                 [2, 0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 1, 0, 0, 1, 0, 0],
                                 [-2, 0, 1, -3, 1, 0, 0, 0]]),
                  b_eq=np.array([1, 2, 6, 6, 2]),
                  vars_constraints=None,
                  name='bron1')
    # sym = Simplex(obj_coefs=np.array([5, 4]),
    #               free_coef=0,
    #               type_of_optimization='max',
    #               A_uneq=np.array([[6, 4],
    #                                [1, 2],
    #                                [-1, 1],
    #                                [0, 1]]),
    #               b_uneq=np.array([24, 6, 1, 2]),
    #               uneq_types=['<='] * 4,
    #               A_eq=None,
    #               b_eq=None,
    #               vars_constraints=None,
    #               name='Reddy Mikkis')
    # sym = Simplex(obj_coefs=np.array([-2, -3, -4]),
    #               free_coef=0,
    #               type_of_optimization='min',
    #               A_uneq=np.array([[3, 2, 1],
    #                                [2, 5, 3]]),
    #               b_uneq=np.array([10, 15]),
    #               uneq_types=['<='] * 2,
    #               A_eq=None,
    #               b_eq=None,
    #               vars_constraints=None,
    #               name='Wiki')
    # sym = Simplex(obj_coefs=np.array([2, 1, -1, 3, -1]),
    #               free_coef=0,
    #               type_of_optimization='min',
    #               A_uneq=None,
    #               b_uneq=None,
    #               uneq_types=None,
    #               A_eq=np.array([[3, 0, 2, 0, -1],
    #                              [1, -1, 1, 0, 0],
    #                              [1, 0, 1, 1, 0]]),
    #               b_eq=np.array([12, 5, 6]),
    #               vars_constraints=None,
    #               name='bron2')
    # sym = Simplex(obj_coefs=np.array([-2, -3, -4]),
    #               free_coef=0,
    #               type_of_optimization='min',
    #               A_uneq=None,
    #               b_uneq=None,
    #               uneq_types=None,
    #               A_eq=np.array([[3, 2, 1],
    #                                [2, 5, 3]]),
    #               b_eq=np.array([10, 15]),
    #               vars_constraints=['arb', '>=', 'arb'],
    #               name = 'Wiki2')
    # sym = Simplex(obj_coefs=np.array([-3, -4]),
    #               free_coef=0,
    #               type_of_optimization='min',
    #               A_uneq=np.array([[1, 0],
    #                                [0, 2],
    #                                [1, 1],
    #                                [-1, 4]]),
    #               b_uneq=np.array([10, 5, 20, 20]),
    #               uneq_types=['>=','>=','<=','<='],
    #               A_eq=None,
    #               b_eq=None,
    #               vars_constraints=None,
    #               name='9')
    # sym = Simplex(obj_coefs=np.array([-3, 1]),
    #               free_coef=0,
    #               type_of_optimization='min',
    #               A_uneq=np.array([[2, -1],
    #                                [1, -2],
    #                                [1, 1]]),
    #               b_uneq=np.array([4, 2, 5]),
    #               uneq_types=['<=', '<=', '<='],
    #               A_eq=None,
    #               b_eq=None,
    #               vars_constraints=['arb', 'arb'],
    #               name='13')
    # sym = Simplex(obj_coefs=np.array([3, -2, 4]),
    #               free_coef=22,
    #               type_of_optimization='max',
    #               A_uneq=np.array([[1, 1, 3],
    #                                [2, 4, 0],
    #                                [1, 2, 0]]),
    #               b_uneq=np.array([12, 14, 6]),
    #               uneq_types=['<=', '<=', '<='],
    #               A_eq=None,
    #               b_eq=None,
    #               vars_constraints=None,
    #               name='Miro')
    sym = Simplex(obj_coefs=np.array([60, 40]),
                  free_coef=0,
                  type_of_optimization='min',
                  A_uneq=np.array([[10, 40],
                                   [10, 20],
                                   [10, 0]]),
                  b_uneq=np.array([10,40,10]),
                  uneq_types=['>=','>=','>='],
                  A_eq=None,
                  b_eq=None,
                  vars_constraints=None,
                  name='farm')
    print(sym.solve())
