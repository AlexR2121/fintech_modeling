from methods import MultiObjectiveProblem
import numpy as np


if __name__ == '__main__':
    # problem = MultiObjectiveProblem(objs_coefs=np.array([[3,4,9,9,0,2,7,3],
    #                                                      [0,3,8,2,1,1,7,8],
    #                                                      [2,6,7,6,3,8,9,1],
    #                                                      [1,6,3,6,7,6,5,7]]),
    #                                 types_of_optimization=['max']*4,
    #                                 free_coefs=np.zeros(4),
    #                                 A_uneq=np.array([[4, 9, 0, 2, 8, 8, 9, 2],
    #                                                  [5, 2, 6, 2, 4, 1, 8, 9],
    #                                                  [4, 4, 8, 5, 3, 3, 8, 5],
    #                                                  [7, 8, 6, 1, 5, 0, 3, 6],
    #                                                  [3, 7, 8, 8, 9, 8, 0, 6]]),
    #                                 b_uneq=np.array([177, 423, 335, 128, 424]),
    #                                 uneq_types=['<=']*5)
    # print(problem.solve_successive(0.1).to_string())

    # problem = MultiObjectiveProblem(objs_coefs=np.array([[ 3,  1,  3,  2,  3],
    #                                                      [10,  9, 12, 14,  9],
    #                                                      [ 7,  8,  9, 12,  6],
    #                                                      [ 7, 10, 17, 10, 12]]),
    #                                 types_of_optimization=['max', 'max', 'min', 'min'],
    #                                 free_coefs=np.zeros(4),
    #                                 A_uneq=np.array([[4, 5, 3, 2, 3],
    #                                                  [2, 4, 4, 4, 2],
    #                                                  [3, 1, 0, 1, 1],
    #                                                  [2, 3, 5, 4, 5],
    #                                                  [1, 2, 6, 3, 2],
    #                                                  [3, 4, 4, 1, 4],
    #                                                  [1, 1, 2, 2, 1],
    #                                                  [1, 0, 0, 0, 0],
    #                                                  [0, 1, 0, 0, 0],
    #                                                  [0, 0, 1, 0, 0],
    #                                                  [0, 0, 0, 1, 0],
    #                                                  [0, 0, 0, 0, 1],
    #                                                  [1, 0, 0, 0, 0],
    #                                                  [0, 1, 0, 0, 0],
    #                                                  [0, 0, 1, 0, 0],
    #                                                  [0, 0, 0, 1, 0],
    #                                                  [0, 0, 0, 0, 1]]),
    #                                 b_uneq=np.array([3000, 4500, 1500, 5000, 4000,
    #                                                  4000, 2000,  100,  100,  100,
    #                                                   100,  100,  500,  500,  500,
    #                                                   500,  500]),
    #                                 uneq_types=['<='] * 7 + ['>='] * 5 + ['<='] * 5)
    # print(problem.solve_successive(0.1).to_string())

    problem = MultiObjectiveProblem(objs_coefs=np.array([[1, 0],
                                                         [0, 1]]),
                                    types_of_optimization=['max']*2,
                                    free_coefs=np.zeros(2),
                                    A_uneq=np.array([[1, 2]]),
                                    b_uneq=np.array([2]),
                                    uneq_types=['<='])
    print(problem.solve_eps_constraints([0.4, 0.4], 0).to_string())

    # problem = MultiObjectiveProblem(objs_coefs=np.array([[  8,   7],
    #                                                      [-34, -24]]),
    #                                 types_of_optimization=['max', 'min'],
    #                                 free_coefs=np.array([0, 1248]),
    #                                 A_uneq=np.array([[4, 3],
    #                                                  [2, 1],
    #                                                  [2, 3]]),
    #                                 b_uneq=np.array([144, 64, 120]),
    #                                 uneq_types=['<=']*3)
    #
    # print(problem.solve_successive([8]).to_string())