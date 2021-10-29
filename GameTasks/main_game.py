from game import Game
from brown_robinson import BrownRobinsonSolver
import numpy as np
from typing import Union
from time import time

def get_random_game(lower: Union[int, float] = 1,
                    upper: Union[int, float] = 10,
                    m: int = 3, n: int = 3,
                    random_state: Union[int, None] = None) -> Game:

    if random_state is not None:
        np.random.seed(random_state)

    rand_coefs = np.random.randint(low=lower, high=upper,
                                   size=(m, n))
    rand_game = Game(coefs = rand_coefs)
    return rand_game

if __name__ == '__main__':

    #Пример вычисления критериев для игры

    # ex_game = Game(coefs = np.array([[33, 10, 20, 26.5],
    #                                  [50, 67, 11.5, 25],
    #                                  [23.5, 35, 40, 58.5]]),
    #                opponent_probs = [0.3, 0.2, 0.4, 0.1],
    #                risk_level = 0.5)
    #
    # ex_game.print_criterions()

    # Примеры решения игр в смешанных стратегиях

    game_coefs = np.array([[3, -4, 2],
                           [1, -3, -7],
                           [-2, 4, 7]])

    # game_coefs = np.array([[6, -2],
    #                        [3, 5]])

    # game_coefs = np.array([[2, 5, 8],
    #                        [7, 4, 3]])

    # game_coefs = np.array([[2, 3, 2, 4],
    #                        [3, 2, 4, 1],
    #                        [4, 1, 3, 1]])
    #
    game_to_solve = Game(coefs = game_coefs)
    solution = game_to_solve.solve_game()
    print(f'p*: {solution[0]}')
    print(f'q*: {solution[1]}')
    print(f'I: {solution[2]}')

    # Пример решения игры методом Брауна-Робинсон
    # br_game = BrownRobinsonSolver(game=game_to_solve, eps=1e-4)
    # print(br_game.solve())

    # Сравнение скорости работы двух методов
    # random_game = get_random_game(-10, 10, 100, 100, 21)
    # br_game = BrownRobinsonSolver(game=random_game, eps=3e-2)
    #
    # start_lpp = time()
    # solution = random_game.solve_game()
    # stop_lpp = time()
    #
    # print('LPP elapsed {} s'.format(round(stop_lpp - start_lpp, 2))) #0.18 s
    #
    # start_br = time()
    # solution_br = br_game.solve()
    # stop_br = time()
    #
    # print('Brown-Robinson elapsed {} s'.format(round(stop_br - start_br, 2))) #42.25 s