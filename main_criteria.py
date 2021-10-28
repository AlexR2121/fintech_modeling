from game import Game
import numpy as np

if __name__ == '__main__':
    # ex_game = Game(coefs = np.array([[33, 10, 20, 26.5],
    #                                  [50, 67, 11.5, 25],
    #                                  [23.5, 35, 40, 58.5]]),
    #                opponent_probs = [0.3, 0.2, 0.4, 0.1],
    #                risk_level = 0.5)
    #
    # ex_game.print_criterions()

    # game_coefs = np.array([[3, -4, 2],
    #                        [1, -3, -7],
    #                        [-2, 4, 7]])
    # game_coefs = np.array([[6, -2],
    #                        [3, 5]])
    game_coefs = np.array([[2, 5, 8],
                           [7, 4, 3]])
    game_to_solve = Game(coefs = game_coefs)
    solution = game_to_solve.solve_game()
    print(f'p*: {solution[0]}')
    print(f'q*: {solution[1]}')
    print(f'I: {solution[2]}')