from game import Game
import numpy as np

if __name__ == '__main__':
    ex_game = Game(coefs = np.array([[33, 10, 20, 26.5],
                                     [50, 67, 11.5, 25],
                                     [23.5, 35, 40, 58.5]]),
                   opponent_probs = [0.3, 0.2, 0.4, 0.1],
                   risk_level = 0.5)

    ex_game.print_criterions()