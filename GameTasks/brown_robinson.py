import numpy as np
from game import Game
from typing import Union

class BrownRobinsonSolver:

    def __init__(self, game: Game, eps: float = 0.05):
        self.coefs = game.A

        self.prev_P: np.ndarray
        self.prev_Q: np.ndarray
        self.best_P: np.ndarray
        self.best_Q: np.ndarray
        self.alpha_k: Union[int, float]
        self.beta_k: Union[int, float]

        self.eps = eps


    def first_iteration(self):
        self.prev_P = np.zeros((1, self.coefs.shape[0]))
        self.prev_Q = np.zeros((self.coefs.shape[1],1))
        i = np.argmax(np.mean(self.coefs, axis=1))
        j = np.argmin(np.mean(self.coefs, axis=0))

        self.prev_P[0,i] = 1
        self.prev_Q[j] = 1

        self.best_P = self.prev_P
        self.best_Q = self.prev_Q

        self.alpha_k = np.min(self.coefs[i])
        self.beta_k = np.max(self.coefs[:,j])

        self.k = 1
        return self

    def iteration(self):

        i_new = np.argmax(np.dot(self.coefs, self.prev_Q))
        j_new = np.argmin(np.dot(self.prev_P, self.coefs))

        Q_k = self.k * self.prev_Q / (self.k + 1)
        Q_k[j_new] += 1 / (self.k + 1)

        P_k = self.k * self.prev_P / (self.k + 1)

        P_k[0,i_new] += 1 / (self.k + 1)

        alpha_P = np.min(np.dot(P_k, self.coefs))
        beta_Q = np.max(np.dot(self.coefs, Q_k))

        if self.alpha_k < alpha_P:
            self.alpha_k = alpha_P
            self.best_P = P_k
        if self.beta_k > beta_Q:
            self.beta_k = beta_Q
            self.best_Q = Q_k

        self.prev_P = P_k
        self.prev_Q = Q_k

        delta = self.beta_k-self.alpha_k
        if delta <= 2*self.eps:
            return True
        else:
            self.k+=1
            return False

    def solve(self):
        self.first_iteration()
        while True:
            out_cond = self.iteration()
            if out_cond:
                break

        self.I = (self.alpha_k + self.beta_k)/2
        return self.best_P, self.best_Q, self.I