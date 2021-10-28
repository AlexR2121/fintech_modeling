from typing import Union, Iterable
import numpy as np
from collections import namedtuple
from Simplex.simplex import Simplex

class Game:
    def __init__(self, coefs: np.ndarray,
                 player_probs: Union[np.ndarray, Iterable, None] = None,
                 opponent_probs: Union[np.ndarray, Iterable, None] = None,
                 risk_level: Union[int, float] = 0.0):
        self.A = coefs
        self.player_probs = player_probs
        self.opponent_probs = opponent_probs
        self.risk_level = risk_level

        out_names = ['criterion_name', 'criterion_vector', 'optimal_strategy']
        self.Out = namedtuple('Out', out_names)

    def compute_baies_criterion(self):
        v = np.average(self.A, axis=1, weights=self.opponent_probs)
        return self.Out("Baies criterion", v.round(2), np.argmax(v))

    def compute_laplace_criterion(self):
        v = np.mean(self.A, axis=1)
        return self.Out('Laplace criterion', v.round(2), np.argmax(v))

    def compute_walds_criterion(self):
        v = np.min(self.A, axis=1)
        return self.Out('Walds criterion', v.round(2), np.argmax(v))

    def compute_savage_criterion(self):
        r = -self.A + np.max(self.A, axis = 0)
        v = np.max(r, axis=1)
        return self.Out('Savage criterion', v.round(2), np.argmin(v))

    def compute_hurwicz_criterion(self):
        min_a = np.min(self.A, axis=1)
        max_a = np.max(self.A, axis=1)
        v = self.risk_level*min_a + (1-self.risk_level)*max_a
        return self.Out('Hurwicz criterion', v.round(2), np.argmax(v))

    def compute_criterions(self):
        bai = self.compute_baies_criterion()
        lap = self.compute_laplace_criterion()
        wal = self.compute_walds_criterion()
        sav = self.compute_savage_criterion()
        hur = self.compute_hurwicz_criterion()
        return {'bai':bai, 'lap':lap, 'wal':wal, 'sav':sav, 'hur':hur}

    def compute_game_price(self):
        low_value = np.max(np.min(self.A, axis=1))
        high_value = np.min(np.max(self.A, axis=0))
        if np.isclose(low_value, high_value):
            self.game_price = low_value
        else:
            print("Игра не решается в чистых стратегиях")
            self.game_price = None
        return self

    def convert_to_positive_A(self):
        min_a = np.min(self.A)
        if min_a <= 0:
            self.increment = 1-min_a
        else:
            self.increment = 0
        self.A_pos = self.A + self.increment
        return self

    def solve_lpps(self):
        player_lpp = Simplex(obj_coefs=np.ones(self.A.shape[0]),
                             free_coef=0,
                             type_of_optimization='min',
                             A_uneq=self.A_pos.T,
                             b_uneq=np.ones(self.A.shape[1]),
                             uneq_types=['>=']*self.A.shape[1],
                             A_eq=None,
                             b_eq=None,
                             vars_constraints=None,
                             name='player_lpp')

        opp_lpp = Simplex(obj_coefs=np.ones(self.A.shape[1]),
                          free_coef=0,
                          type_of_optimization='max',
                          A_uneq=self.A_pos,
                          b_uneq=np.ones(self.A.shape[0]),
                          uneq_types=['<='] * self.A.shape[0],
                          A_eq=None,
                          b_eq=None,
                          vars_constraints=None,
                          name='opp_lpp')

        player_lpp_dict = player_lpp.solve()
        opp_lpp_dict = opp_lpp.solve()

        u = np.array([player_lpp_dict[f'x{i}'] for i in range(self.A.shape[0])])
        v = np.array([opp_lpp_dict[f'x{j}'] for j in range(self.A.shape[1])])

        if np.isclose(player_lpp_dict['Q_opt'],opp_lpp_dict['Q_opt']):
            I = 1/player_lpp_dict['Q_opt']
        else:
            print('Цены не совпадают')
            return
        return u*I, v*I, I-self.increment

    def solve_game(self):
        self.compute_game_price()
        if self.game_price is None:
            self.convert_to_positive_A()
            p, q, game_price = self.solve_lpps()
        else:
            p, q, game_price = None, None, self.game_price
        self.player_probs = p
        self.opponent_probs = q
        return p, q, game_price


    def print_criterions(self):
        crit_dict = self.compute_criterions()
        for criteria in crit_dict.values():
            print("=" * 31)
            print(criteria[0])
            print("="*31)
            for i, v in enumerate(criteria[1]):
                print(f'{i}: {v}')
            print(f'Номер предлагаемой стратегии: {criteria[2]}')
            print("=" * 31)
            print('\n')
