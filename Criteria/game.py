from typing import Union, Iterable
import numpy as np
from collections import namedtuple

class Game:
    def __init__(self, coefs: np.ndarray,
                 opponent_probs: Union[np.ndarray, Iterable],
                 risk_level: Union[int, float] = 0.0):
        self.A = coefs
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
