from piyavsky import Piyavsky
import numpy as np

if __name__ == '__main__':
    """# Найдем минимум функции Сиблинского-Танга на промежутке [-5; 5]."""
    def styblinski_tang(x):
        return 0.5 * x ** 4 - 8 * x ** 2 + 2.5 * x


    optimizer = Piyavsky(styblinski_tang, -5, 5, 1000000, L_mult=2.2)

    # Commented out IPython magic to ensure Python compatibility.
    # %%time
    x_m, y_m = optimizer.fit(5e-4, 50000)

    print('Истинная точка минимума: -2.903504')
    print(f'Точка минимума, найденная методом Пиявского: {np.round(x_m, 5)}')
    print(f'Минимум, найденный методом Пиявского: {np.round(y_m, 5)}')