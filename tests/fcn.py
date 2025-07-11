import numpy as np


def f(x):
    return x**2


def get_data(num_points=10, noise=False, noise_std=2.0, x_range=(0, 10)):
    x = np.random.uniform(x_range[0], x_range[1], num_points)
    y = f(x)

    if noise:
        y += np.random.normal(0, noise_std, num_points)

    return x, y
