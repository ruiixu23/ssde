import sympy as sp

from dynamicals.base import Dynamical


class Lorenz63(Dynamical):
    """Implementation of the Lorenz 63 dynamical.
    """
    def __init__(self):
        super(Lorenz63, self).__init__(num_x=3, num_theta=3)

    def _init_x_labels(self):
        return ['x', 'y', 'z']

    def _init_theta_labels(self):
        return ['\\sigma', '\\rho', '\\beta']

    def _init_F(self):
        x, y, z = self.x
        sigma, rho, beta = self.theta
        return sp.ImmutableMatrix([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
