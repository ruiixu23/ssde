import sympy as sp

from dynamicals.base import Dynamical


class LotkaVolterra(Dynamical):
    """Implementation of the Lotka Volterra dynamical.
    """
    def __init__(self):
        super(LotkaVolterra, self).__init__(num_x=2, num_theta=4)

    def _init_x_labels(self):
        return ['Sheep', 'Wolves']

    def _init_theta_labels(self):
        return ['\\alpha', '\\beta', '\\delta', '\\gamma']

    def _init_F(self):
        x, y = self.x
        alpha, beta, delta, gamma = self.theta
        return sp.ImmutableMatrix([
            alpha * x - beta * x * y,
            delta * x * y - gamma * y
        ])
