import sympy as sp

from dynamicals.base import Dynamical


class Lorenz96(Dynamical):
    """Implementation of the Lorenz 96 dynamical.
    """
    def __init__(self, num_x=4):
        super(Lorenz96, self).__init__(num_x=num_x, num_theta=1)

    def _init_x_labels(self):
        return ['x_{' + str(k + 1) + '}' for k in range(self.num_x)]

    def _init_theta_labels(self):
        return ['F']

    def _init_F(self):
        return sp.ImmutableMatrix([
            (
                (self.x[(k + 1) % self.num_x] - self.x[k - 2]) * self.x[k - 1]
                - self.x[k]
                + self.theta[0]
            )
            for k in range(self.num_x)
        ])
