import sympy as sp

from dynamicals.base import Dynamical


class OrnsteinUhlenbeck(Dynamical):
    """Implementation of the Ornstein Uhlenbeck dynamical.
    """
    def __init__(self):
        super(OrnsteinUhlenbeck, self).__init__(num_x=1, num_theta=2)

    def _init_x_labels(self):
        return ['x']

    def _init_theta_labels(self):
        return ['\\theta', '\\mu']

    def _init_F(self):
        x, = self.x
        theta, mu = self.theta
        return sp.ImmutableMatrix([
            theta * (mu - x)
        ])
