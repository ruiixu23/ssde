import sympy as sp


from dynamicals.base import Dynamical


class DoubleWell(Dynamical):
    """Implementation of the Double Well dynamical.
    """
    def __init__(self):
        super(DoubleWell, self).__init__(num_x=1, num_theta=1)

    def _init_x_labels(self):
        return ['x']

    def _init_theta_labels(self):
        return ['\\theta']

    def _init_F(self):
        x,  = self.x
        theta,  = self.theta
        return sp.ImmutableMatrix([
            4 * x * (theta - x * x)
        ])
