import sympy as sp


from dynamicals.base import Dynamical


class FitzHughNagumo(Dynamical):
    """Implementation of the Double Well dynamical.
    """
    def __init__(self):
        super(FitzHughNagumo, self).__init__(num_x=2, num_theta=3)

    def _init_x_labels(self):
        return ['V', 'R']

    def _init_theta_labels(self):
        return ['a', 'b', 'c']

    def _init_F(self):
        V, R = self.x
        a, b, c = self.theta
        return sp.ImmutableMatrix([
            c * (V - V ** 3 / 3 + R),
            - (V - a + b * R) / c
        ])
