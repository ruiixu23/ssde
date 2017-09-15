import sympy as sp


from dynamicals.base import Dynamical


class GlucoseUptakeYeast(Dynamical):
    """Implementation of the Glucose Uptake Yeast model
    """
    def __init__(self):
        super(GlucoseUptakeYeast, self).__init__(num_x=9, num_theta=10)

    def _init_x_labels(self):
        return ['x_{Glc}^e', 'x_{Glc}^i', 'x_{E-G6P}^i',
                'x_{E-Glc-G6P}^i', 'x_{G6P}^i', 'x_{E-Glc}^e',
                'x_{E-Glc}^i', 'x_E^e', 'x_E^i']

    def _init_theta_labels(self):
        return ['k_1', 'k_{-1}', 'k_2', 'k_{-2}', 'k_3', 'k_{-3}', 'k_4', 'k_{-4}', 'alpha', 'beta']

    def _init_F(self):
        x1, x2, x3, x4, x5, x6, x7, x8, x9 = self.x
        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = self.theta
        return sp.ImmutableMatrix([
            -k1 * x8 * x1 + k2 * x6,
            -k3 * x9 * x2 + k4 * k7,
            k7 * k9 * x5 - k8 * x7,
            k5 * x7 * x5 - k6 * x4,
            -k5 * x7 * x5 + k6 * x4 - k7 * x9 * x5 + k8 * x3,
            k9 * (x7 - x6) + k1 * x8 * x1 - k2 * x6,
            k9 * (x6 - x7) - k5 * x7 * x5 + k6 * x4 + k3 * x9 * x2 - k4 * x7,
            k10 * (x9 - x8) - k1 * x8 * x1 + k2 * x6,
            k10 * x8 * x9 - k7 * x9 * x5 + k8 * x3 - k3 * x9 * x2 + k4 * x7
        ])
