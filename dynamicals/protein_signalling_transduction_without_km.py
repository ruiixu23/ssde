import sympy as sp


from dynamicals.base import Dynamical


class ProteinSignallingTransductionWithoutKm(Dynamical):
    """Implementation of the Protein Signalling Transduction (Model 1)
    Detailed description of the system can be found in the paper:
        Vladislav Vyshemirsky and Mark A Girolami. Bayesian ranking of bio- chemical system models.
        Bioinformatics, 24(6):833-839, 2007.
    """
    def __init__(self):
        super(ProteinSignallingTransductionWithoutKm, self).__init__(num_x=5, num_theta=5)

    def _init_x_labels(self):
        return ['S', 'dS', 'R', 'RS', 'Rpp']

    def _init_theta_labels(self):
        return ['k_{1}', 'k_{2}', 'k_{3}', 'k_{4}', 'V']

    def _init_F(self):
        S, dS, R, RS, Rpp = self.x
        k1, k2, k3, k4, V = self.theta
        return sp.ImmutableMatrix([
            -k1 * S - k2 * S * R + k3 * RS,  # S'
            k1 * S,  # dS'
            -k2 * S * R + k3 * RS + V * Rpp / (3. + Rpp),  # R'
            k2 * S * R - k3 * RS - k4 * RS,  # RS'
            k4 * RS - V * Rpp / (3. + Rpp)  # Rpp'
        ])
