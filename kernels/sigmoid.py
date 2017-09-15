import sympy as sp

from kernels import base


class SigmoidKernel(base.Kernel):
    def __init__(self):
        super(SigmoidKernel, self).__init__(num_phi=3)

    def _init_K(self):
        sigma, a, b = self.phi
        return sigma ** 2 * sp.asin(
            (a + b * self.x * self.y) / sp.sqrt((a + b * self.x * self.x + 1) * (a + b * self.y * self.y + 1)))
