import sympy as sp

from kernels import base


class PeriodicKernel(base.Kernel):
    def __init__(self):
        super(PeriodicKernel, self).__init__(num_phi=3)

    def _init_K(self):
        sigma, p, l = self.phi
        return sigma ** 2 * sp.exp(-1 * sp.sin(sp.pi * sp.Abs(self.x - self.y) / p) ** 2 / l ** 2)
