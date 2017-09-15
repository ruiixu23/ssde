import sympy as sp

from kernels import base


class LocallyPeriodicKernel(base.Kernel):
    def __init__(self):
        super(LocallyPeriodicKernel, self).__init__(num_phi=3)

    def _init_K(self):
        sigma, p, l = self.phi
        return (
            sigma ** 2 *
            sp.exp(-2 * sp.sin(sp.pi * sp.Abs(self.x - self.y) / p) ** 2 / l ** 2) *
            sp.exp(-1 * (self.x - self.y) ** 2 / (2 * l ** 2))
        )
