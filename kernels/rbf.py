import sympy as sp

from kernels import base


class RBFKernel(base.Kernel):
    def __init__(self):
        super(RBFKernel, self).__init__(num_phi=2)

    def _init_K(self):
        sigma, l = self.phi
        return sigma ** 2 * sp.exp(-1 * (self.x - self.y) ** 2 / (2 * l ** 2))
