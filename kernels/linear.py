from kernels import base


class LinearKernel(base.Kernel):
    def __init__(self):
        super(LinearKernel, self).__init__(num_phi=3)

    def _init_K(self):
        sigma_b, sigma_v, c = self.phi
        return sigma_b ** 2 + sigma_v ** 2 * (self.x - c) * (self.y - c)
