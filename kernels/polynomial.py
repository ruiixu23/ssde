from kernels import base


class Polynomial(base.Kernel):
    def __init__(self):
        super(Polynomial, self).__init__(num_phi=2)

    def _init_K(self):
        c, d = self.phi
        return (c + self.x * self.y) ** d
