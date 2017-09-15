import abc

import sympy as sp

import numericals


class Kernel(abc.ABC):
    x_label = 'x_'
    x = sp.Symbol(x_label, real=True, negative=False)
    y_label = 'y_'
    y = sp.Symbol(y_label, real=True, negative=False)
    phi_label = 'phi_'
    dummy_phi = sp.Symbol(phi_label, real=True)

    def __init__(self, num_phi):
        self.num_phi = num_phi
        self.phi = list(sp.symbols('{}[(:{})]'.format(self.phi_label, self.num_phi)))
        self.K = self._init_K()

        modules = [{'DiracDelta': numericals.direc_delta}, 'numpy']

        self.C = self.K
        self.C_func = sp.lambdify((self.x, self.y, self.dummy_phi), self.C, dummify=False, modules=modules)

        self.dC = sp.diff(self.K, self.x)
        self.dC_func = sp.lambdify((self.x, self.y, self.dummy_phi), self.dC, dummify=False, modules=modules)

        self.Cd = sp.diff(self.K, self.y)
        self.Cd_func = sp.lambdify((self.x, self.y, self.dummy_phi), self.Cd, dummify=False, modules=modules)

        self.dCd = sp.diff(self.K, self.x, self.y)
        self.dCd_func = sp.lambdify((self.x, self.y, self.dummy_phi), self.dCd, dummify=False, modules=modules)

    @abc.abstractmethod
    def _init_K(self):
        pass
