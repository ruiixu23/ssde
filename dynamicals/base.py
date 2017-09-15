import abc

import numpy as np

import sympy as sp


from scipy.integrate import ode

from sdeint import itoSRI2


class Dynamical(abc.ABC):
    x_label = 'x_'
    dummy_x = sp.Symbol(x_label, real=True)

    theta_label = 'theta_'
    dummy_theta = sp.Symbol(theta_label, real=True)

    def __init__(self, num_x, num_theta):
        assert num_x > 0, 'The system must have at least one state.'

        self.num_x = num_x
        self.x = list(sp.symbols('{}[(:{})]'.format(self.x_label, self.num_x), real=True))
        self.x_labels = self._init_x_labels()

        self.num_theta = num_theta
        self.theta = list(sp.symbols('{}[(:{})]'.format(self.theta_label, self.num_theta), real=True))
        self.theta_labels = self._init_theta_labels()

        # Refer to the respective init functions for comments
        self.F = self._init_F()
        self.F_funcs = self._init_F_funcs()

        self.x_F_relation = self._init_x_F_relation()
        self.x_dF = self._init_x_dF()
        self.x_dF_funcs = self._init_x_dF_funcs()
        self.x_dF_relation = self._init_x_dF_relation()
        self.x_ddF = self._init_x_ddF()
        self.x_ddF_funcs = self._init_x_ddF_funcs()
        self.x_ddF_relation = self._init_x_ddF_relation()

        self.theta_F_relation = self._init_theta_F_relation()
        self.theta_dF = self._init_theta_dF()
        self.theta_dF_funcs = self._init_theta_dF_funcs()
        self.theta_dF_relation = self._init_theta_dF_relation()
        self.theta_ddF = self._init_theta_ddF()
        self.theta_ddF_funcs = self._init_theta_ddF_funcs()
        self.theta_ddF_relation = self._init_theta_ddF_relation()

        # Write the ODEs as a linear combination of theta
        self.theta_B, self.theta_b = sp.linear_eq_to_matrix(self.F.expand(), self.theta)
        self.theta_b *= -1
        self.theta_B_funcs = self._init_theta_B_funcs()
        self.theta_b_funcs = self._init_theta_b_funcs()

    @abc.abstractmethod
    def _init_x_labels(self):
        pass

    @abc.abstractmethod
    def _init_theta_labels(self):
        pass

    @abc.abstractmethod
    def _init_F(self):
        pass

    def _init_F_funcs(self):
        return [
            sp.lambdify((self.dummy_x, self.dummy_theta), self.F[i, 0], dummify=False, modules='numpy')
            for i in range(self.num_x)
        ]

    def _init_x_F_relation(self):
        """Initialize x_F_relation s.t. x_F_relation[i] contains the indices of the drift functions that
        have dependency on x[i]
        """
        x_indices = dict(zip(self.x, range(self.num_x)))
        x_F_relation = [list() for _ in range(self.num_x)]
        for i in range(self.num_x):
            for symbol in self.F[i, 0].free_symbols:
                if symbol in x_indices:
                    x_F_relation[x_indices[symbol]].append(i)
        return x_F_relation

    def _init_x_dF(self):
        """Initialize x_dF s.t. x_dF[i, j] = dF[j, 0]/dx[i]
        """
        x_dF = sp.SparseMatrix(self.num_x, self.num_x, 0)
        for i in range(self.num_x):
            for j in self.x_F_relation[i]:
                x_dF[i, j] = sp.diff(self.F[j, 0], self.x[i])
        return x_dF.as_immutable()

    def _init_x_dF_funcs(self):
        return [
            [
                sp.lambdify((self.dummy_x, self.dummy_theta), self.x_dF[i, j], dummify=False, modules='numpy')
                if not isinstance(self.x_dF[i, j], sp.numbers.Zero)
                else None
                for j in range(self.num_x)
            ]
            for i in range(self.num_x)
        ]

    def _init_x_dF_relation(self):
        """Initialize x_dF_relation s.t. x_dF_relation[i] contains the indices of the drift functions whose
        derivative w.r.t x[i] is not zero, that is, dF[j, 0]/dx[i] = x_dF[i, j] != 0
        """
        return [
            [
                j
                for j in range(self.num_x)
                if not isinstance(self.x_dF[i, j], sp.numbers.Zero)
            ]
            for i in range(self.num_x)
        ]

    def _init_x_ddF(self):
        """Initialize x_ddF s.t. x_ddF[i, j] = d^2F[j, 0]/dx[i]dx[i]
        """
        x_ddF = sp.SparseMatrix(self.num_x, self.num_x, 0)
        for i in range(self.num_x):
            for j in self.x_dF_relation[i]:
                x_ddF[i, j] = sp.diff(self.x_dF[i, j], self.x[i])
        return x_ddF.as_immutable()

    def _init_x_ddF_funcs(self):
        return [
            [
                sp.lambdify((self.dummy_x, self.dummy_theta), self.x_ddF[i, j], dummify=False, modules='numpy')
                if not isinstance(self.x_ddF[i, j], sp.numbers.Zero)
                else None
                for j in range(self.num_x)
            ]
            for i in range(self.num_x)
        ]

    def _init_x_ddF_relation(self):
        """Initialize x_ddF_realtion s.t. x_ddF_relation[i] contains the indices of the drift functions whose
        second derivative w.r.t x[i] is not zero, that is, d^2F[j]/dx[i]dx[i] = x_ddF[i, j] != 0
        """
        return [
            [
                j
                for j in range(self.num_x)
                if not isinstance(self.x_ddF[i, j], sp.numbers.Zero)
            ]
            for i in range(self.num_x)
        ]

    def _init_theta_F_relation(self):
        theta_indices = dict(zip(self.theta, range(self.num_theta)))
        F_theta_relation = [list() for _ in range(self.num_theta)]
        for i in range(self.num_x):
            for symbol in self.F[i, 0].free_symbols:
                if symbol in theta_indices:
                    F_theta_relation[theta_indices[symbol]].append(i)
        return F_theta_relation

    def _init_theta_dF(self):
        theta_dF = sp.SparseMatrix(self.num_theta, self.num_x, 0)
        for m in range(self.num_theta):
            for i in self.theta_F_relation[m]:
                theta_dF[m, i] = sp.diff(self.F[i, 0], self.theta[m])
        return theta_dF.as_immutable()

    def _init_theta_dF_funcs(self):
        return [
            [
                sp.lambdify((self.dummy_x, self.dummy_theta), self.theta_dF[m, i], dummify=False, modules='numpy')
                if not isinstance(self.theta_dF[m, i], sp.numbers.Zero)
                else None
                for i in range(self.num_x)
            ]
            for m in range(self.num_theta)
        ]

    def _init_theta_dF_relation(self):
        return [
            [
                i
                for i in range(self.num_x)
                if not isinstance(self.theta_dF[m, i], sp.numbers.Zero)
            ]
            for m in range(self.num_theta)
        ]

    def _init_theta_ddF(self):
        theta_ddF = [sp.SparseMatrix(self.num_theta, self.num_theta, 0) for _ in range(self.num_x)]
        for i in range(self.num_x):
            for m in range(self.num_theta):
                for n in range(self.num_theta):
                    if n <= m:
                        theta_ddF[i][m, n] = sp.diff(self.F[i, 0], self.theta[m], self.theta[n])
                    else:
                        break
        return theta_ddF

    def _init_theta_ddF_funcs(self):
        return [
            np.array([
                [
                    sp.lambdify((self.dummy_x, self.dummy_theta), self.theta_ddF[i][m, n], dummify=False,
                                modules='numpy')
                    if not isinstance(self.theta_ddF[i][m, n], sp.numbers.Zero)
                    else None
                    for n in range(self.num_theta)
                ]
                for m in range(self.num_theta)
            ])
            for i in range(self.num_x)
        ]

    def _init_theta_ddF_relation(self):
        return [
            [
                (m, n)
                for m in range(self.num_theta)
                for n in range(self.num_theta)
                if not isinstance(self.theta_ddF[i][m, n], sp.numbers.Zero)
            ]
            for i in range(self.num_x)
        ]

    def _init_theta_B_funcs(self):
        return [
            [
                sp.lambdify(self.dummy_x, self.theta_B[i, n], dummify=False, modules='numpy')
                for n in range(self.num_theta)
            ]
            for i in range(self.num_x)
        ]

    def _init_theta_b_funcs(self):
        return [
            sp.lambdify(self.dummy_x, self.theta_b[i, 0], dummify=False, modules='numpy')
            for i in range(self.num_x)
        ]

    def generate_sample_path(self, theta, rho_2, X_0, spl_tps):
        assert theta.size == self.num_theta, 'Wrong number of drift parameters.'
        if rho_2 is not None:
            assert rho_2.size == self.num_x, 'Wrong number of diffusion variance.'
        assert X_0.size == self.num_x, 'Wrong number of initial states.'

        if rho_2 is not None:
            # Solve the SDEs using the 1.0 order solver
            F_func = sp.lambdify((self.dummy_x, self.dummy_theta), self.F, dummify=False, modules='numpy')
            G_matrix = np.diag(np.sqrt(rho_2))
            X_spl = itoSRI2(lambda x, t: F_func(x, theta).ravel(), lambda x, t: G_matrix, X_0, spl_tps).T
            return X_spl
        else:
            # Solve the ODEs using explicit Runge-Kutta method of order (4)5
            X_spl = np.empty((self.num_x, spl_tps.size))
            F_func = sp.lambdify((self.dummy_x, self.dummy_theta), self.F, dummify=False, modules='numpy')
            integrator = ode(lambda t, x: F_func(x, theta).ravel())
            integrator.set_integrator('dopri5', rtol=1e-12, nsteps=250, first_step=1e-5, max_step=1e-3)
            integrator.set_initial_value(X_0, spl_tps[0])
            for i in range(spl_tps.size):
                X_spl[:, i] = integrator.integrate(spl_tps[i])
                if not integrator.successful():
                    raise RuntimeError('Failed to numerically integrate the dynamical system.')
            return X_spl


