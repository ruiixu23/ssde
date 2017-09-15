import numpy as np
import sympy as sp

from scipy import optimize

from core.laplace_mean_field import LaplaceMeanField


class LaplaceMeanFieldSymbolic(LaplaceMeanField):
    def __init__(self, dynamical, config, gp):
        super(LaplaceMeanFieldSymbolic, self).__init__(dynamical, config, gp)

        self.use_theta_X_tilde = None
        self.use_theta_theta_tilde = None

        self.eta_theta_tilde = None
        self.eta_theta_args = None

        self.theta_objective_func = None
        self.theta_gradient_func = None
        self.theta_Hessian_func = None

        self.use_X_X_tilde = None
        self.use_X_theta_tilde = None

        self.eta_X_tilde = None
        self.eta_X_args = None

        self.X_common_objective_func = None
        self.X_objectives_func = None
        self.X_gradients_func = None
        self.X_Hessians_func = None

    def _construct_X_sym(self, use_X_tilde):
        dynamical = self.dynamical
        num_x = dynamical.num_x
        num_est_t = self.config.est_tps.size

        X_tilde_sym = sp.Matrix(
            sp.symbols('{}[(:{})][(:{})]'.format(dynamical.x_label, num_x, num_est_t), real=True)
        ).reshape(num_x, num_est_t)

        if use_X_tilde:
            X_sym = sp.zeros(num_x, num_est_t)
            for i in range(num_x):
                for t in range(num_est_t):
                    X_sym[i, t] = sp.exp(X_tilde_sym[i, t])
        else:
            X_sym = X_tilde_sym

        return X_sym, X_tilde_sym

    def _construct_theta_sym(self, use_theta_tilde):
        dynamical = self.dynamical
        num_theta = dynamical.num_theta

        theta_tilde_sym = sp.Matrix(
            sp.symbols('{}[(:{})]'.format(dynamical.theta_label, num_theta), real=True)
        )

        if use_theta_tilde:
            theta_sym = sp.zeros(num_theta, 1)
            for m in range(num_theta):
                theta_sym[m, 0] = sp.exp(theta_tilde_sym[m, 0])
        else:
            theta_sym = theta_tilde_sym

        return theta_sym, theta_tilde_sym

    def _construct_F_sym(self, X_sym, theta_sym, use_X_tilde):
        dynamical = self.dynamical
        num_x = dynamical.num_x
        num_est_t = self.config.est_tps.size

        theta_subs = dict(zip(dynamical.theta, theta_sym))
        F_sym = sp.Matrix.hstack(*[
            dynamical.F.xreplace(theta_subs).xreplace(dict(zip(dynamical.x, X_sym.col(t))))
            for t in range(num_est_t)
        ]).as_mutable()

        if use_X_tilde:
            for i in range(num_x):
                for t in range(num_est_t):
                    F_sym[i, t] = F_sym[i, t] / X_sym[i, t]

        return F_sym

    def _construct_common_objectives(self, X_sym, F_sym):
        dynamical = self.dynamical
        num_x = dynamical.num_x
        m = self.gp.m
        Lambda = self.gp.Lambda
        debug = self.config.debug

        if debug:
            print('Constructing common objectives:', end=' ')

        common_objectives_sym = [None] * num_x
        for i in range(num_x):
            if debug:
                print(i + 1, end=' ')
            F_mx_i_sym = F_sym.row(i).T - sp.Matrix(m[i]) * X_sym.row(i).T
            common_objectives_sym[i] = sp.expand(0.5 * F_mx_i_sym.T * sp.Matrix(Lambda[i]) * F_mx_i_sym)

        if debug:
            print()
            print('Lambdifying common objectives:', end=' ')

        common_objective_sym = 0.
        for i in range(num_x):
            if debug:
                print(i + 1, end=' ')
            common_objective_sym += common_objectives_sym[i][0, 0]
        common_objective_func = sp.lambdify((dynamical.dummy_x, dynamical.dummy_theta), common_objective_sym,
                                            dummify=False, modules='numpy')

        if debug:
            print()

        return common_objectives_sym, common_objective_func

    def _construct_theta_gradient(self, theta_tilde_sym, common_objectives_sym):
        dynamical = self.dynamical
        num_x = dynamical.num_x
        num_theta = dynamical.num_theta
        debug = self.config.debug

        if debug:
            print('Constructing theta gradient:', end=' ')

        theta_gradient_sym = sp.zeros(1, num_theta)
        for i in range(num_x):
            if debug:
                print(i + 1, end=' ')
            theta_gradient_sym += common_objectives_sym[i].jacobian(theta_tilde_sym)

        if debug:
            print()
            print('Lambdifying theta gradient.')

        theta_gradient_func = sp.lambdify((dynamical.dummy_x, dynamical.dummy_theta), theta_gradient_sym,
                                          dummify=False, modules='numpy')

        return theta_gradient_sym, theta_gradient_func

    def _construct_theta_Hessian(self, theta_tilde_sym, theta_gradient_sym):
        dynamical = self.dynamical
        debug = self.config.debug

        if debug:
            print('Constructing theta Hessian.')

        theta_Hessian_sym = theta_gradient_sym.jacobian(theta_tilde_sym)

        if debug:
            print('Lambdifying theta Hessian.')

        theta_Hessian_func = sp.lambdify((dynamical.dummy_x, dynamical.dummy_theta), theta_Hessian_sym,
                                         dummify=False, modules='numpy')

        return theta_Hessian_sym, theta_Hessian_func

    def _construct_X_objectives(self, X_sym):
        dynamical = self.dynamical
        num_x = dynamical.num_x
        mu = self.gp.mu
        inv_Sigma = self.gp.inv_Sigma
        debug = self.config.debug

        if debug:
            print('Constructing X objectives:', end=' ')

        X_objectives_sym = [None] * num_x
        for i in range(num_x):
            if debug:
                print(i + 1, end=' ')
            X_mu_i_sym = X_sym.row(i).T - sp.Matrix(mu[i])
            X_objectives_sym[i] = sp.expand(0.5 * X_mu_i_sym.T * sp.Matrix(inv_Sigma[i]) * X_mu_i_sym)

        if debug:
            print()
            print('Lambdifying X objectives:', end=' ')

        X_objectives_func = [None] * num_x
        for i in range(num_x):
            if debug:
                print(i + 1, end=' ')
            X_objectives_func[i] = sp.lambdify((dynamical.dummy_x, dynamical.dummy_theta), X_objectives_sym[i][0, 0],
                                               dummify=False, modules='numpy')
        if debug:
            print()

        return X_objectives_sym, X_objectives_func

    def _construct_X_gradients(self, X_tilde_sym, common_objectives_sym, X_objectives_sym):
        dynamical = self.dynamical
        num_x = dynamical.num_x
        debug = self.config.debug

        if debug:
            print('Constructing X gradients:', end=' ')

        X_gradients_sym = [None] * num_x
        for i in range(num_x):
            if debug:
                print(i + 1, end=' ')
            X_gradient_sym = X_objectives_sym[i].jacobian(X_tilde_sym.row(i))
            for j in dynamical.x_F_relation[i]:
                X_gradient_sym += common_objectives_sym[j].jacobian(X_tilde_sym.row(i))
            X_gradients_sym[i] = X_gradient_sym

        if debug:
            print()
            print('Lambdifying X gradients:', end=' ')

        X_gradients_func = [None] * num_x
        for i in range(num_x):
            if debug:
                print(i + 1, end=' ')
            X_gradients_func[i] = sp.lambdify((dynamical.dummy_x, dynamical.dummy_theta), X_gradients_sym[i],
                                              dummify=False, modules='numpy')

        if debug:
            print()

        return X_gradients_sym, X_gradients_func

    def _construct_X_Hessians(self, X_tilde_sym, X_gradients_sym):
        dynamical = self.dynamical
        num_x = dynamical.num_x
        debug = self.config.debug

        if debug:
            print('Constructing X hessians:', end=' ')

        X_Hessians_sym = [None] * num_x
        for i in range(num_x):
            if debug:
                print(i + 1, end=' ')
            X_Hessians_sym[i] = X_gradients_sym[i].jacobian(X_tilde_sym.row(i))

        if debug:
            print()
            print('Lambdifying X hessians:', end=' ')

        X_Hessians_func = [None] * num_x
        for i in range(num_x):
            if debug:
                print(i + 1, end=' ')
            X_Hessians_func[i] = sp.lambdify((dynamical.dummy_x, dynamical.dummy_theta), X_Hessians_sym[i],
                                             dummify=False, modules='numpy')

        if debug:
            print()

        return X_Hessians_sym, X_Hessians_func

    def _construct_objective_functions(self, group, use_X_tilde, use_theta_tilde):
        if group == 'X':
            self.use_X_X_tilde = use_X_tilde
            self.use_X_theta_tilde = use_theta_tilde
        elif group == 'theta':
            self.use_theta_X_tilde = use_X_tilde
            self.use_theta_theta_tilde = use_theta_tilde
        else:
            raise RuntimeError('Group must be either X or theta.')

        debug = self.config.debug

        if debug:
            print('Constructing {} objectives...'.format(group))
            if use_X_tilde:
                print('Using X tilde.')
            else:
                print('Using X.')

            if use_theta_tilde:
                print('Using theta tilde.')
            else:
                print('Using theta.')

        X_sym, X_tilde_sym = self._construct_X_sym(use_X_tilde)
        theta_sym, theta_tilde_sym = self._construct_theta_sym(use_theta_tilde)
        F_sym = self._construct_F_sym(X_sym, theta_sym, use_X_tilde)

        if debug:
            print('  X_sym[0, 0] = ', X_sym[0, 0])
            print('  theta_sym[0, 0] = ', theta_sym[0, 0])
            print('  F_sym[0, 0] = ', F_sym[0, 0])

        common_objectives_sym, common_objective_func = self._construct_common_objectives(X_sym, F_sym)
        if group == 'X':
            X_objectives_sym, X_objectives_func = self._construct_X_objectives(X_sym)

            # The gradients and Hessians are always taken w.r.t. X_tilde_sym since X_sym == X_tilde_sym
            # when no reparameterization is used.
            X_gradients_sym, X_gradients_func = self._construct_X_gradients(X_tilde_sym,
                                                                            common_objectives_sym,
                                                                            X_objectives_sym)

            X_Hessians_sym, X_Hessians_func = self._construct_X_Hessians(X_tilde_sym, X_gradients_sym)

            self.X_common_objective_func = common_objective_func
            self.X_objectives_func = X_objectives_func
            self.X_gradients_func = X_gradients_func
            self.X_Hessians_func = X_Hessians_func
        else:
            # The gradient and Hessian are always taken w.r.t. theta_tilde_sym since theta_sym == theta_tilde_sym
            # when no reparameterization is used.
            theta_gradient_sym, theta_gradient_func = self._construct_theta_gradient(theta_tilde_sym,
                                                                                     common_objectives_sym)
            theta_Hessian_sym, theta_Hessian_func = self._construct_theta_Hessian(theta_tilde_sym,
                                                                                  theta_gradient_sym)
            self.theta_objective_func = common_objective_func
            self.theta_gradient_func = theta_gradient_func
            self.theta_Hessian_func = theta_Hessian_func

        if debug:
            print('Done.')

    def construct_X_objective_functions(self, use_X_X_tilde, use_X_theta_tilde):
        self._construct_objective_functions('X', use_X_X_tilde, use_X_theta_tilde)

    def construct_theta_objective_functions(self, use_theta_X_tilde, use_theta_theta_tilde):
        self._construct_objective_functions('theta', use_theta_X_tilde, use_theta_theta_tilde)

    def init_optimization(self):
        self.eta_theta = np.zeros(self.dynamical.num_theta)
        self.eta_theta_tilde = np.zeros(self.dynamical.num_theta)

        self.eta_X = self.gp.mu.copy()
        self.eta_X_tilde = np.log(self.eta_X)

        if self.use_theta_X_tilde:
            arg_1 = self.eta_X_tilde
        else:
            arg_1 = self.eta_X
        if self.use_theta_theta_tilde:
            arg_2 = self.eta_theta_tilde
        else:
            arg_2 = self.eta_theta
        self.eta_theta_args = (arg_1, arg_2)

        if self.use_X_X_tilde:
            arg_1 = self.eta_X_tilde
        else:
            arg_1 = self.eta_X
        if self.use_X_theta_tilde:
            arg_2 = self.eta_theta_tilde
        else:
            arg_2 = self.eta_theta
        self.eta_X_args = (arg_1, arg_2)
        self.i = 0

    def optimize_theta(self):
        opt_method = self.config.opt_method
        if opt_method == 'BFGS':
            if self.use_theta_theta_tilde:
                result = optimize.minimize(fun=self.theta_objective_and_gradient, x0=self.eta_theta_tilde,
                                           method=opt_method, jac=True)
                self.eta_theta[:] = np.exp(self.eta_theta_tilde)
            else:
                result = optimize.minimize(fun=self.theta_objective_and_gradient, x0=self.eta_theta,
                                           method='Newton-CG', jac=True)
                self.eta_theta_tilde[:] = np.log(self.eta_theta)
        else:
            if self.use_theta_theta_tilde:
                result = optimize.minimize(fun=self.theta_objective_and_gradient, x0=self.eta_theta_tilde,
                                           method=opt_method, jac=True, hess=self.theta_Hessian)
                self.eta_theta[:] = np.exp(self.eta_theta_tilde)
            else:
                result = optimize.minimize(fun=self.theta_objective_and_gradient, x0=self.eta_theta,
                                           method='Newton-CG', jac=True, hess=self.theta_Hessian)
                self.eta_theta_tilde[:] = np.log(self.eta_theta)
        return result

    def optimize_x(self):
        i = self.i
        opt_method = self.config.opt_method
        if opt_method == 'BFGS':
            if self.use_X_X_tilde:
                result = optimize.minimize(fun=self.x_objective_and_gradient, x0=self.eta_X_tilde[i],
                                           method=opt_method, jac=True)
                self.eta_X[i] = np.exp(self.eta_X_tilde[i])
            else:
                result = optimize.minimize(fun=self.x_objective_and_gradient, x0=self.eta_X[i],
                                           method=opt_method, jac=True)
                self.eta_X_tilde[i] = np.log(self.eta_X[i])
        else:
            if self.use_X_X_tilde:
                result = optimize.minimize(fun=self.x_objective_and_gradient, x0=self.eta_X_tilde[i],
                                           method=opt_method, jac=True, hess=self.x_Hessian)
                self.eta_X[i] = np.exp(self.eta_X_tilde[i])
            else:
                result = optimize.minimize(fun=self.x_objective_and_gradient, x0=self.eta_X[i],
                                           method=opt_method, jac=True, hess=self.x_Hessian)
                self.eta_X_tilde[i] = np.log(self.eta_X[i])
        return result

    def finalize_optimization(self):
        self.Xi_theta[:] = 0
        self.Xi_X[:] = 0

    def theta_objective_and_gradient(self, theta):
        if self.use_theta_theta_tilde:
            self.eta_theta_tilde[:] = theta
        else:
            self.eta_theta[:] = theta
        objective = self.theta_objective_func(*self.eta_theta_args)
        gradient = self.theta_gradient_func(*self.eta_theta_args).ravel()
        return objective, gradient

    def theta_Hessian(self, theta):
        return self.theta_Hessian_func(*self.eta_theta_args)

    def x_objective_and_gradient(self, x):
        if self.use_X_X_tilde:
            self.eta_X_tilde[self.i] = x
        else:
            self.eta_X[self.i] = x
        objective = self.X_objectives_func[self.i](*self.eta_X_args)
        objective += self.X_common_objective_func(*self.eta_X_args)
        gradient = self.X_gradients_func[self.i](*self.eta_X_args).ravel()
        return objective, gradient

    def x_Hessian(self, x):
        return self.X_Hessians_func[self.i](*self.eta_X_args)
