import numpy as np

from scipy import optimize

from core.laplace_mean_field import LaplaceMeanField
from dynamicals import OrnsteinUhlenbeck


class LaplaceMeanFieldSDE(LaplaceMeanField):
    def __init__(self, dynamical, config, gp):
        super(LaplaceMeanFieldSDE, self).__init__(dynamical, config, gp)

        num_x = self.dynamical.num_x
        num_theta = self.dynamical.num_theta
        num_est_t = self.config.est_tps.size

        self.mx = np.empty((num_x, num_est_t))  # m.dot(x)
        self.F = np.empty_like(self.mx)  # F(X, theta)
        self.F_mx = np.empty_like(self.mx)  # F - mx
        self.Lambda_F_mx = np.empty((num_x, num_est_t))  # Lambda.dot(F - mx)
        self.F_mx_Lambda_F_mx = np.empty(num_x)  # (F - mx).T.dot(Lambda).dot(F - mx)

        self.theta_B = np.empty((num_x, num_est_t, num_theta))
        self.theta_b = np.empty((num_x, num_est_t))
        self.theta_B_Lambda = np.empty((num_x, num_theta, num_est_t))  # theta_B.dot(Lambda)

        self.inv_Xi_theta = np.empty_like(self.Xi_theta)
        self.Psi_theta = np.empty((num_x, num_theta))

        self.x_objectives = np.empty(num_x)
        self.x_gradients = np.empty((num_x, num_est_t))
        self.x_Hessians = np.empty((num_x, num_est_t, num_est_t))
        self.diagonal = np.zeros((num_est_t, num_est_t))
        self.diagonal_indices = np.diag_indices_from(self.diagonal)

        self.ou_X = np.empty((num_x, num_est_t))

    def init_optimization(self):
        num_x = self.dynamical.num_x

        ou = OrnsteinUhlenbeck()
        ou_theta = np.array([1., 0.])  # Set theta = 1, mu = 0 for the Ornstein-Uhlenbeck process
        for i in range(num_x):
            # Sample the initial states from N(mu, rho ** 2 / (2 * theta)) to generate stationary sample path
            ou_X_0 = np.random.normal(ou_theta[1], np.sqrt(self.config.rho_2[i] / (2 * ou_theta[0])), 1)
            ou_rho_2 = np.array([self.config.rho_2[i]])
            self.ou_X[i] = ou.generate_sample_path(
                ou_theta, ou_rho_2, ou_X_0, self.config.spl_tps).ravel()[self.config.est_t_indices]

        self.eta_X[:] = self.gp.mu.copy()
        self.Xi_X[:] = self.gp.Sigma.copy()
        self.i = 0

        self.eta_theta[:] = 0.
        self.Xi_theta[:] = 0.
        self.inv_Xi_theta[:] = 0.
        self.Psi_theta[:] = 0.

        self.x_objectives[:] = 0.
        self.x_gradients[:] = 0.
        self.x_Hessians[:] = 0.

        for i in range(num_x):
            self.gp.m[i].dot(self.eta_X[i], out=self.mx[i])

        self.optimize_theta()

        for i in range(num_x):
            self.F[i] = self.dynamical.F_funcs[i](self.eta_X + self.ou_X, self.eta_theta) + self.ou_X[i]
            self.F_mx[i] = self.F[i] - self.mx[i]
            self.gp.Lambda[i].dot(self.F_mx[i], out=self.Lambda_F_mx[i])
            self.F_mx_Lambda_F_mx[i] = self.F_mx[i].dot(self.Lambda_F_mx[i])

    def optimize_theta(self):
        self.inv_Xi_theta[:] = 0.
        for i in range(self.dynamical.num_x):
            for n in range(self.dynamical.num_theta):
                self.theta_B[i, :, n] = self.dynamical.theta_B_funcs[i][n](self.eta_X + self.ou_X)
            self.theta_b[i] = self.dynamical.theta_b_funcs[i](self.eta_X + self.ou_X) + self.ou_X[i]

            self.theta_B[i].T.dot(self.gp.Lambda[i], out=self.theta_B_Lambda[i])
            self.theta_B_Lambda[i].dot(self.mx[i] - self.theta_b[i], out=self.Psi_theta[i])
            self.inv_Xi_theta += self.theta_B_Lambda[i].dot(self.theta_B[i])
            self.Xi_theta = np.linalg.pinv(self.inv_Xi_theta)
            self.Xi_theta.dot(self.Psi_theta.sum(axis=0), out=self.eta_theta)
            np.absolute(self.eta_theta, out=self.eta_theta)

    def optimize_x(self):
        if self.config.opt_method == 'BFGS':
            return optimize.minimize(fun=self.x_objective_and_gradient, x0=self.eta_X[self.i],
                                     method=self.config.opt_method, jac=True)
        else:
            return optimize.minimize(fun=self.x_objective_and_gradient, x0=self.eta_X[self.i],
                                     method=self.config.opt_method, jac=True, hess=self.x_Hessian)

    def finalize_optimization(self):
        self.Xi_theta = 0.5 * (self.Xi_theta + self.Xi_theta.T)  # Ensure symmetry
        for self.i in range(self.dynamical.num_x):
            self.x_objective_and_gradient(self.eta_X[self.i])
            self.Xi_X[self.i] = np.linalg.pinv(self.x_Hessian(self.eta_X[self.i]))
            self.Xi_X[self.i] = 0.5 * (self.Xi_X[self.i] + self.Xi_X[self.i].T)  # Ensure symmetry

    def theta_objective_and_gradient(self, x):
        pass

    def theta_Hessian(self, x):
        pass

    def x_objective_and_gradient(self, x):
        i = self.i
        self.eta_X[i] = x
        self.gp.m[i].dot(self.eta_X[i], out=self.mx[i])
        if self.dynamical.x_F_relation[i]:
            for j in self.dynamical.x_F_relation[i]:
                self.F[j] = self.dynamical.F_funcs[j](self.eta_X + self.ou_X, self.eta_theta) + self.ou_X[j]
                self.F_mx[j] = self.F[j] - self.mx[j]
                self.gp.Lambda[j].dot(self.F_mx[j], out=self.Lambda_F_mx[j])
                self.F_mx_Lambda_F_mx[j] = self.F_mx[j].dot(self.Lambda_F_mx[j])
        else:
            self.F[i] = self.dynamical.F_funcs[i](self.eta_X + self.ou_X, self.eta_theta) + self.ou_X[i]
            self.F_mx[i] = self.F[i] - self.mx[i]
            self.gp.Lambda[i].dot(self.F_mx[i], out=self.Lambda_F_mx[i])
            self.F_mx_Lambda_F_mx[i] = self.F_mx[i].dot(self.Lambda_F_mx[i])

        # Objective
        eta_X_i_mu_i = self.eta_X[i] - self.gp.mu[i]
        self.x_objectives[i] = 0.5 * eta_X_i_mu_i.dot(self.gp.inv_Sigma[i].dot(eta_X_i_mu_i))
        self.x_objectives[i] += 0.5 * self.F_mx_Lambda_F_mx.sum()

        # Gradient
        self.gp.inv_Sigma[i].dot(eta_X_i_mu_i, out=self.x_gradients[i])
        self.x_gradients[i] -= self.Lambda_F_mx[i].dot(self.gp.m[i])  # b.dot(a) is equivalent to a.T.dot(b)
        for j in self.dynamical.x_dF_relation[i]:
            # Element-wise multiplication is used here since the jacobian evaluated is necessarily a diagonal
            # matrix. A diagonal matrix multiplies a vector is equivalent to scaling corresponding elements of
            # the vector by the diagonal entries of the matrix.
            self.x_gradients[i] += (
                self.dynamical.x_dF_funcs[i][j](self.eta_X + self.ou_X, self.eta_theta) * self.Lambda_F_mx[j]
            )

        return self.x_objectives[i], self.x_gradients[i]

    def x_Hessian(self, x):
        i = self.i
        self.x_Hessians[i] = self.gp.inv_Sigma[i].copy()

        for j in self.dynamical.x_ddF_relation[i]:
            self.x_Hessians[i][self.diagonal_indices] += (
                self.dynamical.x_ddF_funcs[i][j](self.eta_X + self.ou_X, self.eta_theta) * self.Lambda_F_mx[j]
            )

        if self.dynamical.x_dF_relation[i]:
            for j in self.dynamical.x_dF_relation[i]:
                tmp = self.dynamical.x_dF_funcs[i][j](self.eta_X + self.ou_X, self.eta_theta)
                if j == i:
                    self.diagonal[self.diagonal_indices] = tmp
                    tmp = self.diagonal - self.gp.m[j].T
                    self.x_Hessians[i] += tmp.dot(self.gp.Lambda[j].dot(tmp.T))
                else:
                    if type(tmp) is np.ndarray:
                        self.x_Hessians[i] += tmp.reshape(-1, 1) * self.gp.Lambda[j] * tmp.reshape(1, -1)
                    else:
                        self.x_Hessians[i] += tmp * self.gp.Lambda[j] * tmp
        else:
            self.x_Hessians[i] -= self.gp.m[i].T.dot(self.gp.Lambda[i].dot(self.gp.m[i]))

        return self.x_Hessians[i]
