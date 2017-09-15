import numpy as np

import kernels
import numericals


class GaussianProcessRegression:
    def __init__(self, dynamical, config):
        self.dynamical = dynamical
        self.config = config

        num_x = self.dynamical.num_x
        num_est_t = self.config.est_tps.size

        self.mu = np.empty((num_x, num_est_t))
        self.Sigma = np.empty((num_x, num_est_t, num_est_t))
        self.inv_Sigma = np.empty_like(self.Sigma)

        self.m = np.empty((num_x, num_est_t, num_est_t))
        self.Lambda = np.empty((num_x, num_est_t, num_est_t))
        self.inv_Lambda = np.empty_like(self.Lambda)

    def run(self):
        self.run_x_regression()
        self.run_dx_regression()

    def run_x_regression(self):
        num_x = self.dynamical.num_x
        obs_tps = self.config.obs_tps
        est_tps = self.config.est_tps

        is_grid_free = obs_tps.size == est_tps.size and np.allclose(obs_tps, est_tps)

        if is_grid_free:
            obs_I = np.eye(obs_tps.size)
        else:
            obs_I = None
        est_I = np.eye(est_tps.size)

        for i in range(num_x):
            name_i, phi_i = self.config.phi[i]
            kernel_i = kernels.KernelFactory.get_kernel(name_i)
            est_C_i = (
                kernel_i.C_func(est_tps.reshape(-1, 1), est_tps.reshape(1, -1), phi_i) +
                1e-10 * est_I  # Add small number to the diagonal to improve numerical stability
            )
            est_inv_C_i = numericals.cholesky_inv(est_C_i)

            if self.config.delta[i]:
                # The state is observable
                if is_grid_free:
                    # The grid free prediction uses the standard Gaussian process regression formula
                    #  -     -       -   - -    -                                       -  -
                    # | obs_Y | ~ N |   | 0 |  | obs_C_i + sigma_2_i * obs_I, obs_est_C_i  |  |
                    # | est_X |     |   | 0 |  | est_obs_C_i,               est_C_i      |  |
                    #  -     -       -   - - ,  -                                       -  -

                    obs_C_i = (
                        kernel_i.C_func(obs_tps.reshape(-1, 1), obs_tps.reshape(1, -1), phi_i) +
                        1e-10 * obs_I  # Add small value to the diagonal to improve numerical stability
                    )
                    obs_est_C_i = kernel_i.C_func(obs_tps.reshape(-1, 1), est_tps.reshape(1, -1), phi_i)
                    est_obs_C_i = kernel_i.C_func(est_tps.reshape(-1, 1), obs_tps.reshape(1, -1), phi_i)

                    tmp = est_obs_C_i.dot(numericals.cholesky_inv(obs_C_i + self.config.sigma_2[i] * obs_I))
                    tmp.dot(self.config.obs_Y[i], out=self.mu[i])
                    self.Sigma[i] = est_C_i - tmp.dot(obs_est_C_i)
                    self.Sigma[i] = 0.5 * (self.Sigma[i] + self.Sigma[i].T)  # Ensure symmetry
                    self.inv_Sigma[i] = numericals.cholesky_inv(self.Sigma[i])
                    self.inv_Sigma[i] = 0.5 * (self.inv_Sigma[i] + self.inv_Sigma[i].T)  # Ensure symmetry
                else:
                    # Uses the shortcut derived in (Eq. 3.16)
                    tmp = (1. / self.config.sigma_2[i]) * est_I + est_inv_C_i
                    self.inv_Sigma[i] = 0.5 * (tmp + tmp.T)  # Ensure symmetry
                    self.Sigma[i] = numericals.cholesky_inv(self.inv_Sigma[i])
                    self.Sigma[i] = 0.5 * (self.Sigma[i] + self.Sigma[i].T)  # Ensure symmetry
                    self.mu[i] = (1. / self.config.sigma_2[i]) * self.Sigma[i].dot(self.config.obs_Y[i])
            else:
                # The state is unobservable
                self.mu[i] = 0.
                self.Sigma[i] = 0.5 * (est_C_i + est_C_i.T)
                self.inv_Sigma[i] = 0.5 * (est_inv_C_i + est_inv_C_i.T)

    def run_dx_regression(self):
        num_x = self.dynamical.num_x
        est_tps = self.config.est_tps

        est_I = np.eye(est_tps.size)

        for i in range(num_x):
            name_i, phi_i = self.config.phi[i]
            kernel_i = kernels.KernelFactory.get_kernel(name_i)

            est_C_i = (
                kernel_i.C_func(est_tps.reshape(-1, 1), est_tps.reshape(1, -1), phi_i) +
                1e-10 * est_I  # Add small value to the diagonal to improve stability
            )
            est_inv_C_i = numericals.cholesky_inv(est_C_i)
            est_dC_i = kernel_i.dC_func(est_tps.reshape(-1, 1), est_tps.reshape(1, -1), phi_i)
            est_Cd_i = kernel_i.Cd_func(est_tps.reshape(-1, 1), est_tps.reshape(1, -1), phi_i)
            est_dCd_i = kernel_i.dCd_func(est_tps.reshape(-1, 1), est_tps.reshape(1, -1), phi_i)

            tmp = est_dC_i.dot(est_inv_C_i)
            self.m[i] = tmp
            self.inv_Lambda[i] = est_dCd_i - tmp.dot(est_Cd_i) + self.config.gamma[i] * est_I
            self.inv_Lambda[i] = 0.5 * (self.inv_Lambda[i] + self.inv_Lambda[i].T)  # Ensure symmetry
            self.Lambda[i] = numericals.cholesky_inv(self.inv_Lambda[i])
            self.Lambda[i] = 0.5 * (self.Lambda[i] + self.Lambda[i].T)  # Ensure symmetry
