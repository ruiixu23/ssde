import abc
import time
import utils

import numpy as np


class LaplaceMeanField(abc.ABC):
    def __init__(self, dynamical, config, gp):
        self.dynamical = dynamical
        self.config = config
        self.gp = gp

        num_x = self.dynamical.num_x
        num_theta = self.dynamical.num_theta
        num_est_t = self.config.est_tps.size

        self.eta_X = np.empty((num_x, num_est_t))
        self.Xi_X = np.empty((num_x, num_est_t, num_est_t))
        self.i = 0

        self.eta_theta = np.empty(num_theta)
        self.Xi_theta = np.empty((num_theta, num_theta))

        self.runtime = 0

    def run(self):
        self.init_optimization()

        if self.config.opt_method not in ['BFGS', 'Newton-CG', 'dogleg', 'trust-ncg']:
            raise RuntimeError('Invalid optimization method.')

        if self.config.plotting_enabled is True:
            figure, plotting_config = utils.create_estimation_figure(self.dynamical, self.config.delta)
            utils.add_sample_path(figure, plotting_config, self.config.spl_X, self.config.spl_tps)
            utils.add_observations(figure, plotting_config, self.config.obs_Y, self.config.obs_tps, self.config.delta)
            utils.add_gaussian_regression_result(figure, plotting_config, self.eta_X, self.config.est_tps)
        else:
            figure = None
            plotting_config = None

        is_all_observed = np.alltrue(self.config.delta)
        if self.config.max_init_iter is None:
            self.config.max_init_iter = 0

        it = 1
        objective = 0
        new_objective = 0
        tic = time.time()
        tac = tic
        while it <= self.config.max_iter:
            new_objective = 0
            success = True

            result = self.optimize_theta()
            if result is not None:
                new_objective += result.fun
                if result.success is False:
                    success = False

            for self.i in range(self.dynamical.num_x):
                if not is_all_observed and self.config.delta[self.i] and it <= self.config.max_init_iter:
                    continue
                result = self.optimize_x()
                if result is not None:
                    new_objective += result.fun
                    if result.success is False:
                        success = False

            tac = time.time()

            if success:
                print('.', end='')
            else:
                print('^', end='')

            if it % self.config.plotting_freq == 0:
                print(' Iteration: {}, Objective {:.4f}, Runtime {:.2f}s.'.format(it, new_objective, tac - tic))
                if figure and plotting_config:
                    utils.add_estimation_step(figure, plotting_config, self.dynamical, self.eta_X, self.config.est_tps,
                                              self.config.theta, self.eta_theta)

            if abs(new_objective - objective) < self.config.opt_tol:
                if is_all_observed or it > self.config.max_init_iter:
                    break
                else:
                    it = self.config.max_init_iter

            objective = new_objective
            it += 1

        self.runtime = tac - tic

        print()
        print('Objective {:.4f}, Runtime {:.2f}s.'.format(new_objective, self.runtime))

        self.finalize_optimization()

        if figure and plotting_config:
            utils.add_estimation_result(figure, plotting_config, self.dynamical, self.eta_X, self.config.est_tps,
                                        self.config.theta, self.eta_theta)
            utils.plot_estimation_result(plotting_config, self.dynamical, self.config.spl_X, self.config.spl_tps,
                                         self.config.obs_Y, self.config.obs_tps, self.config.delta,
                                         self.eta_X, self.Xi_X, self.config.est_tps,
                                         self.config.theta, self.eta_theta, self.Xi_theta)

    def save_result(self, directory, filename):
        data = {
            'eta_X': self.eta_X,
            'Xi_X': self.Xi_X,
            'eta_theta': self.eta_theta,
            'Xi_theta': self.Xi_theta,
            'runtime': self.runtime,
            'obs_Y': self.config.obs_Y
        }
        utils.save_data(directory, filename, data)

    @abc.abstractmethod
    def init_optimization(self):
        pass

    @abc.abstractmethod
    def optimize_theta(self):
        pass

    @abc.abstractmethod
    def optimize_x(self):
        pass

    @abc.abstractmethod
    def finalize_optimization(self):
        pass

    @abc.abstractmethod
    def theta_objective_and_gradient(self, theta):
        pass

    @abc.abstractmethod
    def theta_Hessian(self, theta):
        pass

    @abc.abstractmethod
    def x_objective_and_gradient(self, x):
        pass

    @abc.abstractmethod
    def x_Hessian(self, x):
        pass
