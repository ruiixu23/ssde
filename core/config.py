import utils


class Config:
    def __init__(self):
        self.spl_t_0 = None
        self.spl_t_T = None
        self.spl_freq = None
        self.spl_tps = None

        self.obs_t_0 = None
        self.obs_t_T = None
        self.obs_freq = None
        self.obs_tps = None
        self.obs_t_indices = None

        self.est_t_0 = None
        self.est_t_T = None
        self.est_freq = None
        self.est_tps = None
        self.est_t_indices = None

        self.X_0 = None
        self.theta = None
        self.rho_2 = None

        self.phi = None
        self.sigma_2 = None
        self.delta = None
        self.gamma = None

        self.opt_method = None
        self.opt_tol = None
        self.max_init_iter = None
        self.max_iter = None

        self.plotting_enabled = None
        self.plotting_freq = None

        self.debug = None

        self.spl_X = None
        self.obs_Y = None

    def create_time(self, spl_t_0, spl_t_T, spl_freq, obs_t_0, obs_t_T, obs_freq, est_t_0, est_t_T, est_freq):
        spl_tps, obs_tps, obs_t_indices, est_tps, est_t_indices = utils.create_time(
            spl_t_0, spl_t_T, spl_freq, obs_t_0, obs_t_T, obs_freq, est_t_0, est_t_T, est_freq)

        self.spl_t_0 = spl_t_0
        self.spl_t_T = spl_t_T
        self.spl_freq = spl_freq
        self.spl_tps = spl_tps

        self.obs_t_0 = obs_t_0
        self.obs_t_T = obs_t_T
        self.obs_freq = obs_freq
        self.obs_tps = obs_tps
        self.obs_t_indices = obs_t_indices

        self.est_t_0 = est_t_0
        self.est_t_T = est_t_T
        self.est_freq = est_freq
        self.est_tps = est_tps
        self.est_t_indices = est_t_indices

    def load_config(self, directory, filename):
        config = utils.load_data(directory, filename)
        self.spl_t_0 = config['spl_t_0']
        self.spl_t_T = config['spl_t_T']
        self.spl_freq = config['spl_freq']
        self.spl_tps = config['spl_tps']

        self.obs_t_0 = config['obs_t_0']
        self.obs_t_T = config['obs_t_T']
        self.obs_freq = config['obs_freq']
        self.obs_tps = config['obs_tps']
        self.obs_t_indices = config['obs_t_indices']

        self.est_t_0 = config['est_t_0']
        self.est_t_T = config['est_t_T']
        self.est_freq = config['est_freq']
        self.est_tps = config['est_tps']
        self.est_t_indices = config['est_t_indices']

        self.X_0 = config['X_0']
        self.theta = config['theta']
        self.rho_2 = config['rho_2']

        self.phi = config['phi']
        self.sigma_2 = config['sigma_2']
        self.delta = config['delta']
        self.gamma = config['gamma']

        self.opt_method = config['opt_method']
        self.opt_tol = config['opt_tol']

        self.max_init_iter = config['max_init_iter']
        self.max_iter = config['max_iter']

        self.plotting_enabled = config['plotting_enabled']
        self.plotting_freq = config['plotting_freq']

        self.spl_X = config['spl_X']
        if 'obs_Y' in config:
            self.obs_Y = config['obs_Y']

    def save_config(self, directory, filename):
        config = {
            'spl_t_0': self.spl_t_0,
            'spl_t_T': self.spl_t_T,
            'spl_freq': self.spl_freq,
            'spl_tps': self.spl_tps,

            'obs_t_0': self.obs_t_0,
            'obs_t_T': self.obs_t_T,
            'obs_freq': self.obs_freq,
            'obs_tps': self.obs_tps,
            'obs_t_indices': self.obs_t_indices,

            'est_t_0': self.est_t_0,
            'est_t_T': self.est_t_T,
            'est_freq': self.est_freq,
            'est_tps': self.est_tps,
            'est_t_indices': self.est_t_indices,

            'X_0': self.X_0,
            'theta': self.theta,
            'rho_2': self.rho_2,

            'phi': self.phi,
            'sigma_2': self.sigma_2,
            'delta': self.delta,
            'gamma': self.gamma,

            'opt_method': self.opt_method,
            'opt_tol': self.opt_tol,

            'max_init_iter': self.max_init_iter,
            'max_iter': self.max_iter,

            'plotting_enabled': self.plotting_enabled,
            'plotting_freq': self.plotting_freq,

            'spl_X': self.spl_X
        }

        if self.obs_Y is not None:
            config['obs_Y'] = self.obs_Y

        utils.save_data(directory, filename, config)
