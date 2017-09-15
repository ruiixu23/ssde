import numpy as np


def collect_observations(spl_X, obs_t_indices, sigma_2=None, constraints=None):
    obs_Y = spl_X[:, obs_t_indices].copy()

    if sigma_2 is not None:
        num_x, _ = spl_X.shape
        num_obs_t = obs_t_indices.size

        if constraints is None:
            assert len(sigma_2) == num_x, \
                'The size of obs_noise_vars must equal to the number of states {}.'.format(num_x)

            if num_x == 1:
                obs_Y += np.sqrt(sigma_2[0]) * np.random.standard_normal((1, num_obs_t))
            else:
                mean = np.zeros(num_x)
                cov = np.diag(np.sqrt(sigma_2))
                obs_Y += np.random.multivariate_normal(mean, cov, num_obs_t).T
        else:
            lower, upper = constraints
            if lower is None:
                lower = -1 * np.Inf
            if upper is None:
                upper = np.Inf

            sigma = np.sqrt(sigma_2)
            for i in range(num_x):
                for n in range(num_obs_t):
                    while True:
                        tmp = obs_Y[i][n] + sigma[i] * np.random.standard_normal(1)[0]
                        if lower < tmp < upper:
                            obs_Y[i][n] = tmp
                            break
    return obs_Y
