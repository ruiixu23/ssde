import os
import sys

sys.path.append(os.path.abspath('..'))

import core
import dynamicals
import utils


def main():
    argv = sys.argv

    directory = argv[1]
    dynamical_name = argv[2].strip()
    repetition_num = argv[3]

    (spl_t_0, spl_t_T, spl_tps, spl_freq,
     obs_t_0, obs_t_T, obs_freq, obs_tps, obs_t_indices,
     est_t_0, est_t_T, est_freq, est_tps, est_t_indices,
     X_0, theta, rho_2, phi, sigma_2, delta, gamma,
     opt_method, opt_tol, max_init_iter, max_iter, plotting_enabled, plotting_freq,
     spl_X, obs_Y) = utils.load_sde_config(directory, utils.CONFIG_FILENAME)

    if dynamical_name == 'lorenz-96':
        dynamical = dynamicals.Lorenz96(X_0.size)
    elif dynamical_name == 'lorenz-63':
        dynamical = dynamicals.Lorenz63()
    else:
        raise ValueError('Unknown dynamical system {}'.format(dynamical_name))

    data = core.laplace_mean_field(dynamical,
                                   spl_X, spl_tps,
                                   obs_Y, obs_tps, obs_t_indices,
                                   est_tps, est_t_indices,
                                   theta, rho_2, phi, sigma_2, delta, gamma,
                                   opt_method, opt_tol, max_init_iter, max_iter,
                                   plotting_enabled, plotting_freq)
    utils.save_data(directory, utils.DATA_FILENAME.format(repetition_num), data)

if __name__ == "__main__":
    main()