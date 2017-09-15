import matplotlib.pyplot as plt
import numpy as np


def _create_figure(num_rows, num_cols):
    return plt.figure(figsize=plt.figaspect(1 / num_cols) * np.array([1, num_rows]))


def _get_x_indices(dynamical, delta=None):
    num_x = dynamical.num_x
    if num_x <= 6:
        return np.arange(num_x)
    elif delta is None or np.alltrue(delta):
        return np.array([0, 1, 2, num_x - 3, num_x - 2, num_x - 1])
    else:
        unobserved_state_indices = np.random.permutation(np.where(delta <= 0)[0])[:3]
        observed_state_indices = np.random.permutation(np.where(delta > 0)[0])[:6 - unobserved_state_indices.size]
        return np.concatenate((unobserved_state_indices, observed_state_indices))


def _add_x_config(ax, config):
    if config is not None and 'x' in config:
        x_config = config['x']
        if 'xlim' in x_config:
            ax.set_xlim(*x_config['xlim'])
        if 'ylim' in config['x']:
            ax.set_ylim(*x_config['ylim'])


def create_estimation_figure(dynamical, delta):
    x_indices = _get_x_indices(dynamical, delta)
    x_labels = dynamical.x_labels
    num_cols = 3
    num_rows = int(np.ceil(len(x_indices) / num_cols)) + 1
    figure = _create_figure(num_rows, num_cols)

    config = {
        'num_cols': num_cols,
        'num_rows': num_rows,
        'x_indices': x_indices
    }

    for subplot_index, x_index in enumerate(x_indices):
        ax = figure.add_subplot(num_rows, num_cols, subplot_index + num_cols + 1)
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(r'State ${}$'.format(x_labels[x_index]))

    ax = figure.add_subplot(num_rows, num_cols, 1)
    ax.set_ylabel('Value')
    ax.set_title('Theta')

    figure.tight_layout()
    plt.show()
    plt.pause(0.001)
    figure.canvas.draw()

    return figure, config


def add_sample_path(figure, config, spl_X, spl_tps):
    x_indices = config['x_indices']
    num_cols = config['num_cols']
    num_rows = config['num_rows']

    for subplot_index, x_index in enumerate(x_indices):
        ax = figure.add_subplot(num_rows, num_cols, subplot_index + num_cols + 1)
        ax.plot(spl_tps, spl_X[x_index], color='C0', linestyle='-', linewidth=1.5, label='Sample path')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc=0)

    figure.canvas.draw()


def add_observations(figure, config, obs_Y, obs_tps, delta):
    x_indices = config['x_indices']
    num_cols = config['num_cols']
    num_rows = config['num_rows']

    for subplot_index, x_index in enumerate(x_indices):
        if not delta[x_index]:
            # Skip the unobserved states
            continue
        ax = figure.add_subplot(num_rows, num_cols, subplot_index + num_cols + 1)
        ax.scatter(obs_tps, obs_Y[x_index], marker='.', color='C1', label='Observation')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc=0)

    figure.canvas.draw()


def add_gaussian_regression_result(figure, config, est_X, est_tps):
    x_indices = config['x_indices']
    num_cols = config['num_cols']
    num_rows = config['num_rows']

    for subplot_index, x_index in enumerate(x_indices):
        ax = figure.add_subplot(num_rows, num_cols, subplot_index + num_cols + 1)
        ax.plot(est_tps, est_X[x_index], color='0.', linestyle='-.', linewidth=1.5, label='GP init')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc=0)

    figure.canvas.draw()


def _add_theta(figure, num_rows, num_cols, theta_labels, theta, est_theta=None, est_theta_cov=None):
    ax = figure.add_subplot(num_rows, num_cols, 1)
    ax.cla()
    bar_width = 0.3
    bar_indices = np.arange(theta.size)
    ax.bar(bar_indices, theta, bar_width, color='C0', label='Truth')
    if est_theta is None:
        est_theta = np.zeros_like(theta)
    if est_theta_cov is None:
        ax.bar(bar_indices + bar_width, est_theta, bar_width, color='C2', label='Estimation')
    else:
        ax.bar(bar_indices + bar_width, est_theta, bar_width, color='C2', label='Estimation',
               yerr=2 * np.sqrt(np.diagonal(est_theta_cov)),
               error_kw=dict(ecolor='0.2', capsize=3., capthick=1.))
    ax.set_ylabel('Value')
    ax.set_title('Theta')
    ax.set_xticks(bar_indices + bar_width / 2)
    ax.set_xticklabels([r'${}$'.format(label) for label in theta_labels])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, loc=0)


def add_estimation_step(figure, config, dynamical, est_X, est_tps, theta, est_theta):
    x_indices = config['x_indices']
    num_cols = config['num_cols']
    num_rows = config['num_rows']

    for subplot_index, x_index in enumerate(x_indices):
        ax = figure.add_subplot(num_rows, num_cols, subplot_index + num_cols + 1)
        ax.plot(est_tps, est_X[x_index], color='0.6', linestyle='--', linewidth=0.8)

    _add_theta(figure, num_rows, num_cols, dynamical.theta_labels, theta, est_theta)

    figure.canvas.draw()


def add_estimation_result(figure, config, dynamical, est_X, est_tps, theta, est_theta):
    x_indices = config['x_indices']
    num_cols = config['num_cols']
    num_rows = config['num_rows']

    for subplot_index, x_index in enumerate(x_indices):
        ax = figure.add_subplot(num_rows, num_cols, subplot_index + num_cols + 1)
        ax.plot(est_tps, est_X[x_index], color='C2', linestyle='-', linewidth=1.5, label='Estimation')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc=0)

    _add_theta(figure, num_rows, num_cols, dynamical.theta_labels, theta, est_theta)

    figure.canvas.draw()


def plot_estimation_result(config, dynamical, spl_X, spl_tps, obs_Y, obs_tps, delta, est_X, est_X_cov, est_tps,
                           theta, est_theta, est_theta_cov):
    x_indices = config['x_indices']
    state_labels = dynamical.x_labels

    num_cols = 3
    num_rows = int(np.ceil(len(x_indices) / num_cols)) + 1
    figure = _create_figure(num_rows, num_cols)

    # Plot states
    for subplot_index, x_index in enumerate(x_indices):
        ax = figure.add_subplot(num_rows, num_cols, subplot_index + num_cols + 1)
        ax.plot(spl_tps, spl_X[x_index], color='C0', linestyle='-', label='Sample path')

        if delta is None or delta[x_index]:
            ax.scatter(obs_tps, obs_Y[x_index], color='C1', marker='.', label='Observation')

        ax = figure.add_subplot(num_rows, num_cols, subplot_index + num_cols + 1)
        ax.errorbar(est_tps, est_X[x_index], color='C2', linestyle='-', label='Estimation',
                    yerr=2 * np.sqrt(np.diagonal(est_X_cov[x_index])),
                    ecolor='0.2', capsize=3., capthick=1.)

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(r'State ${}$'.format(state_labels[x_index]))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc=0)

    # Plot parameters
    _add_theta(figure, num_rows, num_cols, dynamical.theta_labels, theta, est_theta, est_theta_cov)

    figure.tight_layout()
    plt.show()
    plt.pause(0.001)
    figure.canvas.draw()


def plot_kernels(dynamical, x, covs):
    num_x = covs.shape[0]
    if num_x <= 6:
        x_indices = np.arange(num_x)
    else:
        x_indices = np.array([0, 1, 2, num_x - 3, num_x - 2, num_x - 1])

    if dynamical:
        state_labels = dynamical.x_labels
    else:
        state_labels = None

    num_cols = 3
    num_rows = int(np.ceil(len(x_indices) / num_cols))
    figure = _create_figure(num_rows, num_cols)
    mean = np.zeros(x.size)

    # Plot kernels
    for subplot_index, state_index in enumerate(x_indices):
        ax = figure.add_subplot(num_rows, num_cols, subplot_index + 1)
        samples = np.random.multivariate_normal(mean, covs[state_index], 20)
        for j in range(0, samples.shape[0]):
            ax.plot(x, samples[j, :], linewidth=1., color='0.85')
        samples = np.random.multivariate_normal(mean, covs[state_index], 3)
        for j in range(0, samples.shape[0]):
            ax.plot(x, samples[j, :], linewidth=1.7, color='C{}'.format(j))
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        if state_labels:
            ax.set_title(r'Kernel for State ${}$'.format(state_labels[state_index]))

    figure.tight_layout()
    plt.show()
    plt.pause(0.001)
    figure.canvas.draw()


def plot_kernel(x, cov):
    figure = _create_figure(1, 1)
    ax = figure.add_subplot(1, 1, 1)
    mean = np.zeros(x.size)
    samples = np.random.multivariate_normal(mean, cov, 20)
    for i in range(0, samples.shape[0]):
        ax.plot(x, samples[i, :], linewidth=1., color='0.85')
    samples = np.random.multivariate_normal(mean, cov, 3)
    for i in range(0, samples.shape[0]):
        ax.plot(x, samples[i, :], linewidth=1.7, color='C{}'.format(i))
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

    figure.tight_layout()
    plt.show()
    plt.pause(0.001)
    figure.canvas.draw()


def plot_states(dynamical, spl_X, spl_tps, obs_Y=None, obs_tps=None, delta=None, est_X=None, est_tps=None, config=None):
    x_indices = _get_x_indices(dynamical, None)
    x_labels = dynamical.x_labels

    num_cols = 3
    num_rows = int(np.ceil(len(x_indices) / num_cols))
    figure = _create_figure(num_rows, num_cols)

    # Plot states
    for subplot_index, state_index in enumerate(x_indices):
        ax = figure.add_subplot(num_rows, num_cols, subplot_index + 1)
        ax.plot(spl_tps, spl_X[state_index], color='C0', linestyle='-', linewidth=1.5, label='Sample path')

        if obs_Y is not None:
            if delta is None or delta[state_index]:
                ax.scatter(obs_tps, obs_Y[state_index], color='C1', marker='.', label='Observation')

        if est_X is not None:
            ax.plot(est_tps, est_X[state_index], color='C2', linestyle='-', linewidth=1.5, label='Estimation')

        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(r'State ${}$'.format(x_labels[state_index]))
        _add_x_config(ax, config)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, loc=0)

    figure.tight_layout()
    plt.show()
    plt.pause(0.001)
    figure.canvas.draw()


def plot_states_lorenz_63(dynamical, spl_X, obs_Y=None, est_X=None):
    from mpl_toolkits.mplot3d import Axes3D

    state_labels = dynamical.x_labels
    figure = plt.figure(figsize=plt.figaspect(.9 / 1.))
    ax = figure.gca(projection='3d')
    ax.view_init(elev=15)
    ax.plot(spl_X[0], spl_X[1], spl_X[2], color='C0', linestyle='-', linewidth=1.5, label='Sample path')

    if obs_Y is not None:
        ax.scatter(obs_Y[0], obs_Y[1], obs_Y[2], color='C1', marker='.', label='Observation')

    if est_X is not None:
        ax.plot(est_X[0], est_X[1], est_X[2], color='C2', linestyle='-', linewidth=1.5, label='Estimation')

    ax.set_xlabel(r'State ${}$'.format(state_labels[0]))
    ax.set_ylabel(r'State ${}$'.format(state_labels[1]))
    ax.set_zlabel(r'State ${}$'.format(state_labels[2]))
    ax.set_title('Lorenz 63')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.tight_layout()
    plt.show()
