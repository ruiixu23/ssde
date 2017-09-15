import numpy as np

__labels = {
    't_0': '{}_t_0',
    't_T': '{}_t_T',
    'freq': '{}_freq',
}


def __get_label(label, group):
    return __labels[label].format(group)


def __assert_ints(t_0, t_T, freq, group):
    assert type(t_0) == int, 'Parameter {} must be an integer.'.format(__get_label('t_0', group))
    assert type(t_T) == int, 'Parameter {} must be an integer.'.format(__get_label('t_T', group))
    assert type(freq) == int, 'Parameter {} must be an integer.'.format(__get_label('freq', group))


def __assert_smaller(a, a_label, b, b_label):
    assert a < b, 'Parameter {} must be smaller than {}.'.format(a_label, b_label)


def __assert_smaller_equal(a, a_label, b, b_label):
    assert a <= b, 'Parameter {} must be smaller than or equal to {}.'.format(a_label, b_label)


def __assert_dividable(a, a_label, b, b_label):
    assert a % b == 0, 'Parameter {} must be dividable by {}.'.format(a_label, b_label)


def create_time(spl_t_0, spl_t_T, spl_freq, obs_t_0, obs_t_T, obs_freq, est_t_0, est_t_T, est_freq):
    __assert_ints(spl_t_0, spl_t_T, spl_freq, 'spl')
    __assert_smaller(spl_t_0, __get_label('t_0', 'spl'), spl_t_T, __get_label('t_T', 'spl'))

    __assert_ints(obs_t_0, obs_t_T, obs_freq, 'obs')
    __assert_smaller(obs_t_0, __get_label('t_0', 'obs'), obs_t_T, __get_label('t_T', 'obs'))

    __assert_ints(est_t_0, est_t_T, est_freq, 'est')
    __assert_smaller(est_t_0, __get_label('t_0', 'est'), est_t_T, __get_label('t_T', 'est'))

    __assert_dividable(spl_freq, __get_label('freq', 'spl'), obs_freq, __get_label('freq', 'obs'))
    __assert_smaller_equal(spl_t_0, __get_label('t_0', 'spl'), obs_t_0, __get_label('t_0', 'obs'))
    __assert_smaller_equal(obs_t_T, __get_label('t_T', 'obs'), spl_t_T, __get_label('t_T', 'spl'))

    __assert_dividable(spl_freq, __get_label('freq', 'spl'), est_freq, __get_label('freq', 'est'))
    __assert_smaller_equal(spl_t_0, __get_label('t_0', 'spl'), est_t_0, __get_label('t_0', 'est'))
    __assert_smaller_equal(est_t_T, __get_label('t_T', 'est'), spl_t_T, __get_label('t_T', 'spl'))

    spl_tps = np.linspace(spl_t_0, spl_t_T, (spl_t_T - spl_t_0) * spl_freq + 1, endpoint=True)

    obs_t_indices = np.arange(
        obs_t_0 * spl_freq, obs_t_T * spl_freq + int(spl_freq / obs_freq), int(spl_freq / obs_freq))
    obs_tps = spl_tps[obs_t_indices]

    est_t_indices = np.arange(
        est_t_0 * spl_freq, est_t_T * spl_freq + int(spl_freq / est_freq), int(spl_freq / est_freq))
    est_tps = spl_tps[est_t_indices]

    return spl_tps, obs_tps, obs_t_indices, est_tps, est_t_indices


def create_time_points(t_0, t_T, freq):
    return np.linspace(t_0, t_T, (t_T - t_0) * freq + 1, endpoint=True)
