import numpy as np


def get_theta_mean(data):
    return np.mean([item['eta_theta'] for item in data], axis=0)


def get_theta_var(data):
    return np.var([item['eta_theta'] for item in data], axis=0, ddof=1)


def get_X_mean(data):
    return np.mean([item['eta_X'] for item in data], axis=0)


def get_X_var(data):
    return np.var([item['eta_X'] for item in data], axis=0, ddof=1)


def get_runtime_mean(data):
    return np.mean([item['runtime'] for item in data])


def get_runtime_var(data):
    return np.var([item['runtime'] for item in data], ddof=1)
