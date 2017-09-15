import os
import pickle
import sys
# Enable module import from the parent directory from notebooks
sys.path.append(os.path.abspath('..'))
import time

import matplotlib as mpl
# Select plotting backend
mpl.use('nbAgg')

import matplotlib.pyplot as plt
# Customize plotting
plt.style.use('seaborn-paper')
plt.rcParams['axes.labelsize'] = 11.0
plt.rcParams['axes.titlesize'] = 12.0
plt.rcParams['errorbar.capsize'] = 3.0
plt.rcParams['figure.dpi'] = 72.0
plt.rcParams['figure.titlesize'] = 12.0
plt.rcParams['legend.fontsize'] = 10.
plt.rcParams['lines.linewidth'] = 1.
plt.rcParams['xtick.labelsize'] = 11.0
plt.rcParams['ytick.labelsize'] = 11.0

import numpy as np
import sympy as sp
sp.init_printing(euler=True, use_latex=True)

from IPython import display
from scipy import io, optimize
from sklearn import metrics

import core
import dynamicals
import kernels
import numericals
import utils