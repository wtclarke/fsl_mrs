# ------------------------------------------------------------------------
# User file for defining a model

# Parameter behaviour
# 'variable' : one per time point
# 'fixed'    : same for all time points
# 'dynamic'  : model-based change across time points

Parameters = {
   'Phi_0'    : 'fixed',
   'Phi_1'    : 'fixed',
   'conc'     : {'Met1': 'fixed',
                 'Met2': {'dynamic': 'model_exp', 'params': ['c_amp', 'c_adc']},
                 'Met3': {'dynamic': 'model_lin', 'params': ['c_amp', 'c_lin']},
                 'other': 'variable'},
   'eps'      : 'fixed',
   'gamma'    : {'1': 'variable',
                 'other': 'fixed'},
   'sigma'    : 'fixed',
   'baseline' : 'fixed'
}

Bounds = {
    'c_amp': (0, None),
    'c_adc': (0, 10),
    'c_lin': (0, 1),
    'gamma': (0, None),
    'sigma': (0, None)
}

# Dynamic models here
from numpy import exp
from numpy import asarray, ones_like


# Exponential model
def model_exp(p, t):
    # p = [amp,adc]
    return p[0] * exp(-p[1] * t)


# Linear model
def model_lin(p, t):
    # p = [amp,adc]
    return p[0] - p[1] * t

# ------------------------------------------------------------------------
# Gradients


def model_exp_grad(p, t):
    e1 = exp(-p[1] * t)
    g0 = e1
    g1 = -t * p[0] * e1
    return asarray([g0, g1], dtype=object)


def model_lin_grad(p, t):
    g0 = ones_like(t)
    g1 = -t
    return asarray([g0, g1], dtype=object)
