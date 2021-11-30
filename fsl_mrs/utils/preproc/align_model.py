"""Model for dynamic fitting based alignment

Author: William Clarke <william.clarke@ndcn.ox.ac.uk>
Copyright (C) 2021 University of Oxford
"""

Parameters = {
    'Phi_0': 'variable',
    'Phi_1': 'fixed',
    'conc': 'fixed',
    'eps': 'variable',
    'gamma': 'fixed',
    'sigma': 'fixed',
    'baseline': 'fixed'
}

Bounds = {
    'gamma': (0, None),
    'sigma': (0, None)
}
