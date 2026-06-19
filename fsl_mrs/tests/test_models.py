'''FSL-MRS test script

Test model functions

Copyright Will Clarke, University of Oxford, 2022'''

import fsl_mrs.models as models
import numpy as np

all_models = [
    'lorentzian',
    'voigt',
    'free_shift',
    'free_shift_lorentzian',
    'negativevoigt']
modules = [
    models.lorentzian,
    models.voigt,
    models.freeshift,
    models.freeshift_lorentzian,
    models.negativevoigt]


def test_getModelFunctions():
    for model, mod in zip(all_models, modules):
        functions = models.getModelFunctions(model)
        assert len(functions) == 5
        assert mod.err == functions[0]
        assert mod.grad == functions[1]
        assert mod.forward == functions[2]
        assert mod.x2param == functions[3]
        assert mod.param2x == functions[4]


def test_getModelForward():
    for model, mod in zip(all_models, modules):
        function = models.getModelForward(model)
        assert mod.forward == function


def test_getModelJac():
    for model, mod in zip(all_models, modules):
        function = models.getModelJac(model)
        assert mod.jac == function


def test_getInit():
    for model, mod in zip(all_models, modules):
        function = models.getInit(model)
        assert mod.init == function


# A test case
# Simulated for ten basis spectra, two groups and polynomial baseline order
# of 2 (6 coeffs)
n_basis = 10
n_groups = 2
b_order = 2
n_baseline = b_order + 1

answer_names = []
answer_sizes = []
# Lorentzian model
answer_names.append(['conc', 'gamma', 'eps', 'Phi_0', 'Phi_1', 'baseline'])
answer_sizes.append([10, 2, 2, 1, 1, 6])

# Voigt model
answer_names.append(['conc', 'gamma', 'sigma', 'eps', 'Phi_0', 'Phi_1', 'baseline'])
answer_sizes.append([10, 2, 2, 2, 1, 1, 6])

# Freeshift model
# Voigt + an eps term for each basis (i.e. 10)
answer_names.append(['conc', 'gamma', 'sigma', 'eps', 'Phi_0', 'Phi_1', 'baseline'])
answer_sizes.append([10, 2, 2, 10, 1, 1, 6])

# Freeshift lorentzian model
# Lorentzian + an eps term for each basis (i.e. 10)
answer_names.append(['conc', 'gamma', 'eps', 'Phi_0', 'Phi_1', 'baseline'])
answer_sizes.append([10, 2, 10, 1, 1, 6])

# Negative voigt  model
# same as voigt
answer_names.append(['conc', 'gamma', 'sigma', 'eps', 'Phi_0', 'Phi_1', 'baseline'])
answer_sizes.append([10, 2, 2, 2, 1, 1, 6])


def test_FSLModel_vars():
    for model, ans_n, ans_s in zip(all_models, answer_names, answer_sizes):
        names, sizes = models.FSLModel_vars(model, n_basis, n_groups, n_baseline)
        assert names == ans_n
        assert sizes == ans_s


def test_FSLModel_bounds():
    for model, ans_s in zip(all_models, answer_sizes):
        # Newton
        bounds = models.FSLModel_bounds(model, n_basis, n_groups, n_baseline, 'Newton')
        assert len(bounds) == sum(ans_s)

        # MH
        LB, UB = models.FSLModel_bounds(model, n_basis, n_groups, n_baseline, 'MH')
        assert len(LB) == sum(ans_s)
        assert len(UB) == sum(ans_s)


def test_FSLModel_mask():
    for model, ans_s in zip(all_models, answer_sizes):
        mask = models.FSLModel_mask(model, n_basis, n_groups, n_baseline)
        # Final 6 (baseline) parameters should be zero
        assert sum(mask[:-6]) == len(mask[:-6])
        assert sum(mask[-6:]) == 0
        assert len(mask) == sum(ans_s)


def _gradient_from_jacobian(mod, x, nu, t, m, B, G, g, data, indices):
    """Previous concatenated-Jacobian gradient formula."""
    S = mod.forward(x, nu, t, m, B, G, g)[indices, None]
    dS = mod.jac(x, nu, t, m, B, G, g, indices)
    Spec = data[indices, None]

    return np.real(
        np.sum(
            S * np.conj(dS)
            + np.conj(S) * dS
            - np.conj(Spec) * dS
            - Spec * np.conj(dS),
            axis=0))


def _random_model_parameters(mod, rng, n_basis, n_groups, n_baseline):
    con = rng.uniform(0.1, 2.0, n_basis)
    gamma = rng.uniform(0.01, 0.2, n_groups)
    sigma = rng.uniform(0.01, 0.2, n_groups)
    eps_groups = rng.uniform(-0.05, 0.05, n_groups)
    eps_basis = rng.uniform(-0.05, 0.05, n_basis)
    baseline = rng.normal(size=n_baseline)

    if mod is models.lorentzian:
        return mod.param2x(con, gamma, eps_groups, 0.1, 1E-5, baseline)
    if mod is models.voigt or mod is models.negativevoigt:
        return mod.param2x(con, gamma, sigma, eps_groups, 0.1, 1E-5, baseline)
    if mod is models.freeshift:
        return mod.param2x(con, gamma, sigma, eps_basis, 0.1, 1E-5, baseline)
    if mod is models.freeshift_lorentzian:
        return mod.param2x(con, gamma, eps_basis, 0.1, 1E-5, baseline)
    raise ValueError(f'Unexpected model module {mod}.')


def test_model_grads_match_concatenate_implementation():
    rng = np.random.default_rng(20260617)
    n_points = 128
    n_basis = 5
    n_groups = 2
    n_baseline_params = 4

    nu = np.linspace(-2000.0, 2000.0, n_points)[:, None]
    t = (np.arange(n_points) / 4000.0)[:, None]
    m = rng.normal(size=(n_points, n_basis)) + 1j * rng.normal(size=(n_points, n_basis))
    B = rng.normal(size=(n_points, n_baseline_params))\
        + 1j * rng.normal(size=(n_points, n_baseline_params))
    G = [0, 1, 0, 1, 1]
    data = rng.normal(size=n_points) + 1j * rng.normal(size=n_points)
    indices = slice(10, 102)

    for mod in modules:
        x = _random_model_parameters(mod, rng, n_basis, n_groups, n_baseline_params)
        new_grad = mod.grad(x, nu, t, m, B, G, n_groups, data, indices)
        old_grad = _gradient_from_jacobian(mod, x, nu, t, m, B, G, n_groups, data, indices)

        assert np.allclose(new_grad, old_grad)

# TO DO test getFittedModel
