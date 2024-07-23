'''FSL-MRS test script

Test model functions

Copyright Will Clarke, University of Oxford, 2022'''

import fsl_mrs.models as models

all_models = ['lorentzian', 'voigt', 'free_shift']
modules = [models.lorentzian, models.voigt, models.freeshift]


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

# TO DO test getFittedModel
