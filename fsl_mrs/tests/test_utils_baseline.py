'''FSL-MRS test script

Test baseline tools

Copyright Will Clarke, University of Oxford, 2021'''

from fsl_mrs.utils import baseline
from fsl_mrs.core import MRS
import numpy as np
from pytest import raises


# Test the spline creation functions
def test_spline_regressor_creation():

    # _spline_basis
    basis = baseline._spline_basis(
        100,
        10,
        degree=3
    )
    assert basis.shape == (100, 10 + 3)

    # prepare_pspline_regressor
    fullbasis = baseline.prepare_pspline_regressor(
        100,
        (0, 4),
        (0, 80),
        bases_per_ppm=2)

    assert fullbasis.shape == (100, 22)
    assert np.allclose(np.real(fullbasis[:, :11]), np.imag(fullbasis[:, 11:]))
    assert np.sum(fullbasis[80:, :]) == 0


def test_diff_mat():
    diff = baseline._pspline_diff(10)

    assert np.allclose(
        diff,
        np.diff(
            np.eye(10),
            n=2)
    )


def test_ed_lambda_functions():

    basis = baseline._spline_basis(
        100,
        10,
        degree=3
    )

    # Large penalty
    edlow = baseline._ed_from_lambda(
        basis,
        1E10
    )
    assert np.isclose(edlow, 2)

    # small penalty
    edhigh = baseline._ed_from_lambda(
        basis,
        1E-15
    )
    assert np.isclose(edhigh, basis.shape[1])

    # Test reverse
    lam_loose = baseline.lambda_from_ed(
        basis.shape[1] - 1,
        basis,
    )
    assert np.isclose(baseline._ed_from_lambda(
        basis,
        lam_loose),
        basis.shape[1] - 1,
        atol=1E-2)

    lam_tight = baseline.lambda_from_ed(
        2.5,
        basis,
    )
    assert np.isclose(baseline._ed_from_lambda(
        basis,
        lam_tight),
        2.5,
        atol=1E-2)


# prepare_penalised_functions
def test_prepare_penalised_functions():
    basis = baseline.prepare_pspline_regressor(
        100,
        (0, 5),
        (0, 100))

    # Check with no penalty applied
    moderr, modgrad = baseline.prepare_penalised_functions(
        basis.shape[1] / 2,
        lambda x: np.sum(x),
        lambda x: x,
        basis,
        lambda x: x,
    )

    input = np.arange(basis.shape[1]) ** 2

    assert isinstance(moderr(input), float)
    assert modgrad(input).size == basis.shape[1]

    assert np.isclose(moderr(input), np.sum(input), atol=1E-1)
    assert np.allclose(modgrad(input), input, atol=1E-1)

    # Check with maximum penalty applied
    moderr, modgrad = baseline.prepare_penalised_functions(
        2,
        lambda x: np.sum(x),
        lambda x: x,
        basis,
        lambda x: x,
    )

    assert moderr(input) > np.sum(input)


# Test the polynomial creation functions
def test_polynomial_regressor_creation():
    # Create dummy mrs
    mrs = MRS(FID=np.zeros((1000,)), cf=100, bw=2000, nucleus='1H')
    baseline_order = 3
    ppmlim = (-1, 1)
    B = baseline.prepare_polynomial_regressor(
        mrs.numPoints,
        baseline_order,
        mrs.ppmlim_to_range(ppmlim))

    assert B.shape == (mrs.numPoints, 2 * (baseline_order + 1))
    assert np.sum(B[:, 0]) == 100
    assert np.sum(B[:, 1]) == 100j


# Test the Baseline class
def test_baseline_class_init():
    mrs = MRS(FID=np.zeros((1000,)), cf=100, bw=2000, nucleus='1H')
    obj = baseline.Baseline(
        mrs,
        (0, 5),
        "poly, 1",
        None)

    assert obj._spectral_points == mrs.numPoints
    assert obj._ppm_range == mrs.ppmlim_to_range((0, 5))
    assert obj._ppm_limits == (0, 5)

    obj = baseline.Baseline(
        mrs,
        None,
        "poly, 1",
        None)

    assert obj._ppm_range == (0, mrs.numPoints)
    assert obj._ppm_limits is None


def test_baseline_class_mode():

    # Create dummy mrs
    mrs = MRS(FID=np.zeros((1000,)), cf=100, bw=2000, nucleus='1H')

    with raises(
            baseline.BaselineError,
            match="The input to the 'baseline' option must be in the format:"):
        _ = baseline.Baseline(
            mrs,
            (0, 5),
            "total rubbish",
            None)

    with raises(
            baseline.BaselineError,
            match="With 'Spline, ', either specify an ED number"):
        _ = baseline.Baseline(
            mrs,
            (0, 5),
            "spline, rubbish",
            None)

    with raises(
            baseline.BaselineError,
            match='Minimum spline ED is 2'):
        _ = baseline.Baseline(
            mrs,
            (0, 5),
            "spline, 1",
            None)

    obj = baseline.Baseline(
        mrs,
        (0, 5),
        "spline, 3",
        None)
    assert obj.mode == "spline"
    assert obj.spline_penalty == 3

    obj = baseline.Baseline(
        mrs,
        (0, 5),
        "spline, 10.6",
        None)
    assert obj.mode == "spline"
    assert obj.spline_penalty == 10.6

    obj = baseline.Baseline(
        mrs,
        (0, 5),
        "spline, stiff",
        None)
    assert obj.mode == "spline"
    assert obj.spline_penalty == 10

    with raises(
            baseline.BaselineError,
            match="polynomial_order is only defined for mode='polynomial'"):
        obj.polynomial_order

    # Polynomial
    obj = baseline.Baseline(
        mrs,
        (0, 5),
        "poly, 1",
        None)
    assert obj.mode == "polynomial"
    assert obj.polynomial_order == 1

    with raises(
            baseline.BaselineError,
            match="spline_penalty is only defined for mode='spline'"):
        obj.spline_penalty

    with raises(
            baseline.BaselineError,
            match="Specify an integer number with 'poly'/'polynomial"):
        obj = baseline.Baseline(
            mrs,
            (0, 5),
            "polynomial, 10.2",
            None)

    with raises(
            baseline.BaselineError,
            match="Specify an integer number with 'poly'/'polynomial"):
        obj = baseline.Baseline(
            mrs,
            (0, 5),
            "polynomial, notanumber",
            None)

    obj = baseline.Baseline(
        mrs,
        (0, 5),
        "off",
        None)
    assert obj.mode == "off"
    assert obj.disabled

    # Legacy
    obj = baseline.Baseline(
        mrs,
        (0, 5),
        None,
        -1)
    assert obj.mode == "off"
    assert obj.disabled

    obj = baseline.Baseline(
        mrs,
        (0, 5),
        None,
        0)
    assert obj.mode == "polynomial"
    assert obj.polynomial_order == 0


# Test the regressor fetch
def test_baseline_regressor():
    mrs = MRS(FID=np.zeros((1000,)), cf=100, bw=2000, nucleus='1H')
    obj = baseline.Baseline(
        mrs,
        (0, 5),
        "poly, 1",
        None)

    assert obj.regressor.shape == (mrs.numPoints, 4)

    obj = baseline.Baseline(
        mrs,
        (0, 5),
        "off",
        None)

    assert obj.regressor.shape == (mrs.numPoints, 2)

    obj = baseline.Baseline(
        mrs,
        (0, 5),
        "spline, 10",
        None)

    assert obj.regressor.shape == (mrs.numPoints, 2 * (5 * 15 + 3))


# Test the function modification
def test_baseline_prepare_penalised_fit_functions():
    mrs = MRS(FID=np.zeros((1000,)), cf=100, bw=2000, nucleus='1H')
    obj = baseline.Baseline(
        mrs,
        (0, 5),
        "spline, 78",
        None)

    moderr, modgrad = obj.prepare_penalised_fit_functions(
        lambda x: np.sum(x),
        lambda x: x,
        lambda x: x,
    )

    input = np.arange(obj.n_basis * 2) ** 2

    assert isinstance(moderr(input), float)
    assert modgrad(input).size == obj.n_basis * 2

    assert np.isclose(moderr(input), np.sum(input), atol=1E-1)
    assert np.allclose(modgrad(input), input, atol=1E-1)
