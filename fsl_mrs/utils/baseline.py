#!/usr/bin/env python

# baseline.py - Functions associated with the baseline description
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

import typing
import re

import numpy as np
from scipy.special import gamma
from scipy.optimize import fminbound

if typing.TYPE_CHECKING:
    from fsl_mrs.core.mrs import MRS
from fsl_mrs.utils.misc import regress_out


spline_baseline_specifiers = {
    'very-stiff': 3,
    'stiff': 10,
    'moderate': 20,
    'flexible': 30,
    'very-flexible': 50
}
spline_baseline_specifier_names = "'very-stiff', 'stiff', 'moderate', 'flexible', and 'very-flexible'"


class BaselineError(Exception):
    pass


class Baseline:
    def __init__(
            self,
            mrs: 'MRS',
            ppmlim: tuple,
            baseline_selection: str,
            baseline_order: int | None) -> None:

        # Store information needed from mrs object
        self._spectral_points = mrs.numPoints
        self._ppm_limits = ppmlim
        self._ppm_range  = mrs.ppmlim_to_range(ppmlim)

        # Defaults
        self._spline_description = None

        # Handle legacy option
        if baseline_order is not None:
            # Handle the legacy way of disabling
            if baseline_order < 0:
                self.mode = 'off'
            # Otherwise set to polynomial
            else:
                self.mode = 'polynomial'
                self._order = baseline_order
            return

        num_str = r'(:?[\d.e\+\-]+)'
        inputre = re.compile(
            rf'((?:poly)|(?:spline)|(?:off))(?:nomial)?,?\s?({num_str}|[\w\-]+)?',
            re.IGNORECASE)

        match = inputre.match(baseline_selection)
        if match is None:
            raise BaselineError(
                "The input to the 'baseline' option must be in the format:\n",
                "- 'poly, N', where N is the polynomial baseline order,"
                " e.g. 'poly, 2' for a quadratic baseline,\n"
                "- 'spline, description', where description is one of the specifiers listed below\n,"
                "- 'spline, ED', where ED is the effective dimension (2, to inf) where 2 is most rigid\n,"
                "- 'off', which disables the baseline.\n"
                f"The spline specifiers are {spline_baseline_specifier_names}")

        if match[1] == 'poly':
            try:
                self._order = int(match[2])
            except ValueError:
                raise BaselineError("Specify an integer number with 'poly'/'polynomial")

            if self._order < 0:
                print('Polynomial baseline order set less than 1, disabling.')
                self.mode = 'off'
            else:
                self.mode = 'polynomial'

        elif match[1] == 'spline':
            self.mode = 'spline'
            try:
                ed = float(match[2])
                self._penalty = ed
                if ed < 2:
                    raise BaselineError("Minimum spline ED is 2")
            except ValueError:
                if match[2] in spline_baseline_specifiers:
                    self._penalty = spline_baseline_specifiers[match[2]]
                    self._spline_description = match[2]
                else:
                    raise BaselineError(
                        "With 'Spline, ', either specify an ED number "
                        f"or one of the descriptors: {spline_baseline_specifier_names}.")

        elif match[1] == 'off':
            self.mode = 'off'

    @property
    def disabled(self):
        return self.mode == 'off'

    @property
    def spline_penalty(self):
        if self.mode == 'spline':
            return self._penalty
        else:
            raise BaselineError(
                f"spline_penalty is only defined for mode='spline', mode is {self.mode}.")

    @property
    def polynomial_order(self):
        if self.mode == 'polynomial':
            return self._order
        else:
            raise BaselineError(
                f"polynomial_order is only defined for mode='polynomial', mode is {self.mode}.")

    @property
    def regressor(self):
        # Prepare baseline regressor
        if self.mode == 'polynomial':
            return prepare_polynomial_regressor(self._spectral_points, self._order, self._ppm_range)
        elif self.mode == 'spline':
            return prepare_pspline_regressor(self._spectral_points, self._ppm_limits, self._ppm_range)
        elif self.mode == 'off':
            return prepare_polynomial_regressor(self._spectral_points, 0, self._ppm_range)

    @property
    def n_basis(self) -> int:
        """Returns the number of regressors (for each of real/imag)

        :return: Number of baselien regressors per real/imag channel
        :rtype: int
        """
        return int(self.regressor.shape[1] / 2)

    def prepare_penalised_fit_functions(
            self,
            err_function: typing.Callable,
            grad_function: typing.Callable,
            x2b: typing.Callable):
        """_summary_

        _extended_summary_

        :param err_function: _description_
        :type err_function: typing.Callable
        :param grad_function: _description_
        :type grad_function: typing.Callable
        :param x2b: _description_
        :type x2b: typing.Callable
        :return: _description_
        :rtype: _type_
        """
        if self.mode in ("off", "polynomial"):
            return err_function, grad_function
        return prepare_penalised_functions(
            self.spline_penalty,
            err_function,
            grad_function,
            self.regressor,
            x2b)

    def cov_penalty_term(
            self,
            n_fit_params: int) -> np.ndarray:
        """Return any penalty term needed for the laplace covariance estimation.

        Output will be the size of J@J.T

        :param n_fit_params: Number of parameters in fitting model
        :type n_fit_params: int
        :return: penalty term matrix size (n_fit_params, n_fit_params)
        :rtype: np.ndarray
        """
        if self.mode in ("off", "polynomial"):
            return np.zeros((n_fit_params, n_fit_params))
        return calculate_lap_cov_penalty_term(
            self.spline_penalty,
            self.regressor,
            n_fit_params)

    def mh_penalty_term(
            self,
    ) -> typing.Callable:
        """Returns a function that will calculate the penalty from the current fit parameters

        penalty_function = bline_obj.mh_penalty_term()

        :return: Calculation function
        :rtype: typing.Callable
        """
        return calculate_mh_liklihood_term(
            self.spline_penalty,
            self.regressor)

    def __str__(self) -> str:
        if self.mode == "polynomial":
            return f"polynomial baseline of order {self._order}"
        elif self.mode == "spline":
            if self._spline_description:
                return f"p-spline (penalised spline) baseline with effective dimension {self._penalty} "\
                    f"({self._spline_description})"
            else:
                return f"p-spline (penalised spline) baseline with effective dimension {self._penalty}"
        elif self.mode == "off":
            return "baseline disabled"


# P-spline baseline functions
def _spline_basis(n_points: int, n_spline: int, degree: int = 3) -> np.ndarray:
    """Generate a spline basis matrix.

    Adapted from R code in "Splines, knots, and penalties", Eilers 2010.

    :param n_points: Number of frequency points in basis
    :type n_points: int
    :param n_spline: Number of bases
    :type n_spline: int
    :param degree: Spline order, defaults to 3 / cubic
    :type degree: int, optional
    :return: Basis matrix
    :rtype: np.ndarray
    """
    x = np.arange(1, n_points + 1)
    dx = (x.max() - x.min()) / n_spline
    knots = np.arange(
        x.min() - degree * dx,
        x.max() + (degree + 1) * dx,
        dx)

    def trunc_power(a, limit, power):
        return (a - limit) ** power * (a > limit)

    P = np.stack(
        [trunc_power(x, kt, degree) for kt in knots]).T

    D = np.diff(np.eye(knots.size), n=degree + 1, axis=0) / (gamma(degree + 1) * dx ** degree)
    B = ((-1)**(degree + 1)) * P @ D.T
    return B


def prepare_pspline_regressor(
        n_points: int,
        ppmlim: tuple,
        ppmlim_points: tuple,
        bases_per_ppm: int = 15) -> np.ndarray:
    """Creates the complex spline basis matrix.

    :param n_points: Number of frequency points
    :type n_points: int
    :param ppmlim: limits over which to construct basis (outside set to zero)
    :type ppmlim: tuple
    :param ppmlim_points: indices that correspond to the ppm limits
    :type ppmlim_points: tuple
    :param bases_per_ppm: Number of spine bases per ppm, defaults to 15
    :type bases_per_ppm: int, optional
    :return: Complex basis array
    :rtype: np.ndarray
    """

    first, last = ppmlim_points
    ppm_range = ppmlim[1] - ppmlim[0]
    single_basis = _spline_basis(
        last - first,
        int(bases_per_ppm * ppm_range))

    # Insert the single basis twice into the output
    # First is real, second is imaginary
    n_basis = single_basis.shape[1]
    full_complex_basis = np.zeros((n_points, n_basis * 2), complex)
    full_complex_basis[first:last, :n_basis] = single_basis
    full_complex_basis[first:last, n_basis:] = 1j * single_basis

    return full_complex_basis


def _pspline_diff(n_basis: int) -> np.ndarray:
    """Create the difference matrix for applying the p-spline penalties

    Calculates second order differences.

    :param n_basis: Number of bases in baseline regressor
    :type n_basis: int
    :return: difference matrix (n_basis x n_basis - 2)
    :rtype: np.ndarray
    """
    return np.diff(
        np.eye(n_basis),
        n=2)


def _ed_from_lambda(basis: np.ndarray, lam: float) -> float:
    """Calculate the effective dimension (ED) from lambda

    Hastie T, Tibshirani R. Generalized Additive Models. London, UK: Chapman and Hall; 1990.

    :param basis: real part of the baseline basis
    :type basis: np.ndarray
    :param lam: penalty scaling lambda
    :type lam: float
    :return: ED
    :rtype: float
    """
    diff = _pspline_diff(basis.shape[1])
    inv = np.linalg.lstsq(
        (basis.T @ basis) + lam * (diff @ diff.T),
        np.eye(basis.shape[1]))[0]
    H = inv @ (basis.T @ basis)
    return np.trace(np.real(H))


def lambda_from_ed(target_ed: float, basis: np.ndarray) -> float:
    """Calculate the penalty scaling from the effective dimension (ED)

    Hastie T, Tibshirani R. Generalized Additive Models. London, UK: Chapman and Hall; 1990.

    :param target_ed: Requested ED
    :type target_ed: float
    :param basis: real or imaginary submatrix of the baseline regressors
    :type basis: np.ndarray
    :return: estimated lambda
    :rtype: float
    """
    def loss_func(x):
        # breakpoint()
        return (_ed_from_lambda(basis, x) - target_ed)**2

    # Note I had to bring in the limits as otherwise got odd behaviour
    return fminbound(loss_func, 1E-7, 1E7)


def prepare_penalised_functions(
        penalty: float,
        err_func: typing.Callable,
        grad_func: typing.Callable,
        basis: np.ndarray,
        x2b: typing.Callable
) -> tuple[typing.Callable, typing.Callable]:
    """Generate the penalised minimisation functions from the original, un-penalised functions.

    :param penalty: penalty ED
    :type penalty: float
    :param err_func: unmodified error function
    :type err_func: typing.Callable
    :param grad_func: unmodified gradient function
    :type grad_func: typing.Callable
    :param basis: baseline basis
    :type basis: numpy.ndarray
    :param x2b: Function used to extract "b" the baseline betas from a list of all parameters "x"
    :type x2b: typing.Callable
    :return: Returns modified, penalised error and gradient functions
    :rtype: tuple[typing.Callable, typing.Callable]
    """

    n_basis = int(basis.shape[1] / 2)

    diff_mat = _pspline_diff(n_basis)

    penalty_lambda = lambda_from_ed(
        penalty,
        basis[:, :n_basis])

    def penalised_error(*args):
        b = x2b(args[0])
        return err_func(*args)\
            + penalty_lambda * np.linalg.norm(b[:(n_basis)] @ diff_mat)**2\
            + penalty_lambda * np.linalg.norm(b[(n_basis):] @ diff_mat)**2

    def penalised_grad(*args):
        additional = np.zeros_like(args[0])
        b = x2b(args[0])
        diff_term = 2 * penalty_lambda * (diff_mat @ diff_mat.T)
        additional[-2 * n_basis:-n_basis] = diff_term @ b[:n_basis]
        additional[-n_basis:] = diff_term @ b[n_basis:]
        out = grad_func(*args) + additional
        return out

    return penalised_error, penalised_grad


def calculate_mh_liklihood_term(
        penalty: float,
        basis: np.ndarray
) -> typing.Callable:
    """Create the function that calculates the MH likelihood penalty value

    :param penalty: penalty ED
    :type penalty: float
    :param basis: baseline basis
    :type basis: numpy.ndarray
    :return: calculation function
    :rtype: typing.Callable
    """

    n_basis = int(basis.shape[1] / 2)

    diff_mat = _pspline_diff(n_basis)

    penalty_lambda = lambda_from_ed(
        penalty,
        basis[:, :n_basis])

    def mh_penalty(p) -> float:
        """Returns the liklihood penalty term

        :param p: All fit parameters, last section is baseline
        :type p: np.ndarray
        :return: penalty
        :rtype: float
        """
        return penalty_lambda * np.linalg.norm(p[(-n_basis * 2):(-n_basis)] @ diff_mat)**2\
            + penalty_lambda * np.linalg.norm(p[-n_basis:] @ diff_mat)**2

    return mh_penalty


def calculate_lap_cov_penalty_term(
        penalty: float,
        basis: np.ndarray,
        n_params: int
) -> np.ndarray:
    """Generate the additional penalty term needed for the laplace covariance estimation

    2 * lambda * D.T @ D

    :param penalty: penalty ED
    :type penalty: float
    :param basis: baseline basis
    :type basis: numpy.ndarray
    :param n_params: Number of parameters in full fitting model
    :type n_params: int
    :return: Returns penalty matrix the same shape as J@J.T
    :rtype: np.ndarray
    """

    n_basis = int(basis.shape[1] / 2)

    diff_mat = _pspline_diff(n_basis)

    penalty_lambda = lambda_from_ed(
        penalty,
        basis[:, :n_basis])

    pterm = 2 * penalty_lambda * (diff_mat @ diff_mat.T)
    pterm_full = np.zeros((n_params, n_params))
    pterm_full[-pterm.shape[0]:, -pterm.shape[0]:] = pterm
    pterm_full[-2 * pterm.shape[0]:-pterm.shape[0], -2 * pterm.shape[0]:-pterm.shape[0]] = pterm
    return pterm_full


# Polynomial baseline functions
def prepare_polynomial_regressor(
        n_points: int,
        baseline_order: int,
        ppmlim_points: tuple | None) -> np.ndarray:
    """Prepare a set of polynomial baseline regressors

    Real regressors then imaginary of order `baseline_order`

    :param n_points: Number of points in the spectrum
    :type n_points: int
    :param baseline_order: The polynomial order of regressors
    :type baseline_order: int
    :param ppmlim_points: Indices of the polynomial range in the frequency direction
    :type ppmlim_points: tuple | None
    :return: Matrix of regressors.
    :rtype: np.ndarray
    """

    if ppmlim_points:
        first, last = ppmlim_points
    else:
        first = 0
        last = n_points

    B = []
    x = np.zeros(n_points, complex)
    x[first:last] = np.linspace(-1, 1, last - first)

    for i in range(baseline_order + 1):
        regressor  = x**i
        if i > 0:
            regressor  = regress_out(regressor, B, keep_mean=False)

        B.append(regressor.flatten())
        B.append(1j * regressor.flatten())
    B = np.asarray(B).T
    tmp = B.copy()
    B   = 0 * B
    B[first:last, :] = tmp[first:last, :].copy()

    return B
