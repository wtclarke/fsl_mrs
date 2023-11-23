# dyn_results.py - Class for collating dynMRS results
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2021 University of Oxford
import copy
import warnings
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from fsl_mrs.utils.misc import calculate_lap_cov, gradient
from fsl_mrs.utils.plotting import plot_general_corr


class ResultLoadError(Exception):
    pass


# Loading function
def load_dyn_result(load_dir, dyn_obj=None):
    """Load a saved dynamic fitting result from directory.

    The directory should cointain two csv files (dyn_results and init_results).
    And either the user must pass the asociated dynMRS object as the dyn_obj
    argument, or the directory must also contain a dyn.pkl file.s

    :param load_dir: Directory to load. Creaed using dynMRS.save method
    :type load_dir: str or pathlib.Path
    :param dyn_obj: Associated dynMRS object or if None will attempt to load a
     nested dynmrs_obj directory, defaults to None
    :type dyn_obj: fsl_mrs.dynamic.dynMRS, optional
    :return: Dynamic results object
    :rtype: dynRes_newton or dynRes_mcmc
    """
    if not isinstance(load_dir, Path):
        load_dir = Path(load_dir)

    sample_df = pd.read_csv(load_dir / 'dyn_results.csv', index_col='samples')
    init_df = pd.read_csv(load_dir / 'init_results.csv', index_col=0)

    if sample_df.shape[0] == 1:
        cls = dynRes_newton
    else:
        cls = dynRes_mcmc

    if dyn_obj:
        return cls(sample_df, dyn_obj, init_df)
    elif (load_dir / 'dynmrs_obj').is_dir():
        from .dynmrs import dynMRS
        dyn_obj = dynMRS.load(load_dir / 'dynmrs_obj')
        return cls(sample_df, dyn_obj, init_df)
    else:
        raise ResultLoadError('Dynamic object required. Pass directly or ensure dyn.pkl is availible')


# Plotting functions:
def subplot_shape(plots_needed):
    """Calculates the number and shape of subplots needed for
    given number of plots.

    :param plots_needed: Number of plots needed
    :type plots_needed: int
    :return: Number of columns and number of rows
    :rtype: tuple
    """
    col = int(np.floor(np.sqrt(plots_needed)))
    row = int(np.floor(np.sqrt(plots_needed)))
    counter = 0
    while col * row < plots_needed:
        if counter % 2:
            col += 1
        else:
            row += 1
        counter += 1
    return col, row


class dynRes:
    """Base dynamic results class. Not intended to be created directly, but via child classes
    dynRes_newton and dynRes_mcmc."""

    def __init__(self, samples, dyn, init):
        """Initilisation of dynRes class object.

        Typically called from __init__ of dynRes_newton and dynRes_mcmc

        :param samples: Array of free parameters returned by optimiser, can be 2D in mcmc case.
        :type samples: numpy.ndarray
        :type init: pd.DataFrame
        :param dyn: Copy of dynMRS class object.
        :type dyn: fsl_mrs.dynamic.dynMRS
        :param init: Results of the initilisation optimisation, containing 'resList' and 'x'.
        :type init: dict
        :type init: pd.DataFrame
        """
        if isinstance(samples, pd.DataFrame):
            self._data = samples
        else:
            self._data = pd.DataFrame(data=samples, columns=dyn.free_names)
            self._data.index.name = 'samples'

        self._dyn = copy.deepcopy(dyn)

        # Store the init as dataframe
        if isinstance(init, pd.DataFrame):
            self._init_x = init
        else:
            self._init_x = pd.DataFrame(init['x'], columns=self._dyn.mapped_names)

    def save(self, save_dir, save_dyn_obj=False):
        """Save the results to a directory

        Saves the two dataframes to csv format. If save_dyn_obj=True then the ._dyn object
        is also saved.

        :param save_dir: Location to save to, created if neccesary.
        :type save_dir: str or pathlib.Path
        :param save_dyn_obj: Save dynMRS object nested directory, defaults to False
        :type save_dyn_obj: bool, optional
        """
        if not isinstance(save_dir, Path):
            save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)

        # Save the two dataframes that contain the optimisation results
        # Everything else can be derived from these
        self._data.to_csv(save_dir / 'dyn_results.csv')
        self._init_x.to_csv(save_dir / 'init_results.csv')
        # Also save the free parameter covariances, which cannot be reconstructed without the data load.
        # But these are needed for peak combinations in 2nd level group analysis
        self.cov_free.to_csv(save_dir / 'dyn_cov.csv')

        # Save summaries of results
        # 1. mean + std of free parameters
        pd.concat((self.mean_free, self.std_free), axis=1, keys=['mean', 'sd'])\
            .to_csv(save_dir / 'free_parameters.csv')
        # 2. mean + std of mapped parameters
        pd.concat((self.dataframe_mapped, self.std_mapped), keys=['mean', 'std'], axis=0)\
            .T\
            .reorder_levels([1, 0], axis=1)\
            .sort_index(axis=1, level=0)\
            .to_csv(save_dir / 'mapped_parameters.csv')

        # Save model information
        with open(save_dir / 'model_information.json', 'w') as fp:
            json.dump(self.model_parameters, fp)

        # If selected save the dynamic object to a nested directory
        if save_dyn_obj:
            self._dyn.save(save_dir / 'dynmrs_obj', save_mrs_list=True)

    '''Properties to access the results viewed as the free parameter set.
       These are the free parameters that have been directly optimised.

       obj.dataframe_free returns a pandas dataframe with all mcmc samples
       obj.mean_free returns a pandas series contining the mean free parameters
       obj.x returns a numpy array of the mean free parameter estimates
       obj.std_free returns a pandas series contining the std of the free parameters
       '''
    @property
    def dataframe_free(self):
        """Return the pandas dataframe view of the (free parameter) results, includinge all mcmc samples."""
        return self._data

    @property
    def mean_free(self):
        """Return the (mcmc: mean) free parameters as a pandas Series."""
        return self.dataframe_free.mean()

    @property
    def x(self):
        """Return the (mcmc: mean) free parameters as a numpy array."""
        return self.dataframe_free.mean().to_numpy()

    # Methods implemented in child classes
    @property
    def std_free(self):
        """Implemented in child class

        Returns the standard deviations of the free parameters as a pandas Series
        """
        pass

    @property
    def cov_free(self):
        """Implemented in child class

        Returns the covariance matrix of free parameters
        """
        pass

    @property
    def corr_free(self):
        """Implemented in child class

        Returns the correlation matrix of free parameters
        """
        pass

    # Utility methods
    @property
    def free_names(self):
        """Free names from stored dynamic object"""
        return self._dyn.free_names

    '''Properties to access the results viewed as the mapped parameter set.
       These are the parameters that describe each individual spectrum in the dataset.

       obj.mapped_parameters_array returns a numpy array of all samples of mapped parameters
       obj.dataframe_mapped returns a pandas dataframe contining the mean free parameters
       obj.std_mapped returns a pandas series contining the std of the mapped parameters
       '''

    @property
    def mapped_parameters_array(self):
        """All mapped parameters. Shape is samples x timepoints x parameters.
        Number of samples will be 1 for newton, and >1 for MCMC.

        :return: array of mapped samples
        :rtype: np.array
        """
        mapped_samples = []
        for fp in self._data.to_numpy():
            mapped_samples.append(self._dyn.vm.free_to_mapped(fp))
        return np.asarray(mapped_samples)

    def _time_index_1d(self):
        """Utility function to return the best 1D time index"""
        if self._dyn.time_var.ndim > 1:
            return self._dyn.time_index
        else:
            return self._dyn.time_var

    @property
    def dataframe_mapped(self):
        """Mapped parameters arising from dynamic fit.

        :return: Pandas dataframe containing the mean mapped parameters for each time point
        :rtype: pandas.DataFrame
        """

        return pd.DataFrame(
            self.mapped_parameters_array.mean(axis=0),
            columns=self.mapped_names,
            index=self._time_index_1d())

    # Methods implemented in child classes
    @property
    def std_mapped(self):
        """Implemented in child class

        Returns the standard deviations of the mapped parameters
        """
        pass

    # Utility methods
    @property
    def mapped_names(self):
        """Mapped names from stored dynamic object"""
        return self._dyn.mapped_names

    '''Views of the initilisation parameters.
    Includes methods to vieww as both mapped and free parameter sets'''

    @property
    def init_mapped_params(self):
        """Mapped parameters from initilisation as dataframe

        :return: Pandas dataframe containing the mapped parameters for each time point
        :rtype: pandas.DataFrame
        """
        return self._init_x

    @property
    def init_mapped_parameters_array(self):
        """Mapped parameters from initilisation
        Shape is timepoints x parameters.

        :return: Array of mapped parameters from initilisation
        :rtype: np.array
        """
        return self._init_x.to_numpy()

    @property
    def init_free_parameters(self):
        """Free parameters calculated from the inversion of the dynamic model using the initilisation as input.

        :return: Free parameters estimated from initilisation
        :rtype: np.array
        """
        return self._dyn.vm.mapped_to_free(self.init_mapped_parameters_array)

    @property
    def init_free_dataframe(self):
        """Free parameters calculated from the inversion of the dynamic model using the initilisation as input.

        :return: Free parameters estimated from initilisation
        :rtype: pandas.Series
        """
        return pd.Series(data=self.init_free_parameters, index=self._dyn.free_names)

    @property
    def init_mapped_parameters_fitted_array(self):
        """Mapped parameters resulting from inversion of model using initilised parameters.
        Shape is timepoints x parameters.

        :return: Mapped parameters
        :rtype: np.array
        """
        return self._dyn.vm.free_to_mapped(self.init_free_parameters)

    @property
    def init_mapped_params_fitted(self):
        """Mapped parameters arising from fitting the initilisation parameters to the model.

        :return: Pandas dataframe containing the mean mapped parameters for each time point
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame(self.init_mapped_parameters_fitted_array, columns=self.mapped_names)

    @property
    def model_parameters(self):
        """Model performance parameters e.g. log-likelihood

        :return: Dictionary containing model performance parameters
        :rtype: dict
        """

        n_obs = self._dyn.data[0].shape[0] * self._dyn.vm.ntimes
        n_params = len(self.free_names)
        ll = self._dyn.dyn_loglik(self.x)
        aic = 2 * n_params + 2 * ll
        bic = np.log(n_obs) * n_params + 2 * ll

        return {
            'dynamic log-likelihood': ll,
            'init log-likelihood': self._dyn.dyn_loglik(self.init_free_parameters),
            'number of parameters': n_obs,
            'number of observations': n_params,
            'AIC': aic,
            'BIC': bic}

    '''Methods for collecting results for presentation.'''

    def collected_results(self, to_file=None):
        """Collect the results of dynamic MRS fitting

        Each mapped parameter category gets its own dataframe

        :param to_file: Output path, defaults to None
        :type to_file: str, optional
        :return: Formatted results
        :rtype: dict of pandas Dataframes
        """

        vm      = self._dyn.vm   # variable mapping object
        results = {cat: [] for cat in vm.free_category}  # collect results here
        values = self.x

        # Loop over free parameters
        # Store the values and names of these params in dict
        for val, name, ftype, category, mg in \
                zip(values, vm.free_names, vm.free_types, vm.free_category, vm.free_met_or_group):
            if category == 'conc':
                curr_dict = {'metabolite': mg}
                if ftype == 'fixed':
                    parameter_name = category
                else:
                    parameter_name = name.replace(f'{category}_{mg}_', '')
                curr_dict[parameter_name] = val
                results[category].append(curr_dict)
            elif category in ['eps', 'sigma', 'gamma']:
                curr_dict = {'group': mg}
                if ftype == 'fixed':
                    parameter_name = category
                else:
                    parameter_name = name.replace(f'{category}_{mg}_', '')
                curr_dict[parameter_name] = val
                results[category].append(curr_dict)
            else:
                curr_dict = {}
                if ftype == 'fixed':
                    parameter_name = category
                else:
                    parameter_name = name.replace(f'{category}_{mg}_', '')
                curr_dict[parameter_name] = val
                results[category].append(curr_dict)

        df_dict = {}
        for category in vm.free_category:
            if category == 'conc':
                df = pd.DataFrame(results[category]).set_index('metabolite').sort_index()
                df = df.groupby(df.index).sum(min_count=1)
            elif category in ['eps', 'sigma', 'gamma']:
                df = pd.DataFrame(results[category]).set_index('group').sort_index()
                df = df.groupby(df.index).sum(min_count=1)
            elif category == 'baseline':
                df = pd.DataFrame(results[category])
                border = self._dyn.vm._mapped_sizes[self._dyn.vm.mapped_categories.index('baseline')]
                order = list(range(int(border / 2)))
                nparams = int(df.shape[0] / (2 * border / 2))
                projection = ['real', ] * nparams + ['imag', ] * nparams
                new_index = pd.MultiIndex.from_product(
                    [order, projection],
                    names=['order', 're/im'])
                df.index = new_index
                df = df.groupby(level=[0, 1]).sum(min_count=1)
            else:
                df = pd.DataFrame(pd.DataFrame(results[category]).sum(), columns=[category]).T

            df_dict[category] = df

        # Optionally save out data
        if to_file is not None:
            for category in df_dict:
                df_dict[category]\
                    .to_csv(f'{to_file}_{category}.csv')

        return df_dict

    # Plotting
    def plot_mapped(self, tvals=None, fit_to_init=False, ground_truth=None):
        """Plot each mapped parameter across time points

        :param tvals: 'time' values on x axis, defaults to None / those stored in results object
        :type tvals: list, optional
        :param fit_to_init: Plot the mapped parameters as per initilisation, defaults to False
        :type fit_to_init: bool, optional
        :param ground_truth: If a ground truth exists (from simulation) plot the mapped parameters
            as calculated from this vector, defaults to None
        :type ground_truth: numpy.array, optional
        :return: Figure object
        :rtype: matplotlib.pyplot.figure.Figure
        """

        init_params = self.init_mapped_parameters_array
        fitted_init_params = self.init_mapped_parameters_fitted_array
        dyn_params = self.mapped_parameters_array.mean(axis=0)
        dyn_params_sd = self.mapped_parameters_array.std(axis=0)
        names = self.mapped_names
        if tvals is None:
            tvals = self._time_index_1d()
        if ground_truth is not None:
            gtval = self._dyn.vm.free_to_mapped(ground_truth)
        else:
            gtval = np.empty(len(names), dtype=object)

        # Plot the lot
        row, col = subplot_shape(len(names))

        fig, axes = plt.subplots(row, col, figsize=(20, 20))
        for ax, p_init, p_fit_init, p_dyn, p_dyn_sd, gt, paramname \
                in zip(
                    axes.flatten(),
                    init_params.T,
                    fitted_init_params.T,
                    dyn_params.T,
                    dyn_params_sd.T,
                    gtval.T,
                    names):
            ax.plot(tvals, p_init, 'o', label='init')
            ax.errorbar(tvals, p_dyn, yerr=p_dyn_sd, fmt='-', label='dyn')
            if fit_to_init:
                ax.plot(tvals, p_fit_init, ':', label='fit to init')
            if gt is not None:
                ax.plot(tvals, gt, 'k--', label='Ground Truth')
            ax.set_title(paramname)
            handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='right')
        return fig

    def _calc_fit_from_flatmapped(self, mapped):
        """Return the fitted spectra as an array

        :param mapped: Fitted mapped parameters
        :type mapped: np.ndarray
        :return: Array of fits as spectra
        :rtype: np.ndarray
        """
        fwd = []
        for idx, mp in enumerate(mapped):
            fwd.append(self._dyn.forward[idx](mp))
        return np.asarray(fwd)

    def _sensible_tval_strings(self, override=None):
        """Helper function to generate sensible title strings for the
        dynamic/time dimension.

        :param override: Provide your own list, defaults to None
        :type override: list, optional
        :return: List of strings
        :rtype: List
        """
        if override is not None:
            return [f'#{idx}: {t}' for idx, t in enumerate(override)]
        elif isinstance(self._dyn.time_var, dict):
            return [f'#{idx}' for idx in range(self._dyn._t_steps)]
        elif isinstance(self._dyn.time_var[0], (list, np.ndarray)):
            return [f'#{idx}' for idx in range(self._dyn._t_steps)]
        else:
            return [f'#{idx}: {t}' for idx, t in enumerate(self._dyn.time_var)]

    def plot_spectra(self, init=False, fit_to_init=False, indices=None, tvals=None):
        """Plot individual spectra as fitted using the dynamic model

        :param init: Plot the spectra as per initilisation, defaults to False
        :type init: bool, optional
        :param fit_to_init: Plot the spectra as per fitting the dynamic model to init, defaults to False
        :type fit_to_init: bool, optional
        :param indices: List of indicies to plot, defaults to None which plots up to 16 equally spaced.
        :type indices: list, optional
        :return: plotly figure
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        init_fit = self._calc_fit_from_flatmapped(self.init_mapped_parameters_array)
        init_fitted_fit = self._calc_fit_from_flatmapped(self.init_mapped_parameters_fitted_array)

        dyn_fit = []
        for mp in self.mapped_parameters_array:
            dyn_fit.append(self._calc_fit_from_flatmapped(mp))
        dyn_fit = np.asarray(dyn_fit)
        dyn_fit_mean = np.mean(dyn_fit, axis=0)
        dyn_fit_sd = np.std(dyn_fit.real, axis=0) + 1j * np.std(dyn_fit.imag, axis=0)

        x_axis = self._dyn.mrs_list[0].getAxes(ppmlim=self._dyn._fit_args['ppmlim'])

        colors = dict(data='rgb(67,67,67)',
                      init='rgb(59,59,253)',
                      init_fit='rgb(59,253,59)',
                      dyn='rgb(253,59,59)',
                      dyn_fill='rgba(253,59,59,0.2)')
        line_size = dict(data=1,
                         init=0.5,
                         init_fit=0.5,
                         dyn=1)

        sp_titles = self._sensible_tval_strings(override=tvals)
        n_transients = dyn_fit_mean.shape[0]
        if n_transients > 16 and indices is None:
            indices = np.round(np.linspace(0, n_transients - 1, 16)).astype(int)

        if indices is not None:
            init_fit = init_fit[indices, :]
            init_fitted_fit = init_fitted_fit[indices, :]
            dyn_fit_mean = dyn_fit_mean[indices, :]
            dyn_fit_sd = dyn_fit_sd[indices, :]
            sp_titles = np.asarray(sp_titles)[indices]

        row, col = subplot_shape(len(sp_titles))

        fig = make_subplots(rows=row, cols=col,
                            shared_xaxes=False, shared_yaxes=True,
                            x_title='Chemical shift (ppm)',
                            subplot_titles=sp_titles,
                            horizontal_spacing=0.02,
                            vertical_spacing=0.1)

        for idx in range(len(sp_titles)):
            coldx = int(idx % col)
            rowdx = int(np.floor(idx / col))

            # Only show the first legend entry
            if idx > 0:
                showlgnd = False
            else:
                showlgnd = True

            trace1 = go.Scatter(
                x=x_axis, y=self._dyn.data[idx].real,
                mode='lines',
                name='data',
                line=dict(color=colors['data'], width=line_size['data']),
                legendgroup='data',
                showlegend=showlgnd)
            fig.add_trace(trace1, row=rowdx + 1, col=coldx + 1)

            if init:
                trace2 = go.Scatter(
                    x=x_axis, y=init_fit[idx, :].real,
                    mode='lines',
                    name='init',
                    line=dict(color=colors['init'], width=line_size['init']),
                    legendgroup='init',
                    showlegend=showlgnd)
                fig.add_trace(trace2, row=rowdx + 1, col=coldx + 1)

            if fit_to_init:
                trace3 = go.Scatter(
                    x=x_axis, y=init_fitted_fit[idx, :].real,
                    mode='lines',
                    name='fit to init',
                    line=dict(color=colors['init_fit'], width=line_size['init_fit']),
                    legendgroup='fit_to_init',
                    showlegend=showlgnd)
                fig.add_trace(trace3, row=rowdx + 1, col=coldx + 1)

            trace4 = go.Scatter(
                x=x_axis, y=dyn_fit_mean[idx, :].real,
                mode='lines',
                name='dynamic',
                line=dict(color=colors['dyn'], width=line_size['dyn']),
                legendgroup='fit',
                showlegend=showlgnd)
            fig.add_trace(trace4, row=rowdx + 1, col=coldx + 1)

            if dyn_fit.shape[0] > 1:
                x_area = np.concatenate((x_axis, x_axis[::-1]))
                y_area = np.concatenate((dyn_fit_mean[idx, :].real - 1.96 * dyn_fit_sd[idx, :].real,
                                        dyn_fit_mean[idx, ::-1].real + 1.96 * dyn_fit_sd[idx, ::-1].real))
                trace5 = go.Scatter(x=x_area, y=y_area,
                                    mode='lines',
                                    name=f'95% CI dynamic (t={idx})',
                                    fill='toself',
                                    fillcolor=colors['dyn_fill'],
                                    line=dict(color='rgba(255,255,255,0)'),
                                    hoverinfo="skip",)
                fig.add_trace(trace5, row=rowdx + 1, col=coldx + 1)

        fig.update_xaxes(range=[self._dyn._fit_args['ppmlim'][1], self._dyn._fit_args['ppmlim'][0]],
                         dtick=.5,)
        fig.update_yaxes(zeroline=True,
                         zerolinewidth=1,
                         zerolinecolor='Gray',
                         showgrid=False, showticklabels=False)

        fig.update_layout(template='plotly_white', margin={'t': 30, 'l': 40, 'b': 60})
        # fig.layout.update({'height': 800, 'width': 1000})
        return fig

    def plot_corr(self):
        """Plot free parameter correlations using plotly
        """
        return plot_general_corr(
            self.corr_free.to_numpy(),
            self.corr_free.columns,
            title='Free Parameter Correlations')

    def plot_residuals(self):
        """Generate a 2D plot of residuals plu marginals.

        :return: Matplotlib figure object
        :rtype: matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        dyn_fit = []
        for mp in self.mapped_parameters_array:
            dyn_fit.append(self._calc_fit_from_flatmapped(mp))
        dyn_fit = np.asarray(dyn_fit).mean(axis=0)

        dyn_fit.shape

        residuals = np.asarray(self._dyn.data) - dyn_fit
        residuals /= dyn_fit.max()
        residuals *= 100
        ci95 = residuals.std() * 1.96
        x_axis = self._dyn.mrs_list[0].getAxes(ppmlim=self._dyn._fit_args['ppmlim'])
        yaxis = np.arange(residuals.shape[0])

        dyn_titles = self._sensible_tval_strings()

        xlim = [self._dyn._fit_args['ppmlim'][1], self._dyn._fit_args['ppmlim'][0]]

        fig = plt.figure(figsize=(12, 5))
        gs = gridspec.GridSpec(ncols=5, nrows=4, figure=fig, wspace=0, hspace=0)
        X, Y = np.meshgrid(x_axis, yaxis)
        ax1 = fig.add_subplot(gs[:3, :4])
        im = ax1.pcolormesh(X, Y, np.real(residuals), edgecolors=None, cmap='viridis', vmin=-ci95, vmax=ci95)
        ax1.set_xlim(xlim)
        ax1.set_xticks([])
        ax1.set_yticks([])

        ax2 = fig.add_subplot(gs[3, :4])
        mean_spec_residual = residuals.mean(axis=0).real
        ax2.plot(x_axis, mean_spec_residual, 'k')
        ax2.set_xlim(xlim)
        ax2.set_ylim(
            [1.1 * mean_spec_residual.min(),
             1.1 * mean_spec_residual.max()])
        ax2.set_yticks([-ci95, 0, ci95])
        ax2.set_xlabel('$\\delta$ (ppm)')
        # ax2.set_ylabel('%')

        ax3 = fig.add_subplot(gs[:3, 4])
        ax3.plot((np.abs(residuals)**2).mean(axis=1)**0.5, yaxis, 'k')
        ax3.yaxis.tick_right()
        if len(yaxis) > 15:
            tick_idx = np.round(np.linspace(0, len(yaxis) - 1, 15)).astype(int)
            ax3.set_yticks(
                yaxis[tick_idx],
                labels=np.asarray(dyn_titles)[tick_idx])
        else:
            ax3.set_yticks(yaxis, labels=dyn_titles)
        ax3.set_xlabel('RMSE (%)')
        cax = fig.add_axes(rect=[0.76, 0.11, 0.13, 0.1])
        fig.colorbar(im, orientation='horizontal', cax=cax, label='% max signal')
        cax.set_xticks([-ci95, 0, ci95])
        return fig


class dynRes_mcmc(dynRes):
    # TODO: Separate cov for dyn params and mapped params (like the Newton method)
    """Results class for MCMC optimised dynamic fitting.

    Derived from parent dynRes class.
    """
    def __init__(self, samples, dyn, init):
        """Initilise MCMC dynamic fitting results object.

        Simply calls parent class init.

        :param samples: Array of free parameters returned by optimiser, can be 2D in mcmc case.
        :type samples: numpy.ndarray
        :param dyn: Copy of dynMRS class object.
        :type dyn: fsl_mrs.dynamic.dynMRS
        :param init: Results of the initilisation optimisation, containing 'resList' and 'x'.
        :type init: dict
        """
        super().__init__(samples, dyn, init)

    @property
    def reslist(self):
        """Generate a list of (independent) results objects.

        :return: List of FitRes objects.
        :rtype: list
        """
        return self._dyn.form_FitRes(self.dataframe_free.to_numpy(), 'MH')

    @property
    def cov_free(self):
        """Returns the covariance matrix of free parameters

        :return: Covariance matrix as a DataFrame
        :rtype: pandas.DataFrame
        """
        return self.dataframe_free.cov()

    @property
    def corr_free(self):
        """Returns the correlation matrix of free parameters

        :return: Covariance matrix as a DataFrame
        :rtype: pandas.DataFrame
        """
        return self.dataframe_free.corr()

    @property
    def std_free(self):
        """Returns the standard deviations of the free parameters

        :return: Std as data Series
        :rtype: pandas.Series
        """
        return self.dataframe_free.std()

    @property
    def std_mapped(self):
        """Returns the standard deviations of the mapped parameters

        :return: Std as data Series
        :rtype: pandas.Series
        """
        return pd.DataFrame(
            self.mapped_parameters_array.std(axis=0),
            columns=self.mapped_names,
            index=self._time_index_1d())


class dynRes_newton(dynRes):

    def __init__(self, samples, dyn, init):
        """Initilise TNC optimised dynamic fitting results object.

        Calculates the covariance, correlation and standard deviations using the Fisher information matrix.

        :param samples: Array of free parameters returned by optimiser, can be 2D in mcmc case.
        :type samples: numpy.ndarray
        :param dyn: Copy of dynMRS class object.
        :type dyn: fsl_mrs.dynamic.dynMRS
        :param init: Results of the initilisation optimisation, containing 'resList' and 'x'.
        :type init: dict
        """
        if isinstance(samples, pd.DataFrame):
            super().__init__(samples, dyn, init)
        else:
            super().__init__(samples[np.newaxis, :], dyn, init)

        # Calculate covariance, correlation and uncertainty
        data = np.asarray(dyn.data).flatten()

        # Dynamic (free) parameters
        self._cov_free = calculate_lap_cov(self.x, dyn.full_fwd, data)
        crlb_dyn = np.diagonal(self._cov_free)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in sqrt')
            self._std_free = np.sqrt(crlb_dyn)
        self._corr_free = self._cov_free / (self._std_free[:, np.newaxis] * self._std_free[np.newaxis, :])

        # Mapped parameters
        # import pdb; pdb.set_trace()
        p = dyn.vm.free_to_mapped(self.x)
        self._mapped_params = dyn.vm.mapped_to_dict(p)
        # Mapped parameters covariance etc.
        grad_all = np.transpose(gradient(self.x, dyn.vm.free_to_mapped), (2, 0, 1))
        nt = dyn.vm.ntimes
        nf = len(self.x)
        std = np.zeros((dyn.vm.nmapped, nt))
        for idx in range(dyn.vm.nmapped):
            grad = np.reshape(np.array([grad_all[idx, ll, kk] for ll in range(nf) for kk in range(nt)]), (nf, nt)).T
            s_tmp = np.sqrt(np.diag(grad @ self._cov_free @ grad.T))
            std[idx, :] = np.array(s_tmp).T
        self._std_mapped = std

    @property
    def reslist(self):
        """Generate a list of (independent) results objects.

        :return: List of FitRes objects.
        :rtype: list
        """
        return self._dyn.form_FitRes(self.x, 'Newton')

    @property
    def cov_free(self):
        """Returns the covariance matrix of free parameters

        :return: Covariance matrix as a DataFrame
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame(self._cov_free, self.free_names, self.free_names)

    @property
    def corr_free(self):
        """Returns the correlation matrix of free parameters

        :return: Covariance matrix as a DataFrame
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame(self._corr_free, self.free_names, self.free_names)

    @property
    def std_free(self):
        """Returns the standard deviations of the free parameters

        :return: Std as data Series
        :rtype: pandas.Series
        """
        return pd.Series(self._std_free, self.free_names)

    @property
    def std_mapped(self):
        """Returns the standard deviations of the mapped parameters

        :return: Std as data Series
        :rtype: pandas.Series
        """
        return pd.DataFrame(self._std_mapped.T, columns=self.mapped_names, index=self._time_index_1d())
