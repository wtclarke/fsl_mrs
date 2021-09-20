# dyn_results.py - Class for collating dynMRS results
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2021 University of Oxford
import copy
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fsl_mrs.utils.misc import calculate_lap_cov


# Plotting functions:
def subplot_shape(plots_needed):
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

    def __init__(self, samples, dyn, init):
        self._data = pd.DataFrame(data=samples, columns=dyn.free_names)
        self._data.index.name = 'samples'

        self._dyn = copy.deepcopy(dyn)

        # Store the init mapped representation
        self._init_x = init['x']

    @property
    def data_frame(self):
        return self._data

    @property
    def x(self):
        return self.data_frame.mean().to_numpy()

    @staticmethod
    def flatten_mapped(mapped):
        """Flatten the nested array of mapped parameters created by the variable mapping class into
        a N timpoint x M parameter matrix.

        :param mapped: Nested array representation
        :type mapped: np.array
        :return: Flattened array
        :rtype: np.array
        """
        flattened = []
        for mp in mapped:
            flattened.append(np.hstack(mp))
        return np.asarray(flattened)

    @property
    def mapped_parameters(self):
        """Flattened mapped parameters. Shape is samples x timepoints x parameters.
        Number of samples will be 1 for newton, and >1 for MCMC.

        :return: array of mappes samples
        :rtype: np.array
        """
        mapped_samples = []
        for fp in self._data.to_numpy():
            mapped_samples.append(self.flatten_mapped(self._dyn.vm.free_to_mapped(fp)))
        return np.asarray(mapped_samples)

    @property
    def mapped_parameters_init(self):
        """Flattened mapped parameters from initilisation
        Shape is timepoints x parameters.

        :return: Flattened mapped parameters from initilisation
        :rtype: np.array
        """
        return self.flatten_mapped(self._init_x)

    @property
    def free_parameters_init(self):
        """Free parameters calculated from the inversion of the dynamic model using the initilisation as input.

        :return: Free parameters estimated from initilisation
        :rtype: np.array
        """
        return self._dyn.vm.mapped_to_free(self._init_x)

    @property
    def init_dataframe(self):
        """Free parameters calculated from the inversion of the dynamic model using the initilisation as input.

        :return: Free parameters estimated from initilisation
        :rtype: np.array
        """
        return pd.DataFrame(data=self.free_parameters_init, index=self._dyn.free_names).T

    @property
    def mapped_parameters_fitted_init(self):
        """Mapped parameters resulting from inversion of model using initilised parameters.
        Shape is timepoints x parameters.

        :return: Mapped parameters
        :rtype: np.array
        """
        return self.flatten_mapped(self._dyn.vm.free_to_mapped(self.free_parameters_init))

    @property
    def mapped_names(self):
        """Mapped names from stored dynamic object"""
        return self._dyn.mapped_names

    @property
    def free_names(self):
        """Free names from stored dynamic object"""
        return self._dyn.free_names

    def collected_results(self, to_file=None):
        """Collect the results of dynamic MRS fitting

        Each mapped parameter group gets its own dict

        :param dynres:  output of dynmrs.fit()
        :type dynres: dict
        :param to_file: Output path, defaults to None
        :type to_file: str, optional
        :return: Formatted results
        :rtype: dict of dict
        """

        vm      = self._dyn.vm   # variable mapping object
        results = {}             # collect results here
        values  = self.x   # get the optimisation results here
        counter = 0
        metab_names = self._dyn.metabolite_names

        # Loop over parameter groups (e.g. conc, Phi_0,  Phi_1, ... )
        # Each will have a number of dynamic params associated
        # Store the values and names of these params in dict
        for index, param in enumerate(vm.mapped_names):
            isconc = param == 'conc'  # special case for concentration params
            results[param] = {}
            nmapped = vm.mapped_sizes[index]  # how many mapped params?
            beh     = vm.Parameters[param]    # what kind of dynamic model?
            if (beh == 'fixed'):  # if fixed, just a single value per mapped param
                if not isconc:
                    results[param]['name'] = [f'{param}_{x}' for x in range(nmapped)]
                else:
                    results[param]['metab'] = metab_names
                results[param]['Values'] = values[counter:counter + nmapped]
                counter += nmapped
            elif (beh == 'variable'):
                if not isconc:
                    results[param]['name'] = [f'{param}_{x}' for x in range(nmapped)]
                else:
                    results[param]['metab'] = metab_names
                for t in range(vm.ntimes):
                    results[param][f't{t}'] = values[counter:counter + nmapped]
                    counter += nmapped
            else:
                if 'dynamic' in beh:
                    dyn_name = vm.Parameters[param]['params']
                    nfree    = len(dyn_name)
                    if not isconc:
                        results[param]['name'] = [f'{param}_{x}' for x in range(nmapped)]
                    else:
                        results[param]['metab'] = metab_names
                    tmp = []
                    for t in range(nmapped):
                        tmp.append(values[counter:counter + nfree])
                        counter += nfree
                    tmp = np.asarray(tmp)
                    for i, t in enumerate(dyn_name):
                        results[param][t] = tmp[:, i]

        if to_file is not None:
            import pandas as pd
            for param in vm.mapped_names:
                df = pd.DataFrame(results[param])
                df.to_csv(to_file + f'_{param}.csv', index=False)

        return results

    # Plotting
    def plot_mapped(self, tvals=None, fit_to_init=False):
        """Plot each mapped parameter across time points

        :param fit_to_init: Plot the mapped parameters as per initilisation, defaults to False
        :type fit_to_init: bool, optional
        """

        init_params = self.mapped_parameters_init
        fitted_init_params = self.mapped_parameters_fitted_init
        dyn_params = self.mapped_parameters.mean(axis=0)
        dyn_params_sd = self.mapped_parameters.std(axis=0)
        names = self.mapped_names
        if tvals is None:
            tvals = self._dyn.time_var

        # Plot the lot
        row, col = subplot_shape(len(names))

        fig, axes = plt.subplots(row, col, figsize=(20, 20))
        for ax, p_init, p_fit_init, p_dyn, p_dyn_sd, paramname \
                in zip(axes.flatten(), init_params.T, fitted_init_params.T, dyn_params.T, dyn_params_sd.T, names):
            ax.plot(tvals, p_init, 'o', label='init')
            if fit_to_init:
                ax.plot(tvals, p_fit_init, ':', label='fit to init')
            ax.errorbar(tvals, p_dyn, yerr=p_dyn_sd, fmt='-', label='dyn')
            ax.set_title(paramname)
            handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='right')
        plt.show()

    def plot_spectra(self, init=False, fit_to_init=False):
        """Plot individual spectra as fitted using the dynamic model

        :param init: Plot the spectra as per initilisation, defaults to False
        :type init: bool, optional
        :param fit_to_init: Plot the spectra as per fitting the dynamic model to init, defaults to False
        :type fit_to_init: bool, optional
        :param init:
        :return: plotly figure
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        def calc_fit_from_flatmapped(mapped):
            fwd = []
            for mp in mapped:
                fwd.append(self._dyn.forward(mp))
            return np.asarray(fwd)

        init_fit = calc_fit_from_flatmapped(self.mapped_parameters_init)
        init_fitted_fit = calc_fit_from_flatmapped(self.mapped_parameters_fitted_init)

        dyn_fit = []
        for mp in self.mapped_parameters:
            dyn_fit.append(calc_fit_from_flatmapped(mp))
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

        sp_titles = [f'#{idx}: {t}' for idx, t in enumerate(self._dyn.time_var)]

        row, col = subplot_shape(len(sp_titles))

        fig = make_subplots(rows=row, cols=col,
                            shared_xaxes=False, shared_yaxes=True,
                            x_title='Chemical shift (ppm)',
                            subplot_titles=sp_titles,
                            horizontal_spacing=0.05,
                            vertical_spacing=0.05)

        for idx in range(len(sp_titles)):
            coldx = int(idx % col)
            rowdx = int(np.floor(idx / col))

            trace1 = go.Scatter(x=x_axis, y=self._dyn.data[idx].real,
                                mode='lines',
                                name=f'data (t={idx})',
                                line=dict(color=colors['data'], width=line_size['data']))
            fig.add_trace(trace1, row=rowdx + 1, col=coldx + 1)

            if init:
                trace2 = go.Scatter(x=x_axis, y=init_fit[idx, :].real,
                                    mode='lines',
                                    name=f'init (t={idx})',
                                    line=dict(color=colors['init'], width=line_size['init']))
                fig.add_trace(trace2, row=rowdx + 1, col=coldx + 1)

            if fit_to_init:
                trace3 = go.Scatter(x=x_axis, y=init_fitted_fit[idx, :].real,
                                    mode='lines',
                                    name=f'fit to init (t={idx})',
                                    line=dict(color=colors['init_fit'], width=line_size['init_fit']))
                fig.add_trace(trace3, row=rowdx + 1, col=coldx + 1)

            trace4 = go.Scatter(x=x_axis, y=dyn_fit_mean[idx, :].real,
                                mode='lines',
                                name=f'dynamic (t={idx})',
                                line=dict(color=colors['dyn'], width=line_size['dyn']))
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

        fig.update_layout(template='plotly_white')
        # fig.layout.update({'height': 800, 'width': 1000})
        return fig


class dynRes_mcmc(dynRes):

    def __init__(self, samples, dyn, init):
        super().__init__(samples, dyn, init)

    @property
    def cov(self):
        return self.data_frame.cov()

    @property
    def corr(self):
        return self.data_frame.corr()

    @property
    def std(self):
        return self.data_frame.std()


class dynRes_newton(dynRes):

    def __init__(self, samples, dyn, init):
        super().__init__(samples[np.newaxis, :], dyn, init)

        # Calculate covariance, correlation and uncertainty
        data = np.asarray(dyn.data).flatten()
        # TODO: replace the below with something that uses the hessian from scipy.optimize
        self._cov = calculate_lap_cov(samples, dyn.full_fwd, data)
        crlb = np.diagonal(self._cov)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in sqrt')
            self._std = np.sqrt(crlb)
        self._corr = self._cov / (self._std[:, np.newaxis] * self._std[np.newaxis, :])

    @property
    def cov(self):
        return pd.DataFrame(self._cov, self.free_names, self.free_names)

    @property
    def corr(self):
        return pd.DataFrame(self._corr, self.free_names, self.free_names)

    @property
    def std(self):
        return pd.Series(self._std, self.free_names)
