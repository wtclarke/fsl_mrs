import json
from copy import deepcopy
from pathlib import Path
from sys import stdout
import argparse

# 3rd party imports
import pandas as pd

# Dash and plotly imports
from dash import Dash, dcc, html, ctx
# from dash import no_update
import dash.dash_table as dct
from dash.dash_table.Format import Format
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go

# FSL-MRS imports
from fsl_mrs.utils import results
from fsl_mrs.utils import mrs_io
from fsl_mrs.utils import misc
from fsl_mrs.utils import plotting
from fsl_mrs.utils.baseline import prepare_baseline_regressor


def main():
    parser = argparse.ArgumentParser(
        description="FSL Magnetic Resonance Spectroscopy"
                    " - Dashboard for SVS results.")

    parser.add_argument(
        'input_type',
        choices=['dir', 'list'],
        help='Select between input type')
    parser.add_argument(
        'input',
        type=Path,
        metavar='DIR or FILE',
        help='Directory contiaining individual results directories '
             'or Text file contiaining line-separated list of results directories.')

    # ADDITONAL OPTIONAL ARGUMENTS
    parser.add_argument('-v', '--verbose',
                        required=False, action="store_true")
    parser.add_argument('-p', '--port',
                        required=False,
                        type=int,
                        default=8050)

    # Parse command-line arguments
    args = parser.parse_args()

    '''Styles'''
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = Dash(__name__, external_stylesheets=external_stylesheets)

    discrete_colour = px.colors.qualitative.Plotly

    '''
    Load the data
    1. The concentration.csv files
    2. The qc.csv files
    3. Load the all_parameters.csv and regenerate mrs/results objects
    '''

    verbose = args.verbose
    # Deal with the two inputs.
    res_dir = []
    if args.input_type == 'dir':
        for fp in args.input.rglob('concentrations.csv'):
            res_dir.append(fp.parent)
    else:
        with open(args.input) as fp:
            res_dir = fp.read().splitlines()
            res_dir = [Path(dir) for dir in res_dir]

    # Check for duplicate names
    # list(dict.fromkeys(x)) finds unique values preserving order
    if len(list(dict.fromkeys(res_dir))) < len(res_dir):
        raise ValueError('Input directories must not be duplicated.')

    res_dir_for_names = [fp for fp in res_dir]
    fit_names = [fp.name for fp in res_dir_for_names]
    while len(list(dict.fromkeys(fit_names))) < len(res_dir):
        res_dir_for_names = [fp.parent for fp in res_dir_for_names]

        if res_dir_for_names[0] == Path('/'):
            raise ValueError('Inputs must be uniquely identifiable at single level.')

        fit_names = [fp.name for fp in res_dir_for_names]

    # 1. Concentration.csv
    if verbose:
        print('Loading concentration data.')
    all_conc = []
    for fp in res_dir:
        all_conc.append(pd.read_csv(fp / 'concentrations.csv', index_col=0, header=[0, 1]))
    conc_df = pd.concat(all_conc, keys=fit_names, names=['dataset', 'Metabolite'])
    col = conc_df.columns
    col.names = [None, None]
    conc_df.columns = col
    conc_df = conc_df.reorder_levels(['Metabolite', 'dataset'], axis=0).sort_index()

    # 2. qc.csv
    if verbose:
        print('Loading QC data.')
    all_qc = []
    for fp in res_dir:
        all_qc.append(pd.read_csv(fp / 'qc.csv', index_col=0, header=0))
    qc_df = pd.concat(all_qc, keys=fit_names, names=['dataset', 'Metabolite'])
    qc_df = qc_df.reorder_levels(['Metabolite', 'dataset'], axis=0).sort_index()

    # 3. Load the all_parameters.csv and regenerate mrs/results objects
    if verbose:
        print('Loading data & results.')
    mrs_store = {}
    res_store = {}
    n_data = len(res_dir)
    for idx, (fp, name) in enumerate(zip(res_dir, fit_names)):
        param_df = pd.read_csv(fp / 'all_parameters.csv')

        # Read options.txt
        with open(fp / 'options.txt', "r") as f:
            orig_args = json.loads(f.readline())

        # Load data into mrs object
        if (fp / 'data').exists():
            FID = mrs_io.read_FID((fp / 'data').resolve())
            basis = mrs_io.read_basis((fp / 'basis').resolve())
        else:  # Use old mechanism for backwards compatibility
            FID = mrs_io.read_FID(orig_args['data'])
            basis = mrs_io.read_basis(orig_args['basis'])
        mrs = FID.mrs(basis=basis)

        if orig_args['conjbasis'] is not None:
            if orig_args['conjbasis']:
                mrs.conj_Basis = True
        else:
            _ = mrs.check_Basis(repair=True)

        if not orig_args['no_rescale']:
            mrs.rescaleForFitting(ind_scaling=orig_args['ind_scale'])
        mrs.keep = orig_args['keep']
        mrs.ignore = orig_args['ignore']

        if orig_args['lorentzian']:
            model = 'lorentzian'
        else:
            model = 'voigt'

        method = orig_args['algo']
        # Generate metabolite groups
        metab_groups = misc.parse_metab_groups(mrs, orig_args['metab_groups'])
        baseline_order = orig_args['baseline_order']
        if baseline_order < 0:
            baseline_order = 0  # Required to make prepare_baseline_regressor run.
        ppmlim = orig_args['ppmlim']
        # Generate baseline polynomials (B)
        B = prepare_baseline_regressor(mrs, baseline_order, ppmlim)

        # Generate results object
        res_store[name] = results.FitRes(
            mrs,
            param_df['mean'].to_numpy(),
            model, method, metab_groups, baseline_order, B, ppmlim,
            runqc=False)

        if orig_args['combine'] is not None:
            res_store[name].combine(orig_args['combine'])

        mrs_store[name] = deepcopy(mrs)
        if verbose:
            stdout.write(f'\r{idx + 1}/{n_data} data & results loaded.')
            stdout.flush()
            # print()
    if verbose:
        print('\n')

    # Figure out metabolites in dataset. Select a metabolite to initilise
    all_metabs = conc_df.index.get_level_values(0).unique().to_list()
    if 'NAA' in all_metabs:
        start_metabs = ['NAA']
    else:
        start_metabs = []

    columns_conc = conc_df\
        .rename(columns={'mean': 'Conc', 'std': 'SD'})\
        .columns.map('|'.join).str.strip('|').to_list()

    if 'Conc|molality' in columns_conc:
        columns_start = 'Conc|molality'
    elif 'Conc|internal' in columns_conc:
        columns_start = 'Conc|internal'
    else:
        columns_start = 'Conc|raw'

    # AvgFig
    avg_fig = plotting.plotly_avg_fit(list(mrs_store.values()), list(res_store.values()))
    avg_fig.update_layout(
        title='Average data',
        height=450,
        margin={'l': 10, 'b': 5, 'r': 10, 't': 40},
        template='plotly_white')

    # Default blank figure for init
    def blank_fig():
        fig = go.Figure(go.Scatter(x=[], y=[]))
        fig.update_layout(template=None)
        fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
        fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)
        return fig

    def blank_table(tab_id):
        blank_entries = [
            {'index': 'count', 'N/A': None},
            {'index': 'mean', 'N/A': None},
            {'index': 'std', 'N/A': None},
            {'index': 'min', 'N/A': None},
            {'index': '25%', 'N/A': None},
            {'index': '50%', 'N/A': None},
            {'index': '75%', 'N/A': None},
            {'index': 'max', 'N/A': None}]
        return dct.DataTable(
            id=tab_id,
            data=blank_entries)

    if verbose:
        print('Rendering...')

    app.layout = html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(
                    all_metabs,
                    start_metabs,
                    multi=True,
                    id='metabolite-selection')],
                style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Dropdown(
                    columns_conc,
                    columns_start,
                    id='conc-selection')],
                style={'width': '23%', 'display': 'inline-block'}),
            html.Div([
                html.Div(children='Copy Conc:', style={'display': 'inline-block'}),
                dcc.Clipboard(
                    id="conc-table-copy",
                    className="button",
                    style={'height': 35, 'width': 35, 'display': 'inline-block'}),
                html.Div(children='Copy QC:', style={'display': 'inline-block'}),
                dcc.Clipboard(
                    id="qc-table-copy",
                    className="button",
                    style={'height': 35, 'width': 35, 'display': 'inline-block'})],
                style={'width': '28%', 'float': 'right', 'display': 'inline-block'})
        ]),

        html.Div([
            html.Div([
                dcc.Graph(
                    id='conc-figure',
                )],
                style={'width': '60%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='fwhm-figure',
                )],
                style={'width': '19%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='snr-figure',
                )],
                style={'width': '19%', 'float': 'right', 'display': 'inline-block'}),
        ]),

        html.Div([
            html.Div(
                [blank_table('conc-table')],
                id='conc-table-container',
                style={'width': '60%', 'vertical-align': 'middle', 'display': 'inline-block'}),
            html.Div(
                [blank_table('qc-table')],
                id='qc-table-container',
                style={'width': '35%', 'vertical-align': 'middle', 'float': 'right', 'display': 'inline-block'})
        ]),

        html.Div([
            html.Div([
                dcc.Graph(
                    id='results-figure',
                    figure=blank_fig())],
                style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(
                    id='avg-plot',
                    figure=avg_fig)],
                style={'width': '48%', 'display': 'inline-block', 'float': 'right'})])
    ])

    def create_conc_violin(metabs, field, selecteddataset):
        formatted_df = conc_df.loc[metabs].reset_index().rename(columns={'mean': 'Conc', 'std': 'SD'})
        formatted_df.columns = formatted_df.columns.map('|'.join).str.strip('|')

        fig = px.violin(
            formatted_df,
            color="Metabolite", x="Metabolite",
            y=field,
            box=True, points='all',
            custom_data=['dataset'],
            hover_data=['dataset'])
        fig.layout.update(showlegend=False)

        fig.update_layout(
            clickmode='event+select',
            margin={'r': 5, 't': 25, 'b': 20},
            template='plotly_white')

        if selecteddataset is not None:
            selected_idx = formatted_df.index[formatted_df.dataset.isin(selecteddataset)].to_list()
            fig.update_traces(
                selectedpoints=selected_idx,
                unselected={'marker': {'opacity': 0.3}})

        fig.update_traces(marker_size=10)
        return fig

    @app.callback(
        Output('conc-figure', 'figure'),
        Input('metabolite-selection', 'value'),
        Input('conc-selection', 'value'),
        Input('conc-figure', 'selectedData'),
        Input('fwhm-figure', 'selectedData'),
        Input('snr-figure', 'selectedData'),)
    def update_conc_violin(metabs, field, selected_data_self, selected_data1, selected_data2):
        if selected_data_self is not None\
                or selected_data1 is not None\
                or selected_data2 is not None:
            if selected_data_self is not None\
                    and ctx.triggered_id in ['conc-figure', 'metabolite-selection', 'conc-selection']:
                # This is here so all metabolites are updated.
                selecteddataset = [sd['customdata'][0] for sd in selected_data_self['points']]
            elif selected_data1 is not None\
                    and ctx.triggered_id == 'fwhm-figure':
                selecteddataset = [sd['customdata'][0] for sd in selected_data1['points']]
            elif selected_data2 is not None\
                    and ctx.triggered_id == 'snr-figure':
                selecteddataset = [sd['customdata'][0] for sd in selected_data2['points']]
            else:
                selecteddataset = None
        else:
            selecteddataset = None
        return create_conc_violin(metabs, field, selecteddataset)

    # Results
    def create_results_figure(dataset):
        fig = plotting.plotly_spectrum(mrs_store[dataset], res_store[dataset])
        fig.update_layout(
            title=dataset,
            height=450,
            margin={'l': 10, 'b': 5, 'r': 10, 't': 40},
            template='plotly_white')
        fig.update_traces(
            cells=dict(height=15),
            cells_font=dict(size=8),
            header_font=dict(size=8),
            selector=dict(type="table"))
        return fig

    @app.callback(
        Output('results-figure', 'figure'),
        Input('conc-figure', 'selectedData'),
        Input('fwhm-figure', 'selectedData'),
        Input('snr-figure', 'selectedData'),
        Input('conc-figure', 'hoverData'))
    def update_results_fig(select_data_conc, selected_data_fwhm, selected_data_snr, hover_data):
        if select_data_conc is not None\
                and ctx.triggered_id == 'conc-figure':
            dataset = select_data_conc['points'][0]['customdata'][0]
        elif selected_data_fwhm is not None\
                and ctx.triggered_id == 'fwhm-figure':
            dataset = selected_data_fwhm['points'][0]['customdata'][0]
        elif selected_data_snr is not None\
                and ctx.triggered_id == 'snr-figure':
            dataset = selected_data_snr['points'][0]['customdata'][0]
        elif hover_data is not None\
                and ctx.triggered_id == 'conc-figure':
            dataset = hover_data['points'][0]['customdata'][0]
        else:
            dataset = None
            return blank_fig()
        return create_results_figure(dataset)

    # QC
    def create_qc_figure(metabolite, selecteddataset, qctype, colour):
        try:
            formatted_qc_df = qc_df.loc[[metabolite, ]].reset_index()
        except KeyError:
            return blank_fig()

        fig = px.violin(
            formatted_qc_df,
            x="Metabolite",
            y=qctype,
            box=True, points='all',
            hover_data=['dataset'],
            color_discrete_sequence=[colour, ])

        if selecteddataset is not None:
            selected_idx = formatted_qc_df.index[formatted_qc_df.dataset.isin(selecteddataset)].to_list()
            fig.update_traces(
                selectedpoints=selected_idx,
                unselected={'marker': {'opacity': 0.3}})

        fig.update_layout(
            showlegend=False,
            clickmode='event+select',
            margin={'r': 5, 't': 25, 'b': 20},
            template='plotly_white')
        return fig

    @app.callback(
        Output('fwhm-figure', 'figure'),
        Output('snr-figure', 'figure'),
        Input('conc-figure', 'selectedData'),
        Input('conc-figure', 'hoverData'),
        Input('fwhm-figure', 'selectedData'),
        Input('snr-figure', 'selectedData'))
    def update_qc_figs(select_data, hover_data, fwhm_select, snr_select):
        # If selection made and callback triggered by hover then no update
        # if (select_data is not None
        #         or fwhm_select is not None
        #         or fwhm_select is not None)\
        #         and ctx.triggered_prop_ids == {'conc-figure.hoverData': 'conc-figure'}:
        #     return no_update, no_update

        if select_data is not None:
            metab = select_data['points'][0]['x']
            selecteddataset = [sd['customdata'][0] for sd in select_data['points']]
            colour = discrete_colour[select_data['points'][0]['curveNumber']]
        elif hover_data is not None:
            metab = hover_data['points'][0]['x']
            selecteddataset = None
            colour = discrete_colour[hover_data['points'][0]['curveNumber']]
        elif ctx.triggered_id is None:
            # I think this is the initialisation pass
            metab = start_metabs[0]
            selecteddataset = None
            colour = discrete_colour[0]
        else:
            metab = None
            selecteddataset = None
            colour = None
            return blank_fig(), blank_fig()

        if ctx.triggered_id == 'fwhm-figure'\
                and fwhm_select is not None:
            selecteddataset = [sd['customdata'][0] for sd in fwhm_select['points']]
        elif ctx.triggered_id == 'snr-figure'\
                and snr_select is not None:
            selecteddataset = [sd['customdata'][0] for sd in snr_select['points']]
        elif ctx.triggered_id in ['fwhm-figure', 'snr-figure']:
            selecteddataset = None
        # print(selecteddataset)
        return create_qc_figure(metab, selecteddataset, 'FWHM', colour),\
            create_qc_figure(metab, selecteddataset, 'SNR', colour)

    @app.callback(
        Output("conc-table-container", "children"),
        Input('metabolite-selection', 'value'),
        Input('conc-selection', 'value'))
    def update_conc_table(metabolites, field):
        if metabolites == []:
            return blank_table('conc-table')
        df = conc_df.rename(columns={'mean': 'Conc', 'std': 'SD'})
        df.columns = df.columns.map('|'.join).str.strip('|')
        df = df.loc[metabolites, field].unstack('Metabolite').describe().reset_index()

        col_format = []
        for col in df.columns:
            if col == 'index':
                col_format.append({"name": col, "id": col})
            else:
                col_format.append({"name": col, "id": col, 'type': 'numeric', 'format': Format(precision=3)})

        return dct.DataTable(
            id='conc-table',
            data=df.to_dict('records'),
            columns=col_format,
            style_cell_conditional=[
                {'if': {'column_id': 'index'},
                 'width': '10%',
                 'textAlign': 'left'},
                {'if': {'column_type': 'numeric'},
                 'textAlign': 'center'}],
            style_as_list_view=True)

    @app.callback(
        Output("qc-table-container", "children"),
        Input('conc-figure', 'selectedData'),
        Input('conc-figure', 'hoverData'))
    def update_qc_table(select_data, hover_data):
        if select_data is not None:
            metab = select_data['points'][0]['x']
        elif hover_data is not None:
            metab = hover_data['points'][0]['x']
        elif ctx.triggered_id is None:
            # I think this is the initialisation pass
            metab = start_metabs[0]
        else:
            metab = None
            return None
        try:
            df = qc_df.loc[metab, ['FWHM', 'SNR']].describe().reset_index()
        except KeyError:
            return None

        col_format = []
        for col in df.columns:
            if col == 'index':
                col_format.append({"name": col, "id": col})
            else:
                col_format.append({"name": col, "id": col, 'type': 'numeric', 'format': Format(precision=3)})

        return dct.DataTable(
            id='qc-table',
            data=df.to_dict('records'),
            columns=col_format,
            style_cell_conditional=[
                {'if': {'column_id': 'index'},
                 'width': '17%',
                 'textAlign': 'left'},
                {'if': {'column_type': 'numeric'},
                 'textAlign': 'center'}],
            style_as_list_view=True)

    @app.callback(
        Output("conc-table-copy", "content"),
        Input("conc-table-copy", "n_clicks"),
        State("conc-table", "derived_virtual_data"))
    def custom_copy(_, data):
        dff = pd.DataFrame(data)
        return dff.to_csv(index=False)  # includes headers

    @app.callback(
        Output("qc-table-copy", "content"),
        Input("qc-table-copy", "n_clicks"),
        State("qc-table", "derived_virtual_data"))
    def custom_copy2(_, data):
        dff = pd.DataFrame(data)
        return dff.to_csv(index=False)  # includes headers

    app.run(debug=True, port=args.port)


if __name__ == '__main__':
    main()
