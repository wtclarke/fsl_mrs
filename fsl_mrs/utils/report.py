#!/usr/bin/env python

# report.py - Generate html report
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#         William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

import pandas as pd
import numpy as np
import os


import plotly
import plotly.graph_objs as go
import nibabel as nib

from fsl_mrs.utils import plotting
from fsl_mrs.utils import misc


def to_div(fig):
    """
    Turns Plotly Figure into HTML
    """
    return plotly.offline.plot(fig,
                               output_type='div',
                               include_plotlyjs='cdn')


def create_plotly_div(mrs, res):
    """
    Create HTML <div> sections for a few different report figures
    The output is a dict

    The following sections are currently produced:

    summary               : Table of concentrations and plot of the fit
    table-lineshape-phase : Table of some nuisance params
    table-qc              : Table of QC parameters
    posteriors            : Heatmap of posterior correlactions and histograms of marginals
    real-imag             : Separate fit for real and imaginary part of the spectrum
    indiv                 : Plot of each metabolite prediction separately
    quant_table           ; Table with quantification results

    """
    divs = {}

    # Summary plot
    fig = plotting.plotly_fit(mrs, res)
    divs['summary'] = to_div(fig)

    # tables
    #  nuisance
    fig = plotting.plot_table_lineshape_phase(res)
    divs['table-lineshape-phase'] = to_div(fig)
    #  qc
    fig = plotting.plot_table_qc(res)
    divs['table-qc'] = to_div(fig)

    # MCMC results (if available)
    if res.method == 'MH':
        fig1 = plotting.plot_corr(res, title='MCMC Correlations')
        fig2 = plotting.plot_dist_mcmc(res, refname=res.concScalings['internalRef'])
    else:
        fig1 = plotting.plot_corr(res, corr=res.corr, title='Laplace approx Correlations')
        fig2 = plotting.plot_dist_approx(res, refname=res.concScalings['internalRef'])
    divs['corr'] = to_div(fig1)
    divs['posteriors'] = to_div(fig2)

    fig = plotting.plot_real_imag(mrs, res, ppmlim=(.2, 4.2))
    divs['real-imag'] = to_div(fig)  # plotly.offline.plot(fig, output_type='div',include_plotlyjs='cdn')

    fig = plotting.plot_indiv_stacked(mrs, res, ppmlim=res.ppmlim)
    divs['indiv'] = to_div(fig)  # plotly.offline.plot(fig, output_type='div',include_plotlyjs='cdn')

    # Quantification table
    if (res.concScalings['molality'] is not None) and res.concScalings['quant_info'].f_GM is not None:
        Q = res.concScalings['quant_info']
        quant_df = pd.DataFrame()
        quant_df['Tissue-water densities (g/cm^3)'] = [Q.d_GM, Q.d_WM, Q.d_CSF]
        quant_df['Tissue volume fractions'] = [Q.f_GM, Q.f_WM, Q.f_CSF]
        quant_df['Water T2 (ms)'] = np.around([Q.t2['H2O_GM'] * 1000, Q.t2['H2O_WM'] * 1000, Q.t2['H2O_CSF'] * 1000])
        quant_df['Water T1 (s)'] = np.around([Q.t1['H2O_GM'], Q.t1['H2O_WM'], Q.t1['H2O_CSF']], decimals=2)
        quant_df.index = ['GM', 'WM', 'CSF']
        quant_df.index.name = 'Tissue'
        quant_df.reset_index(inplace=True)
        tab = plotting.create_table(quant_df)
        fig = go.Figure(data=[tab])
        fig.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0))
        divs['quant_table'] = to_div(fig)

    if res.concScalings['molality'] is not None:
        fig = plotting.plot_references(mrs, res)
        divs['refs'] = to_div(fig)

    return divs


def static_image(imgfile):
    """
    Create plotly Figure object for an image
    """
    import plotly.graph_objects as go
    from PIL import Image
    img = Image.open(imgfile)
    # Create figure
    fig = go.Figure()

    # Constants
    img_width = img.width
    img_height = img.height
    scale_factor = 0.5

    # Add invisible scatter trace.
    fig.add_trace(
        go.Scatter(
            x=[0, img_width * scale_factor],
            y=[0, img_height * scale_factor],
            mode="markers",
            marker_opacity=0
        )
    )

    # Configure axes
    fig.update_xaxes(
        visible=False,
        range=[0, img_width * scale_factor]
    )

    fig.update_yaxes(
        visible=False,
        range=[0, img_height * scale_factor],
        # the scaleanchor attribute ensures that the aspect ratio stays constant
        scaleanchor="x"
    )

    # Add image
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width * scale_factor,
            y=img_height * scale_factor,
            sizey=img_height * scale_factor,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=img)
    )

    # Configure other layout
    fig.update_layout(
        width=img_width * scale_factor,
        height=img_height * scale_factor,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    return fig


def create_report(mrs, res, filename, fidfile, basisfile, h2ofile, date, location_fig=None):
    """
    Creates and writes to file an html report
    """

    divs = create_plotly_div(mrs, res)

    template = f"""<!DOCTYPE html>
    <html>
    <head>
    <style>
    pre {{
    display: block;
    font-family: monospace;
    font-size : 16px;
    white-space: pre;
    margin: 1em 0;
    }}
    .fixed{{
        width: 200px;
    }}
    .flex-item{{
        flex-grow: 1;
    }}

    </style>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body style="background-color:white">
    <div class="header">
    <center><h1>FSL MRS Report</h1></center>
    <hr>
    <pre>
    Date        : {date}
    FID         : {fidfile}
    Basis       : {basisfile}
    H2O         : {h2ofile}
    </pre>
    </div>
    <hr>
    <center>
    <a href="#summary">Summary</a> -
    <a href="#nuisance">Nuisance</a> -
    <a href="#qc">QC</a> -
    <a href="#uncertainty">Uncertainty</a> -
    <a href="#realimag">Real/Imag</a> -
    <a href="#metabs">Metabs</a> -
    <a href="#quantification">Quantification</a> -
    <a href="#methods">Methods</a>
    </center>
    <hr>
    """

    # Summary section
    section = f"""
    <h1><a name="summary">Summary</a></h1>
    <div id=fit>{divs['summary']}</div>
    <hr>
    """
    template += section

    # Location
    if location_fig is not None:
        fig = static_image(location_fig)
        template += f"<h1>Voxel location</h1><div>{to_div(fig)}</div><hr>"

    # Tables section
    section = f"""
    <h1><a name="nuisance">Nuisance parameters</a></h1>
    <div style="width:70%">{divs['table-lineshape-phase']}</div>

    <hr>
    <h1><a name="qc">QC parameters</a></h1>
    <div style="width:70%">{divs['table-qc']}</div>
    <hr>
    """
    template += section

    # Dist section
    section = f"""
    <h1><a name="uncertainty">Uncertainties</a></h1>
    <table width=100%>
    <tr>
    <th style="vertical-align:top">{divs['corr']}</th>
    <th style="vertical-align:top">{divs['posteriors']}</th>
    </tr>
    </table>
    <hr>
    """
    template += section

    # Real/imag section
    section = f"""
    <h1><a name="realimag">Fitting summary (real/imag)</a></h1>
    {divs['real-imag']}
    <hr>
    """
    template += section

    # Indiv spectra section
    section = f"""
    <h1><a name="metabs">Individual metabolite spectra</a></h1>
    {divs['indiv']}
    <hr>
    """
    template += section

    # Quantification information
    # Table of CSF,GM,WM
    # Fractions
    # T2 info (water)
    # T1 info (water)
    # T2 info (metab)
    # T1 info (metab)
    # Tissue water densities
    # Relaxation corrected water
    # Relaxation correction for metab
    # Final scalings for molality & molarity
    if res.concScalings['molality'] is not None:
        Q = res.concScalings['quant_info']
        relax_water_conc = Q.relax_corr_water_molar
        metabRelaxCorr = Q.relax_corr_metab

        if res.concScalings['quant_info'].f_GM is not None:
            section = f"""
            <h1><a name="quantification">Quantification information</a></h1>
            <div style="width:70%">{divs['quant_table']}</div>
            <table>
                <tr>
                    <td class="titles">Metabolite T<sub>2</sub>:</td>
                    <td>{1000*Q.t2['METAB']} ms</td>
                </tr>
                <tr>
                    <td class="titles">Metabolite T<sub>1</sub>:</td>
                    <td>{Q.t1['METAB']} s</td>
                </tr>
                <tr>
                    <td class="titles">Sequence echo time (T<sub>E</sub>):</td>
                    <td>{1000*Q.te} ms</td>
                </tr>
                <tr>
                    <td class="titles">Sequence repetition time (T<sub>R</sub>):</td>
                    <td>{Q.tr} s</td>
                </tr>
                <tr>
                    <td class="titles">Relaxation corrected water concentration:</td>
                    <td>{relax_water_conc:0.0f} mmol/kg</td>
                </tr>
                <tr>
                    <td class="titles">Metabolite relaxation correction (1/e<sup>(-T<sub>E</sub>/T<sub>2</sub>)</sup>):</td>
                    <td>{metabRelaxCorr:0.2f}</td>
                </tr>
                <tr>
                    <td class="titles">Raw concentration to molarity scaling:</td>
                    <td>{res.concScalings["molarity"]:0.2f}</td>
                </tr>
                <tr>
                    <td class="titles">Raw concentration to molality scaling:</td>
                    <td>{res.concScalings["molality"]:0.2f}</td>
                </tr>
            </table>
            <hr>
            """
        else:
            section = f"""
            <h1><a name="quantification">Quantification information</a></h1>
            <table>
                <tr>
                    <td class="titles">Water T<sub>2</sub></td>
                    <td> : {(1000*Q.t2['H2O_GM'] + 1000*Q.t2['H2O_WM']) / 2} ms</td>
                </tr>
                <tr>
                    <td class="titles">Water T<sub>1</sub></td>
                    <td> : {(1000*Q.t1['H2O_GM'] + 1000*Q.t1['H2O_WM']) / 2} ms</td>
                </tr>
                <tr>
                    <td class="titles">Metabolite T<sub>2</sub>:</td>
                    <td>{1000*Q.t2['METAB']} ms</td>
                </tr>
                <tr>
                    <td class="titles">Metabolite T<sub>1</sub>:</td>
                    <td>{Q.t1['METAB']} s</td>
                </tr>
                <tr>
                    <td class="titles">Sequence echo time (T<sub>E</sub>):</td>
                    <td>{1000*Q.te} ms</td>
                </tr>
                <tr>
                    <td class="titles">Sequence repetition time (T<sub>R</sub>):</td>
                    <td>{Q.tr} s</td>
                </tr>
                <tr>
                    <td class="titles">T<sub>2</sub> relaxation corrected water concentration</td>
                    <td> : {relax_water_conc:0.0f} mmol/kg</td>
                </tr>
                <tr>
                    <td class="titles">Metabolite relaxation correction (1/e<sup>(-T<sub>E</sub>/T<sub>2</sub>)</sup>)</td>
                    <td> : {metabRelaxCorr:0.2f}</td>
                </tr>
                <tr>
                    <td class="titles">Raw concentration to molarity scaling</td>
                    <td> : {res.concScalings["molarity"]:0.2f}</td>
                </tr>
                <tr>
                    <td class="titles">Raw concentration to molality scaling</td>
                    <td> : {res.concScalings["molality"]:0.2f}</td>
                </tr>
            </table>
            <hr>
            """
        template += section

    # Real/imag section
    section = f"""
    <h2><a name="refs">Water referencing</a></h2>
    {divs['refs']}
    <hr>
    """
    template += section

    # Details of the methods
    # methodsfile="/path/to/file"
    # methods = np.readtxt(methodsfile)
    from fsl_mrs import __version__
    if res.method == "Newton":
        algo = "Model fitting was performed using the truncated Newton algorithm as implemented in Scipy."
    elif res.method == "MH":
        algo = "Model fitting was performed using the Metropolis Hastings algorithm."
    else:
        algo = ""

    methods = f"<p>Fitting of the SVS data was performed using a Linear Combination model as described in [1] and as implemented in FSL-MRS version {__version__}, part of FSL (FMRIB's Software Library, www.fmrib.ox.ac.uk/fsl). Briefly, basis spectra are fitted to the complex-valued spectrum in the frequency domain. The basis spectra are shifted and broadened with parameters fitted to the data and grouped into {max(res.metab_groups)+1} metabolite groups. A complex polynomial baseline is also concurrrently fitted (order={res.baseline_order}). {algo} <p><h3>References</h3><p>[1] Clarke WT, Jbabdi S. FSL-MRS: An end-to-end spectroscopy analysis package (2020)."
    section = f"""
    <h1><a name="methods">Analysis methods</a></h1>
    <div>{methods}</div>
    <hr>
    """
    template += section

    # End of report
    template += """
    </body>
    </html>
    """

    # write
    with open(filename, 'w') as f:
        f.write(template)


def fitting_summary_fig(mrs, res, filename):
    """
    Simple spectrum+fit plot
    """
    fig = plotting.plot_fit(mrs, pred=res.pred, baseline=res.baseline)
    fig.savefig(filename)


# --------- MRSI reporting
def save_params(params, names, data_hdr, mask, folder, cleanup=True):
    """
    Save MRSI results into NIFTI image files
    """
    for i, p in enumerate(names):
        x = misc.list_to_volume(list(params[:, i]), mask)
        if cleanup:
            x[np.isnan(x)] = 0
            x[np.isinf(x)] = 0
            x[x < 1e-10] = 0
            x[x > 1e10] = 0

        img = nib.Nifti1Image(x, data_hdr.affine)
        filename = os.path.join(folder, p + '.nii.gz')
        nib.save(img, filename)
