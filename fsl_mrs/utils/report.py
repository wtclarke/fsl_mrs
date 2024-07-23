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
from pathlib import Path

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


'''Sections and layout for the single voxel report
The following sections are currently produced:

    summary               : Table of concentrations and plot of the fit
    table-lineshape-phase : Table of some nuisance params
    table-qc              : Table of QC parameters
    posteriors            : Heatmap of posterior correlations and histograms of marginals
    real-imag             : Separate fit for real and imaginary part of the spectrum
    indiv                 : Plot of each metabolite prediction separately
    table-quant           ; Table with quantification results'''


def svs_summary_div(mrs, res):
    """fitting summary - single spectrum fit"""
    fig = plotting.plotly_fit(mrs, res)
    return to_div(fig)


def svs_table_lineshape_phase_div(res):
    """table of nuisance parameters (lineshape, phase, shift)."""
    fig = plotting.plot_table_lineshape_phase(res)
    return to_div(fig)


def svs_table_qc(res):
    """Table of qc parameters (SNR & FWHM)"""
    fig = plotting.plot_table_qc(res)
    return to_div(fig)


def svs_correlations(res):
    """Metabolite concentration correlation figure"""
    if res.method == 'MH':
        fig = plotting.plot_corr(res, title='MCMC Correlations')
    else:
        fig = plotting.plot_corr(res, corr=res.corr, title='Laplace approx Correlations')
    return to_div(fig)


def svs_posteriors(res):
    """Concentration parameter posteriors"""
    if res.method == 'MH':
        fig = plotting.plot_dist_mcmc(res, refname=res.concScalings['internalRef'])
    else:
        fig = plotting.plot_dist_approx(res, refname=res.concScalings['internalRef'])
    return to_div(fig)


def svs_real_imag_plot(mrs, res):
    """View of real and imaginary components with fit"""
    fig = plotting.plot_real_imag(mrs, res, ppmlim=res.ppmlim)
    return to_div(fig)


def svs_basis_plot(mrs, res):
    """View of basis spectrum with data"""
    fig = plotting.plotly_basis(mrs, ppmlim=res.ppmlim)
    return to_div(fig)


def svs_indiv_plot(mrs, res):
    """View of each individual scaled metabolite basis spectrum."""
    fig = plotting.plot_indiv_stacked(mrs, res, ppmlim=res.ppmlim)
    return to_div(fig)


def svs_table_tissue_quant(res):
    """Quantification information table"""
    quant_df = res.concScalings['quant_info'].summary_table
    quant_df.reset_index(inplace=True)
    tab = plotting.create_table(quant_df)
    fig = go.Figure(data=[tab])
    fig.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0))
    return to_div(fig)


def svs_table_quant(res):
    """ Table containing Quantification information
    Table of CSF,GM,WM
    Fractions
    T2 info (water)
    T1 info (water)
    T2 info (metab)
    T1 info (metab)
    Tissue water densities
    Relaxation corrected water
    Relaxation correction for metab
    Final scalings for molality & molarity
    """

    Q = res.concScalings['quant_info']
    relax_water_conc = Q.relax_corr_water_molar
    metabRelaxCorr = Q.relax_corr_metab

    if res.concScalings['quant_info'].f_GM is not None:
        extra_div = svs_table_tissue_quant(res)
        table = f"""
        <div style="width:70%">{extra_div}</div>
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
        table = f"""
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
                <td class="titles">Relaxation corrected water concentration</td>
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
    return table


def svs_plot_refs(mrs, res):
    """View of referencing integration areas"""
    fig = plotting.plot_references(mrs, res)
    return to_div(fig)


def svs_methods_summary(res):
    """Text summary of the methods used."""
    from fsl_mrs import __version__
    if res.method == "Newton":
        algo = "Model fitting was performed using the truncated Newton algorithm as implemented in Scipy."
    elif res.method == "MH":
        algo = "Model fitting was performed using the Metropolis Hastings algorithm."
    else:
        algo = ""

    if res.baseline_mode == "off":
        baseline_string = "No baseline was fitted. "
    else:
        baseline_string = f"A {res._baseline_obj} is concurrently fitted. "

    return \
        f"<p>Fitting of the SVS data was performed using a Linear Combination model as described in [1] and as implemented in FSL-MRS version {__version__}, "\
        f"part of FSL (FMRIB's Software Library, www.fmrib.ox.ac.uk/fsl). "\
        f"Briefly, basis spectra are fitted to the complex-valued spectrum in the frequency domain. "\
        f"The basis spectra are shifted and broadened with parameters fitted to the data and grouped into {max(res.metab_groups)+1} metabolite groups. "\
        f"{baseline_string}"\
        f"{algo} "\
        f"<p><h3>References</h3><p>[1] Clarke WT, Stagg CJ, Jbabdi S. FSL-MRS: An end-to-end spectroscopy analysis package. Magnetic Resonance in Medicine 2021;85:2950-2964 doi: 10.1002/mrm.28630."


'''Layout for the svs report'''


def create_svs_sections(mrs, res, location_fig):
    """
    Create the HTML sections for svs report figures and tables
    includes section headings.
    Sections will appear in order added to list
    The output is a list
    """
    sections = []
    sections.append(
        f"""
        <h1><a name="summary">Summary</a></h1>
        <div id=fit>{svs_summary_div(mrs, res)}</div>
        <hr>
        """)

    if location_fig is not None:
        fig = static_image(location_fig)
        sections.append(
            f"""
            <h1>Voxel location</h1>
            <div>{to_div(fig)}</div>
            <hr>
            """)

    # Tables section
    sections.append(
        f"""
        <h1><a name="nuisance">Nuisance parameters</a></h1>
        <div style="width:70%">{svs_table_lineshape_phase_div(res)}</div>

        <hr>
        <h1><a name="qc">QC parameters</a></h1>
        <div style="width:70%">{svs_table_qc(res)}</div>
        <hr>
        """)

    sections.append(
        f"""
        <h1><a name="uncertainty">Uncertainties</a></h1>
        <table width=100%>
        <tr>
        <th style="vertical-align:top">{svs_correlations(res)}</th>
        <th style="vertical-align:top">{svs_posteriors(res)}</th>
        </tr>
        </table>
        <hr>
        """)

    sections.append(
        f"""
        <h1><a name="basis">Basis spectra summary</a></h1>
        {svs_basis_plot(mrs, res)}
        <hr>
        """)

    sections.append(
        f"""
        <h1><a name="metabs">Individual metabolite estimates</a></h1>
        {svs_indiv_plot(mrs, res)}
        <hr>
        """)

    if res.concScalings['molality'] is not None:
        sections.append(
            f"""
            <h1><a name="quantification">Quantification information</a></h1>
            {svs_table_quant(res)}
            <hr>
            """)

    if res.concScalings['molality'] is not None:
        sections.append(
            f"""
            <h2><a name="refs">Water referencing</a></h2>
            {svs_plot_refs(mrs, res)}
            <hr>
            """)

    sections.append(
        f"""
        <h1><a name="methods">Analysis methods</a></h1>
        <div>{svs_methods_summary(res)}</div>
        <hr>
        """)

    sections_titles = {
        'summary': 'Summary',
        'nuisance': 'Nuisance',
        'qc': 'QC',
        'uncertainty': 'Uncertainty',
        'basis': 'Basis Spectra',
        'metabs': 'Metabs',
        'quantification': 'Quantification',
        'methods': 'Methods'}

    return sections, sections_titles


def create_svs_report(mrs, res, filename, fidfile, basisfile, h2ofile, date, location_fig=None):

    title = "FSL MRS Report"

    preamble = f"""
        <pre>
        Date        : {date}
        FID         : {fidfile}
        Basis       : {basisfile}
        H2O         : {h2ofile}
        </pre>
        """

    sections, section_titles = create_svs_sections(mrs, res, location_fig)

    create_report(title, preamble, section_titles, sections, filename)


'''Sections and layout for the dynamic report'''


def dyn_summary_div(dynres):
    """fitting summary - dynamic spectrum fit"""
    fig = plotting.plotly_dynMRS(
        dynres._dyn.mrs_list,
        dynres.reslist,
        dynres._dyn.time_var)
    return to_div(fig)


def dyn_mapped_div(dynres):
    """Models compared to independent fitting"""
    fig = dynres.plot_mapped(fit_to_init=True)

    return to_div(static_image(fig))


def dyn_fit_q_div(dynres):
    """Generate output on dynamic fit quality"""
    fig = dynres.plot_residuals()
    info_str = '<pre>\n'
    for param in dynres.model_parameters:
        if isinstance(dynres.model_parameters[param], float):
            info_str += f'{param}:   {dynres.model_parameters[param]:0.1f}\n'
        else:
            info_str += f'{param}:   {dynres.model_parameters[param]:0.0f}\n'
    info_str += '</pre>\n'

    return f'''<h3>Fit quality parameters</h3>
               <div id=fit>{info_str}</div>
               <h3>Residuals</h3>
               {to_div(static_image(fig))}
               '''


def dyn_corr(dynres):
    """Generate a HTML div containing the free parameter correlations"""
    return to_div(dynres.plot_corr())


def dyn_free_parameter_summaries(dynres, category):
    """Summary table of mapped parameters"""
    collected_result = dynres.collected_results()[category]
    collected_result.reset_index(inplace=True)
    tab = plotting.create_table(collected_result)
    fig = go.Figure(data=[tab])
    fig.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0))
    return to_div(fig)


def dyn_free_parameter_uncertainties(dynres):
    """Summary table of free parameter SDs"""
    df = pd.concat((dynres.mean_free, dynres.std_free), axis=1, keys=['mean', 'sd'])
    df.reset_index(inplace=True)
    tab = plotting.create_table(df)
    fig = go.Figure(data=[tab])
    fig.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0))
    return to_div(fig)


def dyn_methods_summary(res):
    """Text summary of the methods used."""
    from fsl_mrs import __version__

    return "<p>Fitting was performed using FSL-MRS's dynamic model fitting tool "\
           f"as described in [1] and [2] and as implemented in FSL-MRS version {__version__}, "\
           "part of FSL (FMRIB's Software Library, www.fmrib.ox.ac.uk/fsl). "\
           "<p><h3>References</h3><p>"\
           "[1] Clarke WT, Stagg CJ, Jbabdi S. FSL-MRS: An end-to-end spectroscopy analysis package. "\
           "Magnetic Resonance in Medicine 2021;85:2950-2964 doi: 10.1002/mrm.28630."\
           "[2] Clarke WT, Ligneul C, Cottaar M, Ip IB, Jbabdi S. Universal Dynamic Fitting of Magnetic Resonance Spectroscopy. "\
           "BioRxiv 2013. https://doi.org/10.1101/2023.06.15.544935."


def create_dyn_sections(dynres, location_fig):
    """
    Create the HTML sections for svs report figures and tables
    includes section headings.
    Sections will appear in order added to list
    The output is a list
    """
    sections = []

    sections.append(
        f"""
        <h1><a name="summary">Summary</a></h1>
        <div id=fit>{dyn_summary_div(dynres)}</div>
        <hr>
        """)

    sections.append(
        f"""
        <h1><a name="model_vis">Model Visualisation</a></h1>
        <div id=fit>{dyn_mapped_div(dynres)}</div>
        <hr>
        """)

    sections.append(
        f"""
        <h1><a name="fit_quality">Fit Quality</a></h1>
        <div id=fit>{dyn_fit_q_div(dynres)}</div>
        <hr>
        """)

    mapped_divs = []
    for cat in dynres._dyn.vm.mapped_categories:
        mapped_divs.append(
            f'''<h3>{cat}</h3>
            <div id=fit>{dyn_free_parameter_summaries(dynres, cat)}</div>
            ''')
    mapped_divs_str = '\n'.join(mapped_divs)
    sections.append(
        f"""
        <h1><a name="free_params">Free Parameter Summary</a></h1>
        <h3>Parameter Correlations</h3>
        {dyn_corr(dynres)}
        {mapped_divs_str}
        <hr>
        """)

    sections.append(
        f"""
        <h1><a name="uncertainty">Free Parameter Uncertianty</a></h1>
        <div id=fit>{dyn_free_parameter_uncertainties(dynres)}</div>
        <hr>
        """)

    sections.append(
        f"""
        <h1><a name="methods">Analysis methods</a></h1>
        <div>{dyn_methods_summary(dynres)}</div>
        <hr>
        """)

    sections_titles = {
        'summary': 'Summary',
        'model_vis': 'Model Visualisation',
        'free_params': 'Free Parameter Summary',
        'uncertainty': 'Free Parameter Uncertianty',
        'methods': 'Methods'}

    return sections, sections_titles


def create_dynmrs_report(dynres, filename, fidfile, basisfile, configfile, tvarfiles, date, location_fig=None):

    title = "FSL-MRS Dynamic Fitting Report"

    preamble = f"""
        <pre>
        Date           : {date}
        FID            : {fidfile}
        Basis          : {basisfile}
        Config File    : {configfile}
        Time Variables : {tvarfiles}
        </pre>
        """

    sections, section_titles = create_dyn_sections(dynres, location_fig)

    create_report(title, preamble, section_titles, sections, filename)


# -------- Report creation tools ---------

def create_report(title, preamble, section_titles, sections, filename):
    """Create a HTML report out of sections of HTML formatted data.
    """
    sectiont_text = '\n'.join([f'<a href="#{key}">{section_titles[key]}</a> -' for key in section_titles])

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
        <center><h1>{title}</h1></center>
        <hr>
        {preamble}
        </div>
        <hr>
        <center>
        {sectiont_text}
        </center>
        <hr>
        """
    for section in sections:
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
    plotting.plot_fit(mrs, res, out=filename)


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


# --------Utility functions --------
def static_image(img_in):
    """
    Create plotly Figure object for an image
    """
    import plotly.graph_objects as go
    from PIL import Image
    import matplotlib.figure
    print(type(img_in))
    if isinstance(img_in, matplotlib.figure.Figure):
        img_in.canvas.draw()
        img = Image.frombytes(
            'RGB',
            img_in.canvas.get_width_height(),
            img_in.canvas.tostring_rgb())
    elif isinstance(img_in, (str, Path)):
        img = Image.open(img_in)
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
