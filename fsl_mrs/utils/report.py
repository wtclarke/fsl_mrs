import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path as op
import os


import plotly
import plotly.graph_objs as go

import matplotlib.gridspec as gridspec
from plotly import tools

from fsl_mrs.utils import plotting
from fsl_mrs.utils import plot
from fsl_mrs.utils import misc

import nibabel as nib

import base64


# class Report(object):
#     '''
#     classdocs
#     '''

#     def __init__(self):
#         '''
#         Constructor
#         '''
#         pass


#     def parse(self, template: str, outname: str, **kwargs):

#         env = jinja2.Environment(
#             loader=jinja2.FileSystemLoader(op.dirname(template)),
#             extensions=['jinja2.ext.do']
#         )

#         args = dict(
#             report=self,
#             numpy=np,
#             pandas=pd,
#             plot=plot,
#             **kwargs
#         )

#         template = env.get_template(op.basename(template))
#         html = template.render(**args)

#         with open(outname, 'w') as outfile:
#             outfile.write(html)



#     @staticmethod
#     def plotly_to_div(fig, include_plotly='cdn', **kwargs):
#         div = plotly.offline.plot(fig, output_type='div',
#                                   include_plotlyjs=include_plotly, **kwargs)
#         return div


#     @staticmethod
#     def mpl_to_div(fig, ext='.png', *args, **kwargs):
#         with TemporaryFile(suffix=ext) as tmp:
#             Report.mpl_to_file(fig, tmp, *args, **kwargs)
#             tmp.seek(0)
#             s = base64.b64encode(tmp.read()).decode("utf-8")
#         return f"<div><img src = \"data:image/{format(ext)};base64,{s}\" width=\"100%\"></div>"

#     @staticmethod
#     def mpl_to_file(fig, fname, dpi=100):
#         fig.savefig(fname, dpi=dpi, bbox_inches='tight')


# class MRS_Report(Report):

#     def __init__(self, mrs,res):
#         Report.__init__(self)
#         self.html_template=os.path.join(os.path.dirname(__file__),'mrs_report_template.html')
#         self.mrs = mrs
#         self.res = res

#     def plot_fit(self):
#         fig = plotting.plotly_fit(self.mrs,self.res)
#         return fig

#     def some_mpl_plot(self):
#         fig = plotting.plot_waterfall(self.mrs, ppmlim=(0, 5), proj='real',mod=False)

#         return fig

#     def parse(self, outname: str):
#         super().parse(self.html_template, outname, mrs=self.mrs)


def to_div(fig):
    return plotly.offline.plot(fig,
                               output_type='div',
                               include_plotlyjs='cdn')
def create_plotly_div(mrs,res):
    divs={}

    
    # Summary plot
    fig = plotting.plotly_fit(mrs,res)
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
        fig1 = plotting.plot_corr(res,title='MCMC Correlations')
        fig2 = plotting.plot_dist_mcmc(res,refname=res.concScalings['internalRef'])
    else:
        fig1 = plotting.plot_corr(res,corr=res.corr,title='Laplace approx Correlations')
        fig2 = plotting.plot_dist_approx(res,refname=res.concScalings['internalRef'])    
    divs['corr']       = to_div(fig1)
    divs['posteriors'] = to_div(fig2)
    
    fig = plotting.plot_real_imag(mrs,res,ppmlim=(.2,4.2))
    divs['real-imag'] = to_div(fig) #plotly.offline.plot(fig, output_type='div',include_plotlyjs='cdn')
    
    fig = plotting.plot_indiv_stacked(mrs,res,ppmlim=res.ppmlim)
    divs['indiv'] = to_div(fig) #plotly.offline.plot(fig, output_type='div',include_plotlyjs='cdn')
    
    # Quantification table
    if (res.concScalings['molality'] is not None) and hasattr(res.concScalings['info']['q_info'],'f_GM'):
        Q = res.concScalings['info']['q_info']
        quant_df = pd.DataFrame()
        quant_df['Tissue-water densities'] = [Q.d_GM,Q.d_WM,Q.d_CSF]
        quant_df['Tissue volume fractions'] = [Q.f_GM,Q.f_WM,Q.f_CSF]
        quant_df['Water T2s'] = [Q.T2['H2O_GM'],Q.T2['H2O_WM'],Q.T2['H2O_CSF']]
        quant_df.index = ['GM', 'WM', 'CSF']
        quant_df.index.name = 'Tissue'
        quant_df.reset_index(inplace=True)
        tab = plotting.create_table(quant_df)
        fig = go.Figure(data=[tab])
        fig.update_layout(height=100,margin=dict(l=0,r=0,t=0,b=0))
        divs['quant_table'] = to_div(fig)

    return divs


def static_image(imgfile):
    import plotly.graph_objects as go
    from PIL import Image
    img = Image.open(imgfile)
    # Create figure
    fig = go.Figure()

    # Constants
    img_width  = img.width
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

    # Disable the autosize on double click because it adds unwanted margins around the image
    # More detail: https://plotly.com/python/configuration-options/
    #fig.show(config={'doubleClick': 'reset'})

    return fig

# def create_vox_plot(t1file,voxfile,outdir):
#     def ijk2xyz(ijk,affine):    
#         return affine[:3, :3].dot(ijk) + affine[:3, 3]
#     def xyz2ijk(xyz,affine):
#         inv_affine = np.linalg.inv(affine)
#         return inv_affine[:3, :3].dot(xyz) + inv_affine[:3, 3]
#     def do_plot(slice,rect):
#         vmin = np.quantile(slice,.01)
#         vmax = np.quantile(slice,.99)
#         plt.imshow(slice, cmap="gray", origin="lower",vmin=vmin,vmax=vmax)
#         plt.plot(rect[:, 1], rect[:, 0],c='#FF4646',linewidth=2)
#         plt.xticks([])
#         plt.yticks([])

#     t1      = nib.load(t1file)
#     vox     = nib.load(voxfile)
#     t1_data = t1.get_fdata()

#     # centre of MRS voxel in T1 voxel space (or is it the corner?)
#     ijk   = xyz2ijk(ijk2xyz([0,0,0],vox.affine),t1.affine)
#     i,j,k = ijk
#     # half size of MRS voxel (careful this assumes 1mm resolution T1)
#     si,sj,sk = np.diag(vox.affine[:3,:3])/2
#     # Do the plotting
#     fig = plt.figure(figsize=(15,10))
#     plt.subplot(1,3,1)
#     slice = np.squeeze(t1_data[int(i),:,:])
#     rect  = np.asarray([[j-sj,k-sk],[j+sj,k-sk],[j+sj,k+sk],[j-sj,k+sk],[j-sj,k-sk]])
#     do_plot(slice,rect)
#     plt.subplot(1,3,2)
#     slice = np.squeeze(t1_data[:,int(j),:])
#     rect  = np.asarray([[i-si,k-sk],[i+si,k-sk],[i+si,k+sk],[i-si,k+sk],[i-si,k-sk]])
#     do_plot(slice,rect)
#     plt.subplot(1,3,3)
#     slice = np.squeeze(t1_data[:,:,int(k)])
#     rect  = np.asarray([[i-si,j-sj],[i+si,j-sj],[i+si,j+sj],[i-si,j+sj],[i-si,j-sj]])
#     do_plot(slice,rect)
#     fig.save(os.path.join(outdir,'voxplot.png'))
#     return

def create_report(mrs,res,filename,fidfile,basisfile,h2ofile,date,location_fig = None):

    divs= create_plotly_div(mrs,res)

    
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
    section=f"""
    <h1><a name="summary">Summary</a></h1>    
    <div id=fit>{divs['summary']}</div>
    <hr>
    """
    template+=section

    # Location
    if location_fig is not None:
        fig = static_image(location_fig)
        template+=f"<h1>Voxel location</h1><div>{to_div(fig)}</div><hr>"
    
    # Tables section
    section=f"""
    <h1><a name="nuisance">Nuisance parameters</a></h1>    
    <div style="width:70%">{divs['table-lineshape-phase']}</div>
    
    <hr>
    <h1><a name="qc">QC parameters</a></h1>
    <div style="width:70%">{divs['table-qc']}</div>
    <hr>
    """
    template+=section
    
    # Dist section
    section=f"""
    <h1><a name="uncertainty">Uncertainties</a></h1>
    <table width=100%>
    <tr>
    <th style="vertical-align:top">{divs['corr']}</th>
    <th style="vertical-align:top">{divs['posteriors']}</th>
    </tr>
    </table>
    <hr>
    """
    template+=section

    # Real/imag section
    section=f"""
    <h1><a name="realimag">Fitting summary (real/imag)</a></h1>
    {divs['real-imag']}
    <hr>
    """
    template+=section
    
    # Indiv spectra section
    section=f"""
    <h1><a name="metabs">Individual metabolite spectra</a></h1>
    {divs['indiv']}
    <hr>
    """
    template+=section

    # Quantification information
    # Table of CSF,GM,WM
        # Fractions
        # T2 info (water)
        # T2 info (metab)
        # Tissue water densities
    # Relaxation corrected water
    # Relaxation correction for metab
    # Final scalings for molality & molarity
    if res.concScalings['molality'] is not None:
        Q = res.concScalings['info']['q_info']
        relax_water_conc = Q.relaxationCorrWater_molar
        metabRelaxCorr = Q.relaxationCorrMetab

        if  hasattr(res.concScalings['info']['q_info'],'f_GM'):
            section=f"""
            <h1><a name="quantification">Quantification information</a></h1>
            {divs['quant_table']}            
            <p>The metabolite T<sub>2</sub> is {1000*Q.T2['METAB']} ms.<br>
            The sequence echo time (T<sub>E</sub>) is {1000*Q.TE} ms.<br>
            <p>The T<sub>2</sub> relaxation corrected water concentration is {relax_water_conc:0.0f} mmol/kg.<br>
            The metabolite relaxation correction (1/e<sup>(-T<sub>E</sub>/T<sub>2</sub>)</sup>) is {metabRelaxCorr:0.2f}.<br>
            <p>The final raw concentration to molarity and molality scalings are {res.concScalings["molarity"]:0.2f} and {res.concScalings["molality"]:0.2f}.
            <hr>
            """
        else:
            section=f"""
            <h1><a name="quantification">Quantification information</a></h1>
            <p>The water T<sub>2</sub> is {1000*Q.T2['H2O']} ms.<br>
            The metabolite T<sub>2</sub> is {1000*Q.T2['METAB']} ms.<br>
            The sequence echo time (T<sub>E</sub>) is {1000*Q.TE} ms.<br>
            <p>The T<sub>2</sub> relaxation corrected water concentration is {relax_water_conc:0.0f} mmol/kg.<br>
            The metabolite relaxation correction (1/e<sup>(-T<sub>E</sub>/T<sub>2</sub>)</sup>) is {metabRelaxCorr:0.2f}.<br>
            <p>The final molarity and molality scalings are {res.concScalings["molarity"]:0.2f} and {res.concScalings["molality"]:0.2f}.
            <hr>
            """
        template+=section

    # Details of the methods
    #methodsfile="/path/to/file"
    #methods = np.readtxt(methodsfile)
    from fsl_mrs import __version__
    methods=f"<p>Fitting of the SVS data was performed using a Linear Combination model as described in [1] and as implemented in FSL-MRS version {__version__}, part of FSL (FMRIB's Software Library, www.fmrib.ox.ac.uk/fsl). Briefly, basis spectra are fitted to the complex FID in the frequency domain. The basis spectra are shifted and broadened with parameters fitted to the data and grouped into {max(res.metab_groups)+1} metabolite groups. A complex polynomial baseline is also concurrrently fitted (order={res.baseline_order}). Model fitting was performed using the {res.method} algorithm.<p><h3>References</h3><p>[1] Clarke WT, Jbabdi S. FSL-MRS: An end-to-end spectroscopy analysis package (2020)."
    section=f"""
    <h1><a name="methods">Analysis methods</a></h1>
    <div>{methods}</div>
    <hr>
    """
    template+=section

    # End of report
    template+="""
    </body>    
    </html>
    """
    
    # write 
    with open(filename, 'w') as f:
        f.write(template)

    


def fitting_summary_fig(mrs,res,filename):
    fig = plotting.plot_fit(mrs,pred=res.pred,baseline=res.baseline)
    fig.savefig(filename)




# --------- MRSI reporting
def save_params(params,names,data_hdr,mask,folder,cleanup=True):
    for i,p in enumerate(names):
        x = misc.list_to_volume(list(params[:,i]),mask)
        if cleanup:
            x[np.isnan(x)] = 0
            x[np.isinf(x)] = 0
            x[x<1e-10]     = 0
            x[x>1e10]      = 0
        
        img      = nib.Nifti1Image(x,data_hdr.affine)
        filename = os.path.join(folder,p+'.nii.gz')
        nib.save(img, filename)
