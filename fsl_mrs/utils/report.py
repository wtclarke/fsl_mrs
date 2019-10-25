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



def create_plotly_div(mrs,res):
    divs=[]

    def to_div(fig):
        return plotly.offline.plot(fig, output_type='div',include_plotlyjs='cdn')
    
    # Summary plot
    fig = plotting.plotly_fit(mrs,res)
    divs.append(to_div(fig))

    fig = plotting.plot_table_qc(mrs,res)
    divs.append(to_div(fig))        
    fig = plotting.plot_table_extras(mrs,res)
    divs.append(to_div(fig))        
    
    # Approximate (Laplace) marginals
    fig = plotting.plot_dist_approx(mrs,res)    
    div2 = plotly.offline.plot(fig, output_type='div',include_plotlyjs='cdn')
    divs.append(div2)

    # MCMC results (if available)
    if res.mcmc_cor is not None:
        fig = plotting.plot_mcmc_corr(mrs,res)
        div3 = plotly.offline.plot(fig, output_type='div',include_plotlyjs='cdn')
        divs.append(div3)
        fig = plotting.plot_dist_mcmc(mrs,res)
        div4 = plotly.offline.plot(fig, output_type='div',include_plotlyjs='cdn')
        divs.append(div4)

    fig = plotting.plot_real_imag(mrs,res,ppmlim=(.2,4.2))
    div = plotly.offline.plot(fig, output_type='div',include_plotlyjs='cdn')
    divs.append(div)
    
    fig = plotting.plot_indiv(mrs,res)
    div = plotly.offline.plot(fig, output_type='div',include_plotlyjs='cdn')
    divs.append(div)
    
    return divs

def create_report(mrs,res,filename,fidfile,basisfile,h2ofile,outdir,date):

    divlist= create_plotly_div(mrs,res)


    template = """<!DOCTYPE html>
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
    </style>       
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
    <h1>FSL MRS Report</h1>
    <pre>
    Date        : {}
    FID         : {}
    Basis       : {}
    H2O         : {}
    </pre>
    <hr>  
    """.format(date,fidfile,basisfile,h2ofile)
    for i,div in enumerate(divlist):
        section="""
        <div id='divPlotly{}'>
        <script>
        var plotly_data{} = {}
        </script>
        </div>
        """
        template+=section.format(i,i,div)
    template+="""
    </body>    
    </html>
    """
    
    # write 
    with open(filename, 'w') as f:
        f.write(template)

    
