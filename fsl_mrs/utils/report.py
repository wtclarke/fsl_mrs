import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import jinja2
import pandas as pd
import numpy as np
import os.path as op
import os
from tempfile import TemporaryFile
import base64


import plotly
import plotly.graph_objs as go

import matplotlib.gridspec as gridspec
from plotly import tools

from fsl_mrs.utils import plotting
from fsl_mrs.utils import plot



class Report(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        pass


    def parse(self, template: str, outname: str, **kwargs):

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(op.dirname(template)),
            extensions=['jinja2.ext.do']
        )

        args = dict(
            report=self,
            numpy=np,
            pandas=pd,
            plot=plot,
            **kwargs
        )

        template = env.get_template(op.basename(template))
        html = template.render(**args)

        with open(outname, 'w') as outfile:
            outfile.write(html)



    @staticmethod
    def plotly_to_div(fig, include_plotly='cdn', **kwargs):
        div = plotly.offline.plot(fig, output_type='div', include_plotlyjs=include_plotly, **kwargs)
        return div


    @staticmethod
    def mpl_to_div(fig, ext='.png', *args, **kwargs):
        with TemporaryFile(suffix=ext) as tmp:
            Report.mpl_to_file(fig, tmp, *args, **kwargs)
            tmp.seek(0)
            s = base64.b64encode(tmp.read()).decode("utf-8")
        return f"<div><img src = \"data:image/{format(ext)};base64,{s}\" width=\"100%\"></div>"

    @staticmethod
    def mpl_to_file(fig, fname, dpi=100):
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')


class MRS_Report(Report):

    def __init__(self, mrs):
        Report.__init__(self)
        self.html_template=os.path.join(os.path.dirname(__file__),'mrs_report_template.html')
        self.mrs = mrs

    def plot_fit(self):

        fig = plotting.plot_fit_new(self.mrs)
        
        return fig

    def some_mpl_plot(self):
        fig = plotting.plot_waterfall(self.mrs, ppmlim=(0, 5), proj='real',mod=False)

        return fig

    def parse(self, outname: str):
        super().parse(self.html_template, outname, mrs=self.mrs)


