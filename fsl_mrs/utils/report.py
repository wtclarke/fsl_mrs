import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import jinja2
import pandas as pd
import numpy as np
import os.path as op
from tempfile import TemporaryFile
import base64

class Report(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.properties = OrderedDict()

    def add(self, **kwargs):
        for k, v in kwargs:
            self.properties[k] = v


    def parse(self, template: str, outname: str, **kwargs):

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(op.dirname(template)),
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            extensions=['jinja2.ext.do']
        )

        args = dict(
            report=self,
            numpy=np,
            pandas=pd,
            **kwargs
        )

        template = env.get_template(op.basename(template))
        html = template.render(**args)

        with open(outname, 'w') as outfile:
            outfile.write(html)

    @staticmethod
    def to_base64(fig, ext='.svg', *args, **kwargs):
        with TemporaryFile(suffix=ext) as tmp:
            Report.to_file(fig, tmp, *args, **kwargs)
            tmp.seek(0)
            s = base64.b64encode(tmp.read()).decode("utf-8")
        return 'data:image/{};base64,'.format(ext) + s

    @staticmethod
    def to_file(fig, fname, dpi=100):
        fig.savefig(fname, dpi=dpi, bbox_inches='tight')
        # fig.close()
