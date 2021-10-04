# reporting.py - Routines for generating proc reports
#
# Author: William Clarke <william.clarke@ndcn.ox.ac.uk>
#
# Copyright (C) 2019 University of Oxford
# SHBASECOPYRIGHT

from jinja2 import FileSystemLoader, Environment
import os.path as op
from dataclasses import dataclass
from datetime import datetime
templatePath = op.join(op.dirname(__file__), 'templates')


@dataclass
class figgroup:
    '''Keep figure data together'''
    fig: str = ''
    foretext: str = ''
    afttext: str = ''
    name: str = ''


def plotStyles():
    '''Return plot styles'''
    # colors = {'in':'rgb(67,67,67)',
    #            'out':'rgb(0,0,255)',
    #            'emph':'rgb(255,0,0)',
    #            'diff':'rgb(0,255,0)'}
    colors = {'in': 'rgb(166,206,227)',
              'out': 'rgb(31,120,180)',
              'emph': 'rgb(251,154,153)',
              'diff': 'rgb(51,160,44)',
              'spare': 'rgb(178,223,138)',
              'blk': 'rgb(0,0,0)'}
    line_size = {'in': 2,
                 'out': 2,
                 'emph': 2,
                 'diff': 2,
                 'spare': 2,
                 'blk': 2}
    line = {}
    for key in colors:
        line.update({key: {'color': colors[key],
                           'width': line_size[key]}})

    return line, colors, line_size


def plotAxesStyle(fig, ppmlim, title=None):
    if ppmlim is not None:
        fig.layout.xaxis.update(title_text='Chemical shift (ppm)',
                                tick0=2, dtick=.5,
                                range=[ppmlim[1], ppmlim[0]])
    else:
        fig.layout.xaxis.update(title_text='Chemical shift (ppm)',
                                tick0=2, dtick=.5,
                                autorange="reversed")
    fig.layout.yaxis.update(zeroline=True,
                            zerolinewidth=1,
                            zerolinecolor='Gray',
                            showgrid=False,
                            showticklabels=False)
    if title is not None:
        fig.layout.update({'title': title})
    fig.update_layout(template='plotly_white')


def singleReport(outfile, opName, headerinfo, figurelist):
    """
    Entry point for the script.
    Render a template and write it to file.
    :return:
    """
    # Configure Jinja and ready the loader

    env = Environment(
        loader=FileSystemLoader(searchpath=templatePath)
    )

    # Assemble the templates we'll use
    base_template = env.get_template("report.html")
    figure_section_template = env.get_template("figure_section.html")
    hdr_section_template = env.get_template("header_section.html")
    op_section_template = env.get_template("operation_section.html")

    # Content to be published
    datestr = datetime.now().strftime("%Y-%m-%d %H:%M")
    title = f"Report for {opName} - {datestr}"

    # Construct figures
    figsections = []
    for figs in figurelist:
        figsections.append(figure_section_template.render(
                           figName=figs.name,
                           foretext=figs.foretext,
                           figure=figs.fig,
                           afttext=figs.afttext))

    hdr = hdr_section_template.render(hdrName=opName + '_hdr',
                                      hdrText=headerinfo)

    sections = []
    sections.append(op_section_template.render(
        opName=opName,
        header_sec=hdr,
        fig_sections=figsections
    ))

    with open(outfile, "w") as f:
        f.write(base_template.render(
                title=title,
                sections=sections))
