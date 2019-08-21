import plotly
import plotly.graph_objs as go

def line_plot(x, y,
            title=None,
            margin=None,
            ylabel=None,
            xlabel=None,
            width=480,
            height=480,
            xticklabels=True,
            yticklabels=True):

    yaxis = go.layout.YAxis(
        title=ylabel,
        automargin=True,
        mirror='all',
        showline=True,
        showticklabels=yticklabels
    )

    xaxis = go.layout.XAxis(
        title=xlabel,
        automargin=True,
        mirror='all',
        showline=True,
        showticklabels=xticklabels
    )

    layout = go.Layout(
        title=title,
        autosize=True,
        width=width,
        height=height,
        margin=margin,
        yaxis=yaxis,
        xaxis=xaxis
    )

    div = plotly.offline.plot({
        "data": [go.Scatter(x=x, y=y)],
        "layout": layout,
    }, output_type='div', include_plotlyjs='cdn')

    return div

