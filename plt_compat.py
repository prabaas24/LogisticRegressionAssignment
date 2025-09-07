
"""
plt_compat.py

A small compatibility wrapper that exposes a subset of the matplotlib.pyplot API
using Plotly (plotly.graph_objects and plotly.express) under the hood.

It supports:
- figure()
- plot(x, y, *args, **kwargs)
- scatter(x, y, *args, **kwargs)
- hist(data, bins=..., *args, **kwargs)
- bar(x, height, *args, **kwargs)
- imshow(image, *args, **kwargs)
- xlabel(), ylabel(), title()
- legend()
- show()
- savefig(fname)

Notes:
- savefig will try to write a static image using write_image if kaleido or orca is available.
  If not available, it will write an HTML file as a fallback.
- This is a pragmatic, limited implementation intended to reproduce typical notebook visuals.
"""
from typing import Any
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import inspect
import os

_current_fig = None

def figure(figsize=None):
    global _current_fig
    _current_fig = go.Figure()
    if figsize:
        # figsize is (w,h) in inches (matplotlib). We'll map roughly to pixels.
        try:
            w, h = figsize
            _current_fig.update_layout(width=int(w*100), height=int(h*100))
        except Exception:
            pass
    return _current_fig

def _ensure_fig():
    global _current_fig
    if _current_fig is None:
        figure()
    return _current_fig

def plot(*args, **kwargs):
    """
    Common usages:
    plot(y) or plot(x, y)
    """
    fig = _ensure_fig()
    if len(args) == 1:
        y = args[0]
        x = np.arange(len(y))
    else:
        x, y = args[0], args[1]
    # convert to lists
    try:
        x = list(x)
    except Exception:
        x = list(np.asarray(x).tolist())
    try:
        y = list(y)
    except Exception:
        y = list(np.asarray(y).tolist())
    mode = kwargs.pop('marker', None)
    if mode is None:
        mode = 'lines'
    else:
        mode = 'lines+markers'
    name = kwargs.pop('label', None)
    fig.add_trace(go.Scatter(x=x, y=y, mode=mode, name=name))
    return fig

def scatter(x, y, *args, **kwargs):
    fig = _ensure_fig()
    name = kwargs.pop('label', None)
    fig.add_trace(go.Scatter(x=list(x), y=list(y), mode='markers', name=name))
    return fig

def hist(data, bins=None, *args, **kwargs):
    fig = _ensure_fig()
    nbins = None
    if bins is not None:
        try:
            # if bins is integer
            nbins = int(bins)
        except Exception:
            nbins = None
    fig.add_trace(go.Histogram(x=list(data), nbinsx=nbins, name=kwargs.get('label', None)))
    return fig

def bar(x, height, *args, **kwargs):
    fig = _ensure_fig()
    name = kwargs.pop('label', None)
    fig.add_trace(go.Bar(x=list(x), y=list(height), name=name))
    return fig

def imshow(img, *args, **kwargs):
    global _current_fig
    # use px.imshow for images
    try:
        fig = px.imshow(img, binary_string=False)
        _current_fig = fig
        return fig
    except Exception:
        # fallback: convert to numpy array and use go.Heatmap
        import numpy as np
        arr = np.asarray(img)
        fig = go.Figure(go.Heatmap(z=arr))
        _current_fig = fig
        return fig

def xlabel(label):
    fig = _ensure_fig()
    fig.update_layout(xaxis_title=str(label))

def ylabel(label):
    fig = _ensure_fig()
    fig.update_layout(yaxis_title=str(label))

def title(label):
    fig = _ensure_fig()
    fig.update_layout(title=str(label))

def legend(*args, **kwargs):
    fig = _ensure_fig()
    fig.update_layout(showlegend=True)

def show(renderer=None):
    fig = _ensure_fig()
    # If running in a Jupyter environment, plotly will display inline.
    # Prefer default renderer. If a specific renderer requested, set it temporarily.
    if renderer:
        prev = None
        try:
            import plotly.io as pio
            prev = pio.renderers.default
            pio.renderers.default = renderer
            fig.show()
            if prev is not None:
                pio.renderers.default = prev
        except Exception:
            fig.show()
    else:
        fig.show()

def savefig(fname, *args, **kwargs):
    fig = _ensure_fig()
    # try to write a static image
    try:
        # prefer vector formats if extension known
        ext = os.path.splitext(fname)[1].lower()
        if ext in ('.png', '.jpg', '.jpeg', '.pdf', '.svg', '.webp'):
            fig.write_image(fname)
        else:
            # fallback to html
            fig.write_html(fname)
    except Exception:
        # fallback: write html
        try:
            fig.write_html(fname + ".html")
        except Exception as e:
            print("savefig failed:", e)
