import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def get_minmax(series):
    series_out = (series-series.min())/(series.max()-series.min())
    return series_out

def plot_one_col(df_in, col, thetitle, anno_col=None, scale=True, smooth=False, smooth_win="7D", show_post_lock=True, error_cols=None, ylims=None): 
    df = df_in.copy(deep=True)
    if smooth:
#         print ("Smoothing!")
        df.index = pd.to_datetime(df.index)
        df[col] = df[col].rolling(smooth_win, min_periods=1).mean()
        if error_cols is not None:
            df[error_cols[0]] = df[error_cols[0]].rolling(smooth_win, min_periods=1).mean()
            df[error_cols[1]] = df[error_cols[1]].rolling(smooth_win, min_periods=1).mean()
        df.index = df.index.date
    if scale:
        vals = get_minmax(df[col])
    else:
        vals = df[col]
    if anno_col is not None:
        anno_vals = df[anno_col]
    else:
        anno_vals = None
        
    xvals = df.index.values
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name=col,
        mode="markers+lines", x=xvals, y=vals,
        text=anno_vals,
        showlegend=False
    ))
    if error_cols is not None:
        y_lower = df[error_cols[0]].values
        y_upper = df[error_cols[1]].values
        fig.add_trace(
            go.Scatter(
                name='Upper Bound',
                x=xvals,
                y=y_upper,
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            )
        )
        fig.add_trace(
            go.Scatter(
                name='Lower Bound',
                x=xvals,
                y=y_lower,
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
        )
    fig.update_layout(title=thetitle)
    fig.update_xaxes(title="datetime")
    if ylims is None:
        fig.update_yaxes(title=col)
    else:
        fig.update_yaxes(title=col, range=ylims)
    fig.add_vline(x='2020-03-12', line_color="red") # covid PA stay at home
    fig.add_vline(x='2020-05-15', line_color="yellow") # allegheny enters yellow phase
    fig.add_vline(x='2020-05-26') # protests start
    fig.add_vline(x='2020-06-05', line_color="green") # allegheny enters green phase
    if show_post_lock:
        fig.add_vline(x='2020-11-03') # election
        fig.add_vline(x='2020-11-07') # election results
        fig.add_vline(x='2020-12-24')
        fig.add_vline(x='2020-12-31')
    fig.show()
    
def plot_one_col_trace(fig, df_in, col, thetitle, anno_col=None, scale=True, smooth=False, smooth_win="7D"): 
    df = df_in.copy(deep=True)
    if smooth:
#         print ("Smoothing!")
        df.index = pd.to_datetime(df.index)
        df[col] = df[col].rolling(smooth_win, min_periods=1).mean()
        df.index = df.index.date
    if scale:
        vals = get_minmax(df[col])
    else:
        vals = df[col]
    if anno_col is not None:
        anno_vals = df[anno_col]
    else:
        anno_vals = None
        
    fig.add_trace(go.Scatter(
        mode="lines", x=df.index, y=vals,
        text=anno_vals,
        name=df["device_id"].iloc[0]
    ))
    fig.update_layout(title=thetitle)
    fig.update_xaxes(title="datetime")
    fig.update_yaxes(title=col)
    fig.add_vline(x='2020-03-12') # covid PA stay at home
    fig.add_vline(x='2020-05-15') # allegheny enters yellow phase
    fig.add_vline(x='2020-05-26') # protests start
    fig.add_vline(x='2020-06-05') # allegheny enters green phase
    fig.add_vline(x='2020-11-03') # election
    fig.add_vline(x='2020-11-07') # election results
    fig.add_vline(x='2020-12-24')
    fig.add_vline(x='2020-12-31')
    return fig


def plot_one_col_multiple_dfs(df_in_list, df_lbls_list, col, thetitle, anno_col=None, scale=True, smooth=False, smooth_win="7D", show_post_lock=True, error_cols=None, ylims=None): 
    fig = go.Figure()
    df_cnt_i = -1
    for df_in in df_in_list:
        df_cnt_i += 1
        df = df_in.copy(deep=True)
        df_lbl = df_lbls_list[df_cnt_i]
        if smooth:
    #         print ("Smoothing!")
            df.index = pd.to_datetime(df.index)
            df[col] = df[col].rolling(smooth_win, min_periods=1).mean()
            if error_cols is not None:
                df[error_cols[0]] = df[error_cols[0]].rolling(smooth_win, min_periods=1).mean()
                df[error_cols[1]] = df[error_cols[1]].rolling(smooth_win, min_periods=1).mean()
            df.index = df.index.date
        if scale:
            vals = get_minmax(df[col])
        else:
            vals = df[col]
        if anno_col is not None:
            anno_vals = df[anno_col]
        else:
            anno_vals = None

        xvals = df.index.values
        fig.add_trace(go.Scatter(
            name=col+" "+df_lbl,
            mode="markers+lines", x=xvals, y=vals,
            text=anno_vals,
            showlegend=True
        ))
        if error_cols is not None:
            y_lower = df[error_cols[0]].values
            y_upper = df[error_cols[1]].values
            fig.add_trace(
                go.Scatter(
                    name='Upper Bound',
                    x=xvals,
                    y=y_upper,
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                )
            )
            fig.add_trace(
                go.Scatter(
                    name='Lower Bound',
                    x=xvals,
                    y=y_lower,
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=False
                )
            )
    fig.update_layout(title=thetitle)
    fig.update_xaxes(title="datetime")
    if ylims is None:
        fig.update_yaxes(title=col)
    else:
        fig.update_yaxes(title=col, range=ylims)
    fig.add_vline(x='2020-03-12', line_color="red") # covid PA stay at home
    fig.add_vline(x='2020-05-15', line_color="yellow") # allegheny enters yellow phase
    fig.add_vline(x='2020-05-26') # protests start
    fig.add_vline(x='2020-06-05', line_color="green") # allegheny enters green phase
    if show_post_lock:
        fig.add_vline(x='2020-11-03') # election
        fig.add_vline(x='2020-11-07') # election results
        fig.add_vline(x='2020-12-24')
        fig.add_vline(x='2020-12-31')
    fig.show()
