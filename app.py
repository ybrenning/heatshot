import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html

app = Dash(__name__)

W = 500*1.2
H = 472*1.2
halfcourt = 'http://cdn.ssref.net/req/1/images/bbr/nbahalfcourt.png'


def create_heatmap():
    fig = go.Figure()

    xs = np.load("data/xs.npy")
    ys = np.load("data/ys.npy")

    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode='markers',
            marker_size=7.5,
            marker_color="#ffd700"
        )
    )

    fig.update_layout(xaxis_range=[-5, 495])
    fig.update_layout(yaxis_range=[425, -5])

    fig.update_layout(
        width=W,
        height=H,
        images=[
            dict(
                source=halfcourt,
                xref="paper",
                yref="paper",
                x=0, y=1,
                sizex=1, sizey=1,
                xanchor="left", yanchor="top",
                sizing="fill",
                layer="below"
            )
        ]
    )

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    return fig


app.layout = html.Div([
    html.H1(children='Title of Dash App', style={'textAlign': 'center'}),
    dcc.Graph(figure=create_heatmap(), id='graph-content')
])


if __name__ == '__main__':
    app.run(debug=True)
