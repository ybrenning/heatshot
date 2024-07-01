import os
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from sklearn.neighbors import KernelDensity

app = Dash(__name__)

W = 500*1.2
H = 472*1.2
# halfcourt = 'http://cdn.ssref.net/req/1/images/bbr/nbahalfcourt.png'
halfcourt = 'nbahalfcourt.png'
import plotly.io as pio
from PIL import Image
import base64
from io import BytesIO

# Load the image
image = Image.open(halfcourt)
buffered = BytesIO()
image.save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()

types = ["made", "missed", "all"]

teams_east = [
    "BOS",
    "NYK",
    "MIL",
    "CLE",
    "ORL",
    "IND",
    "PHI",
    "MIA",
    "CHI",
    "ATL",
    "BRK",
    "TOR",
    "CHO",
    "WAS",
    "DET"
]

teams_west = [
    "OKC",
    "DEN",
    "MIN",
    "LAC",
    "DAL",
    "PHO",
    "NOP",
    "LAL",
    "SAC",
    "GSW",
    "HOU",
    "UTA",
    "MEM",
    "SAS",
    "POR"
]

chart_types = ["points", "density"]


def plot_team_shot_chart(team, chart_type, shot_type):
    if shot_type not in types:
        raise ValueError(
            f"{shot_type} is not a valid input. Possible choices: {types}"
        )

    if chart_type.lower() == "density":
        return create_heatmap(team, shot_type)
    elif chart_type.lower() == "points":
        return create_scatter(team, shot_type)
    else:
        raise ValueError(
            f"{chart_type} is not a valid input. "
            f"Possible choices: {chart_types}"
        )


def create_heatmap(team, shot_type):
    path = f"data/{team}"

    x = np.array([])
    y = np.array([])
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".npz") and entry.is_file():
                if entry.name.startswith(shot_type) or shot_type == "all":
                    data = np.load(f"{path}/{entry.name}")
                    x = np.append(x, data["arr_0"])
                    y = np.append(y, data["arr_1"])

    data = np.vstack([x, y]).T

    kde = KernelDensity(bandwidth=30, kernel='epanechnikov')
    kde.fit(data)

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    xmin, xmax = -10, 485
    ymin, ymax = -15, 440

    x_grid = np.linspace(xmin, xmax, 200)
    y_grid = np.linspace(ymin, ymax, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()]).T

    Z = np.exp(kde.score_samples(positions)).reshape(X.shape)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(z=np.sqrt(Z), opacity=0.9)
    )

    fig.update_layout(xaxis_range=[0, 200])
    fig.update_layout(yaxis_range=[0, 200])

    fig.update_layout(
        width=W,
        height=H,
        images=[
            dict(
                source='data:image/png;base64,{}'.format(img_str),
                # source=halfcourt,
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

    fig['layout']['yaxis']['autorange'] = "reversed"

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    return fig


def create_scatter(team, shot_type):
    path = f"data/{team}"

    x = np.array([])
    y = np.array([])
    with os.scandir(path) as it:
        for entry in it:
            if entry.name.endswith(".npz") and entry.is_file():
                if entry.name.startswith(shot_type) or shot_type == "all":
                    data = np.load(f"{path}/{entry.name}")
                    x = np.append(x, data["arr_0"])
                    y = np.append(y, data["arr_1"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers")
    )

    # fig.update_layout(xaxis_range=[0, 200])

    fig.update_layout(
        width=W,
        height=H,
        images=[
            dict(
                # source=halfcourt,
                source='data:image/png;base64,{}'.format(img_str),
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

    fig['layout']['yaxis']['autorange'] = "reversed"

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    return fig


app.layout = html.Div([
    html.H1(children='Title of Dash App', style={'textAlign': 'center'}),
    dcc.Dropdown(
        id="team-dropdown",
        options=teams_east + teams_west + ["curryst01"],
        value="BOS"
    ),
    dcc.RadioItems(
        ["Density", "Points"],
        "Density",
        id="shot-chart-type"
    ),
    dcc.Graph(figure=go.Figure(), id="shot-chart")
])


@app.callback(
    Output("shot-chart", "figure"),
    Input("team-dropdown", "value"),
    Input("shot-chart-type", "value")
)
def plot_heatmap(team, chart_type):
    # TODO: Make shot_type part of the inputs
    return plot_team_shot_chart(team, chart_type=chart_type, shot_type="all")


if __name__ == '__main__':
    app.run(debug=True)
