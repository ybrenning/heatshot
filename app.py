import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from sklearn.neighbors import KernelDensity

from player_data import player_data

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
shot_type_dict = {"Made": "made", "Missed": "missed", "Attempted": "all"}

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

players = [
    "curryst01",
    "antetgi01",
    "jamesle01",
    "doncilu01",
    "jokicni01",
    "gilgesh01",
    "embiijo01",
    "duranke01",
    "irvinky01",
    "edwaran01",
    "georgpa01",
    "bookede01",
    "willizi01",
    "tatumja01",
    "brunsja01",
    "butleji01",
    "goberru01",
    "wembavi01",
    "derozde01",
    "youngtr01",
    "hardeja01",
    "thompkl01"
]


players_dict = {
    "curryst01": "Stephen Curry",
    "antetgi01": "Giannis Antetokounmpo",
    "jamesle01": "LeBron James",
    "doncilu01": "Luka Doncic",
    "jokicni01": "Nikola Jokic",
    "gilgesh01": "Shai Gilgeous-Alexander",
    "embiijo01": "Joel Embiid",
    "duranke01": "Kevin Durant",
    "irvinky01": "Kyrie Irving",
    "edwaran01": "Anthony Edwards",
    "georgpa01": "Paul George",
    "bookede01": "Devin Booker",
    "willizi01": "Zion Williamson",
    "tatumja01": "Jayson Tatum",
    "brunsja01": "Jalen Brunson",
    "butleji01": "Jimmy Butler",
    "goberru01": "Rudy Gobert",
    "wembavi01": "Victor Wembanyama",
    "derozde01": "DeMar DeRozan",
    "youngtr01": "Trae Young",
    "hardeja01": "James Harden",
    "thompkl01": "Klay Thompson"
}

teams_dict = {
    "BOS": "Boston Celtics",
    "NYK": "New York Knicks",
    "MIL": "Milwaukee Bucks",
    "CLE": "Cleveland Cavaliers",
    "ORL": "Orlando Magic",
    "IND": "Indiana Pacers",
    "PHI": "Philadelphia 76ers",
    "MIA": "Miami Heat",
    "CHI": "Chicago Bulls",
    "ATL": "Atlanta Hawks",
    "BRK": "Brooklyn Nets",
    "TOR": "Toronto Raptors",
    "CHO": "Charlotte Hornets",
    "WAS": "Washington Wizards",
    "DET": "Detroit Pistons",
    "OKC": "Oklahoma City Thunder",
    "DEN": "Denver Nuggets",
    "MIN": "Minnesota Timberwolves",
    "LAC": "Los Angeles Clippers",
    "DAL": "Dallas Mavericks",
    "PHO": "Phoenix Suns",
    "NOP": "New Orleans Pelicans",
    "LAL": "Los Angeles Lakers",
    "SAC": "Sacremento Kings",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "UTA": "Utah Jazz",
    "MEM": "Memphis Grizzlies",
    "SAS": "San Antonio Spurs",
    "POR": "Portland Trailblazers"
}

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
                if not entry.name.startswith("dists"):
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
        # For some reason it gets cut off
        height=H+10,
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

    def normalize(value, min_value, max_value, new_min, new_max):
        return ((value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min

    xs = x
    ys = y
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    normalized_x = [normalize(x, x_min, x_max, 0, W) for x in xs]
    normalized_y = [normalize(y, y_min, y_max, 0, H) for y in ys]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=normalized_x, y=normalized_y, mode="markers")
    )

    fig.update_layout(xaxis_range=[0, W+10])
    fig.update_layout(yaxis_range=[0, H])

    fig.update_layout(
        width=W-10,
        height=H+20,
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
    fig.update_yaxes(range=[0, H+100], showgrid=False, zeroline=False, showticklabels=False)

    return fig


app.layout = html.Div([
    html.H1(
        children="Visualizing NBA shooting tendencies",
        style={
            "margin-bottom": "0px",
            "margin-top": "20px",
            "margin-left": "20px"
        }
    ),

    html.Div([
        html.Div(
            [
                html.Div(
                    dcc.RadioItems(
                        ["Team", "Player"],
                        "Team",
                        id="category",
                    ),
                    style={"margin-top": "75px", 'padding': '20px'}
                ),

                html.Img(
                    id="img",
                    src="",
                    style={
                        'width': '100px',
                        'height': 'auto',
                        'alignSelf': 'flex-start',
                        'margin': '20px'
                    }
                ),
            ],

            style={
                'display': 'flex',
                'flexDirection': 'column'
            },
        ),

        html.Div(
            [
                html.Div(
                    dcc.RadioItems(
                        ["Made", "Missed", "Attempted"],
                        "Attempted",
                        id="shot-type",
                    ),
                    style={"margin-top": "75px", 'padding': '20px'}
                ),

                html.Div(
                    dcc.Dropdown(
                        id="dropdown",
                        options=[
                            {"label": teams_dict[team], "value": team}
                            for team in teams_east + teams_west
                        ],
                        value="BOS",
                    ),
                    style={
                        "margin-top": "0px",
                        "margin-bottom": "0px",
                        'padding': '10px',
                        "width": "75%"
                    }
                ),

                html.P(
                    id="player-desc",
                    style={
                        "margin-top": "0px",
                        "margin-left": "10px",
                        "width": "400px"
                    }
                )

            ],

            style={
                'display': 'flex',
                'flexDirection': 'column',
            },
        ),

        dcc.Graph(
            figure=go.Figure(),
            id="shot-chart",
            style={"flex": "1", "margin-left": "0px", "margin-right": "100px"}
        ),

    ],
             style={
             "margin-top": "10px",
             "margin-left": "75px",
             'display': 'flex',
             'flexDirection': 'row',
             'justifyContent': 'space-around',
             "height": "600px",
             },
    ),

    dcc.Graph(
        id="shot-dists",
    ),

], style={"font-family": "Verdana"})


@app.callback(
    Output("player-desc", "children"),
    Input("category", "value"),
    Input("dropdown", "value")
)
def update_player_desc(category, dropdown):
    if category == "Player":
        # TODO: Make this dynamic
        attributes = ["Position", "Shoots", "Height", "Weight"]
        description = []

        for attribute in attributes:
            description.append(attribute + ": " + player_data[dropdown][attribute])
            description.append(html.Br())

        return description
    else:
        return ""


@app.callback(
    Output("img", "src"),
    Input("category", "value"),
    Input("dropdown", "value")
)
def update_image(category, dropdown):
    if category == "Player":
        return f"assets/{dropdown}.jpg"
    else:
        return f"assets/{dropdown}.png"


@app.callback(
    Output("dropdown", "options"),
    Output("dropdown", "value"),
    Input("category", "value")
)
def update_dropdown(category):
    if category == "Team":
        options = [
            {"label": teams_dict[team], "value": team}
            for team in teams_east + teams_west
        ]
        value = "BOS"
    elif category == "Player":
        options = [
            {"label": players_dict[player], "value": player}
            for player in players_dict
        ]
        value = "curryst01"

    return options, value


@app.callback(
    Output("shot-chart", "figure"),
    Input("dropdown", "value"),
    Input("shot-type", "value"),
    # Input("shot-chart-type", "value")
)
def plot_heatmap(team, shot_type, chart_type="density"):

    shot_type = shot_type_dict[shot_type]
    return plot_team_shot_chart(
        team,
        chart_type=chart_type,
        shot_type=shot_type
    )


def plot_dists(dropdown):

    data = np.load(f"data/{dropdown}/dists.npz")
    x, y = data["arr_1"], data["arr_0"]

    print(x, y)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Line Chart'))

    fig.update_layout(title='Line Chart Example',
                      xaxis_title='X Axis Title',
                      yaxis_title='Y Axis Title')

    fig.add_vline(x=22, line_width=3, line_dash="dash", line_color="green")

    return fig


@app.callback(
    Output("shot-dists", "figure"),
    Output("shot-dists", "style"),
    Input("category", "value"),
    Input("dropdown", "value"),
)
def create_dist_graph(category, dropdown):
    if category == "Player":
        return plot_dists(dropdown), {"display": "block"}
    return go.Figure(), {"display": "none"}


if __name__ == '__main__':
    app.run(debug=True)
