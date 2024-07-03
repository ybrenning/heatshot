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
HALFCOURT_LEN = 47

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
        go.Heatmap(
            z=np.sqrt(Z),
            opacity=0.9,
            colorbar=dict(
                title="Square Root of Kernel Density Estimate",
                x=1,
                xanchor="left")
        )
    )

    fig.update_traces(
        colorbar_title_side="right",
        colorscale="Viridis"
    )

    fig.update_layout(xaxis_range=[0, 200])
    fig.update_layout(yaxis_range=[0, 200])

    fig.update_layout(
        width=W,
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
                sizing="stretch",
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
                    style={
                        "margin-top": "75px",
                        "margin-left": "40px",
                        'padding': '20px'
                    }
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
                        "margin-left": "60px",
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
                        "margin-left": "75px",
                        "width": "400px",
                        "font-size": "small",
                    }
                ),

                dcc.Graph(
                    figure=go.Figure(),
                    id="shot-dists",
                    style={
                        "flex-grow": "1",
                        "max-height": "250px",
                        "min-width": "400px"
                    }
                ),

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


def plot_dists(dropdown, category, stat="made"):

    data_made = np.load(f"data/{dropdown}/dists.npz")
    data_missed = np.load(f"data/{dropdown}/dists_missed.npz")

    xs_made, ys_made = data_made["arr_1"], data_made["arr_0"]
    xs_missed, ys_missed = data_missed["arr_1"], data_missed["arr_0"]

    if stat == "fgp":
        made_shape = xs_made.shape[0]
        missed_shape = xs_missed.shape[0]

        pad_made_len = HALFCOURT_LEN - made_shape
        pad_missed_len = HALFCOURT_LEN - missed_shape

        for i in range(0, pad_made_len):
            xs_made = np.append(xs_made, max(xs_made)+i)
            ys_made = np.append(ys_made, 0)
        for i in range(0, pad_missed_len):
            xs_missed = np.append(xs_missed, max(xs_made)+i)
            ys_missed = np.append(ys_missed, 0)

        zero_mask = (ys_made == 0) & (ys_missed == 0)
        ys_pct = np.divide(
            ys_made, ys_missed + ys_made,
            out=np.zeros_like(ys_made, dtype=float),
            where=(ys_missed + ys_made) != 0
        )

        ys_pct[zero_mask] = 0
        ys = ys_pct
    elif stat == "made":
        ys = ys_made
    elif stat == "miss":
        ys = ys_missed
    elif stat == "all":
        ys = np.append(ys_made, ys_missed)

    layout = go.Layout(
        margin=dict(t=20),
        autosize=True
    )

    fig = go.Figure(layout=layout)

    # hover_text_made = [f"{y} shots made from {x} ft" for (x, y) in zip(xs, ys)]
    fig.add_trace(
        go.Scatter(
            x=xs_made,
            y=ys,
            mode='lines',
            name='Line Chart',
            # text=hover_text,
            # hovertemplate="%{text}<extra></extra>"
        )
    )

    fig.update_layout(
        xaxis_title="Shot distance (ft)",
        yaxis_title="No. of made shots"
    )

    annotation_y = 0.5 if category == "Player" else 300
    fig.add_vline(x=22, line_width=3, line_dash="dash", line_color="green")
    fig.add_annotation(
        x=22,
        y=annotation_y,
        text="3PT Line",
        showarrow=False,
        yshift=10,
    )

    fig.update_layout(hovermode="x")
    fig.update_layout(xaxis_range=[0, 40])

    return fig


@app.callback(
    Output("shot-dists", "figure"),
    Output("shot-dists", "style"),
    Input("category", "value"),
    Input("dropdown", "value"),
)
def create_dist_graph(category, dropdown):
    return plot_dists(dropdown, category), {"display": "block"}


if __name__ == '__main__':
    app.run(debug=True)
