import numpy as np
import plotly.graph_objects as go
from sklearn.neighbors import KernelDensity

W = 500*1.2
H = 472*1.2
halfcourt = 'http://cdn.ssref.net/req/1/images/bbr/nbahalfcourt.png'

fig = go.Figure()

path = "data/MIL"
import os

x = np.array([])
y = np.array([])
with os.scandir(path) as it:
    for entry in it:
        if entry.name.endswith(".npz") and entry.is_file():
            if entry.name.startswith("made"):
                x = np.append(x, np.load(f"data/MIL/{entry.name}")["arr_0"])
                y = np.append(y, np.load(f"data/MIL/{entry.name}")["arr_1"])

data = np.vstack([x, y]).T

kde = KernelDensity(bandwidth=30, kernel='epanechnikov')
kde.fit(data)

xmin, xmax = np.min(x), np.max(x)
ymin, ymax = np.min(y), np.max(y)

xmin, xmax = -5, 490
ymin, ymax = -15, 445

x_grid = np.linspace(xmin, xmax, 200)
y_grid = np.linspace(ymin, ymax, 200)
X, Y = np.meshgrid(x_grid, y_grid)
positions = np.vstack([X.ravel(), Y.ravel()]).T

Z = np.exp(kde.score_samples(positions)).reshape(X.shape)

fig.add_trace(
    go.Heatmap(z=np.sqrt(Z), opacity=0.8)
)

fig.update_layout(xaxis_range=[0, 200])
fig.update_layout(yaxis_range=[0, 200])

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

fig['layout']['yaxis']['autorange'] = "reversed"

fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
fig.show()
