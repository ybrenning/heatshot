import random
import re
import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw


# TODO: Aggregate over multiple matches
URL = "https://www.basketball-reference.com/boxscores/shot-chart/202404010CHO.html"
img_path = "nbahalfcourt.png"
output_path = "new.png"


# TODO: Draw to JavaScript app
def draw_dot(image, x, y, color):
    x = int(x)
    y = int(y)
    draw = ImageDraw.Draw(image)

    height = 10
    width = 10

    draw.ellipse((x, y, x+height, y+width), fill=color)

    return image


def process_response(response):
    import numpy as np

    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    r = re.compile(r"^shots-")
    shot_charts = soup.find_all("div", {"id": r})

    r = re.compile(r"^tooltip")
    points = shot_charts[0].find_all("div", {"class": r})

    image = Image.open(img_path)

    xs = []
    ys = []
    for point in points:
        if "miss" in point["class"]:
            color = "red"
        else:
            color = "green"

        style = point["style"]
        x_px, y_px = style.split(";")[1], style.split(";")[0]
        x = x_px.split(":")[-1].strip("px")
        y = y_px.split(":")[-1].strip("px")

        xs.append(int(x))
        ys.append(int(y))
        image = draw_dot(image, x, y, color=color)
 
    image.save(output_path)
    print(xs)
    print(ys)
    np.save("xs.npy", np.array(xs))
    np.save("ys.npy", np.array(ys))


if __name__ == "__main__":
    response = requests.get(URL)
    if response.status_code == 200:
        process_response(response)
    # line(IMG, "new.png")
