import re
import requests
import time
from io import StringIO

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm


base_url = "https://www.basketball-reference.com/" 
img_path = "nbahalfcourt.png"


def process_response(response):
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    r = re.compile(r"^shots-")
    shot_charts = soup.find_all("div", {"id": r})

    r = re.compile(r"^tooltip")
    points = shot_charts[0].find_all("div", {"class": r})

    made_xs = []
    made_ys = []
    missed_xs = []
    missed_ys = []
    for point in points:
        style = point["style"]
        x_px, y_px = style.split(";")[1], style.split(";")[0]
        x = x_px.split(":")[-1].strip("px")
        y = y_px.split(":")[-1].strip("px")

        if "miss" in point["class"]:
            missed_xs.append(int(x))
            missed_ys.append(int(y))
        else:
            made_xs.append(int(x))
            made_ys.append(int(y))

    return missed_xs, missed_ys, made_xs, made_ys


def parse_matches(response):

    html = response.text
    soup = BeautifulSoup(re.sub("<!--|-->", "", html), "html.parser")

    table = soup.find("table")
    rows = table.find_all("tr")
    i = 0

    for row in rows:
        is_row = bool(row.find_all("th", {"scope": "row"}))
        if is_row:
            match_link = row.find_all("a")[1]["href"]
            match_id = match_link.split("/")[2].split(".")[0]
            print("Parsing", match_id)

            i += 1
            if i % 30 == 0:
                for _ in tqdm(range(0, 60), desc="Request cooldown (60s)"):
                    time.sleep(1)

            url = f"{base_url}/boxscores/shot-chart/{match_id}.html"

            response = requests.get(url)
            if response.status_code == 200:
                missed_xs, missed_ys, made_xs, made_ys = process_response(response)
                np.savez(f"data/missed_{match_id}", np.array(missed_xs), np.array(missed_ys))
                np.savez(f"data/made_{match_id}", np.array(made_xs), np.array(made_ys))


if __name__ == "__main__":
    URL = "https://www.basketball-reference.com/teams/BOS/2024_games.html"
    response = requests.get(URL)
    if response.status_code == 200:
        parse_matches(response)
