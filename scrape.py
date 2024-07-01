import os
import re
import requests
import time

import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm


base_url = "https://www.basketball-reference.com/"
img_path = "nbahalfcourt.png"

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


def process_response(response, category):
    if category == "match":
        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        r = re.compile(r"^shots-")
        shot_charts = soup.find_all("div", {"id": r})

        r = re.compile(r"^tooltip")
        points = shot_charts[0].find_all("div", {"class": r})
    elif category == "player":
        html = response.text
        soup = BeautifulSoup(re.sub("<!--|-->", "", html), "html.parser")

        shot_chart = soup.find_all("div", {"class": "shot-area"})[0]

        r = re.compile(r"^tooltip")
        points = shot_chart.find_all("div", {"class": r})
    else:
        raise ValueError(f"{category} is not a valid category")

    made_x = []
    made_y = []
    missed_x = []
    missed_y = []
    for point in points:
        style = point["style"]
        x_px, y_px = style.split(";")[1], style.split(";")[0]
        x = x_px.split(":")[-1].strip("px")
        y = y_px.split(":")[-1].strip("px")

        if "miss" in point["class"]:
            missed_x.append(int(x))
            missed_y.append(int(y))
        else:
            made_x.append(int(x))
            made_y.append(int(y))

    return (
        np.array(missed_x),
        np.array(missed_y),
        np.array(made_x),
        np.array(made_y)
    )


def parse_matches(response, team):

    newpath = f"data/{team}"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

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
                missed_x, missed_y, made_x, made_y = process_response(
                    response=response,
                    category="match"
                )
                np.savez(f"data/{team}/missed_{match_id}", missed_x, missed_y)
                np.savez(f"data/{team}/made_{match_id}", made_x, made_y)


def parse_teams(season):

    for team in teams_east + teams_west:
        url = f"{base_url}/teams/{team}/{season}_games.html"
        response = requests.get(url)
        if response.status_code == 200:
            parse_matches(response, team)


def parse_players(season):

    i = 0
    for player in players:
        newpath = f"data/{player}"
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        i += 1
        if i % 30 == 0:
            for _ in tqdm(range(0, 60), desc="Request cooldown (60s)"):
                time.sleep(1)

        url = f"{base_url}/players/{player[0]}/{player}/shooting/{season}"

        response = requests.get(url)
        if response.status_code == 200:
            missed_x, missed_y, made_x, made_y = process_response(
                response=response,
                category="player"
            )
            np.savez(f"data/{player}/missed", missed_x, missed_y)
            np.savez(f"data/{player}/made", made_x, made_y)

            assert False
        print(url)


if __name__ == "__main__":
    # parse_teams("2024")
    parse_players("2024")
