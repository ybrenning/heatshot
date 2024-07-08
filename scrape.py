import os
import re
import time

import numpy as np
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from utils import teams_east, teams_west, players

base_url = "https://www.basketball-reference.com/"


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


def parse_matches(response, team, option):
    print("Parsing matches for", team)

    newpath = f"data/{team}"
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    html = response.text
    soup = BeautifulSoup(re.sub("<!--|-->", "", html), "html.parser")

    table = soup.find("table")
    rows = table.find_all("tr")
    i = 0

    dists_made = []
    dists_missed = []
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

            if option == "points":
                url = f"{base_url}/boxscores/shot-chart/{match_id}.html"

                response = requests.get(url)
                if response.status_code == 200:
                    missed_x, missed_y, made_x, made_y = process_response(
                        response=response,
                        category="match"
                    )
                    np.savez(f"data/{team}/missed_{match_id}", missed_x, missed_y)
                    np.savez(f"data/{team}/made_{match_id}", made_x, made_y)
            elif option == "dists":
                url = f"{base_url}/boxscores/shot-chart/{match_id}.html"

                response = requests.get(url)
                if response.status_code == 200:
                    current_dists_made, current_dists_missed = process_response_dists(
                        response=response,
                        category="team"
                    )

                    dists_made.extend(current_dists_made)
                    dists_missed.extend(current_dists_missed)

    if option == "dists" and dists_made:
        hist_made = np.histogram(
            dists_made,
            bins=[i for i in range(min(dists_made), max(dists_made))]
        )
        hist_missed = np.histogram(
            dists_missed,
            bins=[i for i in range(min(dists_missed), max(dists_missed))]
        )

        np.savez(f"data/{team}/dists", hist_made[0], hist_made[1])
        np.savez(f"data/{team}/dists_missed", hist_missed[0], hist_missed[1])


def parse_team_shot_points(season):

    for team in teams_east + teams_west:
        url = f"{base_url}/teams/{team}/{season}_games.html"
        response = requests.get(url)
        if response.status_code == 200:
            parse_matches(response, team, option="points")


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

        print("Parsing", player)
        response = requests.get(url)
        if response.status_code == 200:
            missed_x, missed_y, made_x, made_y = process_response(
                response=response,
                category="player"
            )
            np.savez(f"data/{player}/missed", missed_x, missed_y)
            np.savez(f"data/{player}/made", made_x, made_y)


def process_response_dists(response, category):
    html = response.text
    soup = BeautifulSoup(re.sub("<!--|-->", "", html), "html.parser")

    shot_chart = soup.find_all("div", {"class": "shot-area"})[0]

    r = re.compile(r"^tooltip")
    points = shot_chart.find_all("div", {"class": r})

    dists_made = []
    dists_missed = []
    for point in points:
        if category == "player":
            message = point["tip"].split("<br>")[2]
        elif category == "team":
            message = point["tip"].split("<br>")[1]
        else:
            raise ValueError(f"{category} is not a valid category.")

        dist = int(message.split(" ")[-2])

        if "make" in point["class"]:
            dists_made.append(dist)
        else:
            dists_missed.append(dist)

    if category == "team":
        return dists_made, dists_missed
    else:
        hist_made = np.histogram(
            dists_made,
            bins=[i for i in range(min(dists_made), max(dists_made))]
        )
        hist_missed = np.histogram(
            dists_missed,
            bins=[i for i in range(min(dists_missed), max(dists_missed))]
        )

        return hist_made, hist_missed


def parse_player_shot_distances(season):

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

        print("Parsing", player)
        response = requests.get(url)
        if response.status_code == 200:
            hist_made, hist_missed = process_response_dists(
                response=response,
                category="player"
            )
            np.savez(f"data/{player}/dists", hist_made[0], hist_made[1])
            np.savez(f"data/{player}/dists_missed", hist_missed[0], hist_missed[1])


def parse_team_shot_distances(season):

    # TODO: This is only temporary bcs the scraping is bugging
    for team in ["MEM"]:
        url = f"{base_url}/teams/{team}/{season}_games.html"
        response = requests.get(url)
        if response.status_code == 200:
            parse_matches(response, team, option="dists")

        for _ in tqdm(range(0, 60), desc="Request cooldown (60s)"):
            time.sleep(1)
    print("Done!")


if __name__ == "__main__":
    parse_team_shot_distances("2024")
