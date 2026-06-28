"""
Tennis Abstract forecast scraper.

Lightweight copy of the scraping logic from the tennis_ratings project's
tournament_draws_test.py - kept dependency-free (no matplotlib/scikit-learn)
since this is the only part of that module the pick'em app actually needs.
"""

import re

import requests
from bs4 import BeautifulSoup
import pandas as pd


def scrape_tennis_forecast(tour='ATP', tournament='Wimbledon', year=2025):
    """
    Scrape tennis forecast data from tennisabstract.com

    Parameters:
    tour (str): 'ATP' or 'WTA'
    tournament (str): Tournament name (e.g., 'Wimbledon', 'IndianWells', 'Rome')
    year (int): Tournament year

    Returns:
    pd.DataFrame: Forecast data
    """
    tour_upper = tour.upper()

    forecast_tournaments = ['Wimbledon', 'AustralianOpen', 'RolandGarros', 'USOpen']
    if tour_upper == 'WTA':
        if tournament in forecast_tournaments:
            url = f'https://www.tennisabstract.com/current/{year}{tournament}WomenForecast.html'
        else:
            url = f'https://www.tennisabstract.com/current/{year}WTA{tournament}.html'
    else:  # ATP
        if tournament in forecast_tournaments:
            url = f'https://www.tennisabstract.com/current/{year}{tournament}MenForecast.html'
        else:
            url = f'https://www.tennisabstract.com/current/{year}ATP{tournament}.html'

    req_headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, timeout=10, headers=req_headers)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch data from {url}: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    forecast_table = None
    for table in soup.find_all("table"):
        if len(table.find_all("tr")) > 50:
            forecast_table = table
            break

    if not forecast_table:
        raise ValueError("No forecast table found with sufficient data")

    known_rounds = ('R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F', 'W')
    round_names = []
    col_indices = []
    for row in forecast_table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if cells and cells[0].text.strip() == 'Player':
            for i, cell in enumerate(cells):
                name = cell.text.strip()
                if name in known_rounds:
                    round_names.append(name + '%')
                    col_indices.append(i)
            break

    if not round_names:
        col_indices = [2, 3, 4, 5, 6, 7, 8]
        round_names = ['R64%', 'R32%', 'R16%', 'QF%', 'SF%', 'F%', 'W%']

    min_cols = max(col_indices) + 1

    forecast_data = []
    for row in forecast_table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < min_cols:
            continue
        try:
            player = cols[0].text.strip()
            if not player or 'Player' in player:
                continue
            percentages = []
            for i in col_indices:
                text = cols[i].text.strip().replace('%', '')
                percentages.append(float(text) if text else 0.0)
            forecast_data.append([player] + percentages)
        except (ValueError, IndexError):
            continue

    if not forecast_data:
        raise ValueError("No valid forecast data found")

    columns = ["Player"] + round_names
    return pd.DataFrame(forecast_data, columns=columns)


def rearrange_player_name(player):
    """Clean and rearrange player name format"""
    if len(player) > 6 and player.endswith(player[-6:]):
        player = player[:-6]

    match = re.match(r'\((\d+)\)(.+)', player)
    if match:
        return f'{match.group(2).strip()} ({match.group(1)})'
    return player
