"""
bracket.tennis scraper.

Reads a tournament's full draw and results from
https://bracket.tennis/tournaments/<slug>-<year>/<atp|wta>.
The page is server-rendered, so a plain GET contains every match: players,
set scores and (via bold styling) the winner of each finished match.
"""

import re

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = 'https://bracket.tennis/tournaments'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}


def bracket_slug(tournament):
    """'AustralianOpen' -> 'australian-open', 'USOpen' -> 'us-open'."""
    return re.sub(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])', '-', tournament).lower()


def _match_sides(match):
    """The two player slots of a match, top first. A slot that is not decided
    yet (TBD) has no button on the page, so go through the row containers
    to keep slot positions right; such a slot comes back as None."""
    rows = match.select('div.bracket-match > div.flex-1')
    return [_parse_player(row.select_one('button.bracket-button')) for row in rows]


def _parse_player(button):
    link = button.find('a') if button else None
    if link is None:  # slot not decided yet (TBD / bye)
        return None
    seed_span = link.find_next_sibling('span')
    seed_match = re.search(r'\d+', seed_span.text) if seed_span else None
    flag = button.find('use')
    country = flag['href'].rsplit('flag-', 1)[-1].upper() if flag and 'flag-' in flag['href'] else ''
    return {
        'id': link['href'],
        'name': link.text.strip(),
        'seed': int(seed_match.group()) if seed_match else None,
        'country': country,
        'won': 'font-medium' in (button.get('class') or []),
    }


def scrape_bracket(tour='ATP', tournament='Wimbledon', year=2026):
    """
    Returns one row per player in the draw, in draw order (the order the
    pick'em scoring relies on to find first-round opponents).

    Columns: Player ("(seed)Name (IOC)"), Seed (999 if unseeded),
             Wins (matches won so far), Eliminated (0/1),
             BeatenSeeds (seed of each opponent beaten, in round order)
    """
    url = f'{BASE_URL}/{bracket_slug(tournament)}-{year}/{tour.lower()}'
    try:
        response = requests.get(url, timeout=15, headers=HEADERS)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ValueError(f"Failed to fetch data from {url}: {e}")

    soup = BeautifulSoup(response.text, 'html.parser')
    rounds = [h.parent for h in soup.find_all('h5') if h.parent.select('[data-match-id]')]
    if not rounds:
        raise ValueError(f"No bracket found at {url}")

    # The site sometimes links the same player under different slugs in
    # different rounds (e.g. catherine-mcnally / caty-mcnally), so players are
    # followed by bracket position, not identity: the winner of match "r-i"
    # fills slot i % 2 of match "r+1-(i // 2)".
    matches = {}
    for col in rounds:
        for match in col.select('[data-match-id]'):
            r, i = (int(x) for x in match['data-match-id'].split('-'))
            matches[(r, i)] = _match_sides(match)

    def surname(name):
        return name.split()[-1].strip('.').lower()

    def winner_slot(r, idx):
        """Slot (0/1) that won match (r, idx), or None if undecided. The site
        occasionally leaves a finished match (retirement/walkover) without a
        winner mark; then the player already shown in the next round's slot
        for this match tells us who advanced."""
        sides = matches[(r, idx)]
        flagged = [k for k, p in enumerate(sides) if p and p['won']]
        if flagged:
            return flagged[0]
        advanced = matches.get((r + 1, idx // 2), [None, None])[idx % 2]
        if advanced:
            hits = [k for k, p in enumerate(sides) if p and surname(p['name']) == surname(advanced['name'])]
            if len(hits) == 1:
                return hits[0]
        return None

    def origin(r, idx, slot):
        """Round-0 player who ended up in this slot (follows winners back)."""
        while r > 0:
            feeder = (r - 1, 2 * idx + slot)
            winner = winner_slot(*feeder)
            if winner is None:
                return None
            r, idx, slot = feeder[0], feeder[1], winner
        return matches[(0, idx)][slot]

    rows = []
    for i in range(len([k for k in matches if k[0] == 0])):
        for side in (0, 1):
            player = matches[(0, i)][side]
            if player is None:
                continue
            wins, eliminated, beaten = 0, 0, []
            r, idx, slot = 0, i, side
            while (r, idx) in matches:
                winner = winner_slot(r, idx)
                if winner is None:  # not decided yet
                    break
                if winner != slot:
                    eliminated = 1
                    break
                wins += 1
                opponent = origin(r, idx, 1 - slot)
                if opponent is not None:
                    beaten.append(opponent['seed'] or 999)
                r, idx, slot = r + 1, idx // 2, idx % 2
            prefix = f"({player['seed']})" if player['seed'] else ''
            country = f" ({player['country']})" if player['country'] else ''
            rows.append([f"{prefix}{player['name']}{country}", player['seed'] or 999, wins, eliminated, beaten])

    if not rows:
        raise ValueError(f"No players found in first round at {url}")

    return pd.DataFrame(rows, columns=['Player', 'Seed', 'Wins', 'Eliminated', 'BeatenSeeds'])


def rearrange_player_name(player):
    """'(1)Aryna Sabalenka (BLR)' -> 'Aryna Sabalenka (1)'"""
    if len(player) > 6 and player.endswith(player[-6:]):
        player = player[:-6]

    match = re.match(r'\((\d+)\)(.+)', player)
    if match:
        return f'{match.group(2).strip()} ({match.group(1)})'
    return player


def player_key(player):
    """Stable identity for matching picks to the draw: the name without its
    '(seed)' prefix, since the seed can change after picks were saved."""
    return re.sub(r'^\(\d+\)', '', player).strip()
