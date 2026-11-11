"""
Replay a finished tournament as if it were being played live.

Fetches the final bracket.tennis page once, then re-serves it with only the
first K matches decided (round by round, as on a real tournament day), and
runs the real sync_tournament() + get_leaderboard() against each snapshot.
GitHub storage is replaced by in-memory tables, so nothing is written.

    python simulate_live.py [Tournament] [year] [tour]
    python simulate_live.py AustralianOpen 2026 ATP

This tests everything except how fast the site itself updates.
"""

import random
import sys

import pandas as pd
import requests
from bs4 import BeautifulSoup

import bracket_scraper
import pickem_app as A

MATCH_ATTR = 'data-match-id'


def snapshot_html(final_html, played):
    """Final page with only the first `played` matches (round, then slot order)
    decided. Later-round slots stay empty until both feeders are finished,
    like the real page, and unplayed matches have no winner styling."""
    soup = BeautifulSoup(final_html, 'html.parser')
    matches = {}
    for h5 in soup.find_all('h5'):
        for m in h5.parent.select(f'[{MATCH_ATTR}]'):
            r, i = (int(x) for x in m[MATCH_ATTR].split('-'))
            matches[(r, i)] = m
    order = sorted(matches)
    done = set(order[:played])

    for (r, i), m in matches.items():
        for slot, row in enumerate(m.select('div.bracket-match > div.flex-1')):
            button = row.select_one('button.bracket-button')
            if button is None:
                continue
            filled = r == 0 or (r - 1, 2 * i + slot) in done
            if not filled:
                for tag in button.find_all('a'):
                    tag.decompose()
                for tag in button.select('div.whitespace-nowrap'):
                    tag.clear()
            if (r, i) not in done:
                button['class'] = [c for c in button['class'] if c != 'font-medium']
                if filled:  # an in-progress match shows no result yet
                    for tag in button.select('div.whitespace-nowrap'):
                        tag.clear()
    return str(soup)


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


def main():
    tournament = sys.argv[1] if len(sys.argv) > 1 else 'AustralianOpen'
    year = int(sys.argv[2]) if len(sys.argv) > 2 else 2026
    tour = sys.argv[3] if len(sys.argv) > 3 else 'ATP'

    url = f'{bracket_scraper.BASE_URL}/{bracket_scraper.bracket_slug(tournament)}-{year}/{tour.lower()}'
    final_html = requests.get(url, headers=bracket_scraper.HEADERS, timeout=15).text
    total = len(BeautifulSoup(final_html, 'html.parser').select(f'[{MATCH_ATTR}]'))

    # In-memory stand-ins for the GitHub-backed tables.
    tables = {name: pd.DataFrame(columns=cols) for name, cols in A.TABLE_COLUMNS.items()}
    commits = []

    def fake_update(name, fn, message):
        tables[name] = fn(tables[name].copy()).astype(str)
        commits.append(message)
        return tables[name]

    A._read_table = lambda name: tables[name].copy()
    A._update_table = fake_update
    A.get_tournament = lambda tid: {'tour': tour, 'tournament_slug': tournament, 'year': year}

    # Entrants come from the pre-tournament page, like real pick-making.
    bracket_scraper.requests.get = lambda *a, **k: FakeResponse(snapshot_html(final_html, 0))
    entrants = A.load_entrants.__wrapped__(tour, tournament, year)

    # Three players each pick one random roster respecting the group sizes.
    rng = random.Random(7)
    tables['users'] = pd.DataFrame({'id': ['1', '2', '3'], 'username': ['ana', 'ben', 'cy'], 'created_at': ''})
    rows = []
    for uid in (1, 2, 3):
        for group, n in A.GROUPS_CONFIG:
            pool = entrants[entrants['Seeding_Group'] == group]['Player'].tolist()
            for slot, player in enumerate(rng.sample(pool, n)):
                rows.append({'id': len(rows) + 1, 'user_id': uid, 'tournament_id': 1,
                             'group_name': group, 'slot_index': slot, 'player_name': player})
    tables['picks'] = pd.DataFrame(rows).astype(str)

    # Snapshots: start, mid-round, end of every round, final.
    boundaries, played, size = [], 0, total // 2 + 1
    while size >= 1:
        played += size
        boundaries += [played - size // 2, played]
        size //= 2
    steps = sorted({0, *[b for b in boundaries if b <= total], total})

    previous = {}
    for k in steps:
        bracket_scraper.requests.get = lambda *a, k=k, **kw: FakeResponse(snapshot_html(final_html, k))
        before = len(commits)
        ok, msg = A.sync_tournament(1)
        lb, _ = A.get_leaderboard([1])
        points = dict(zip(lb['username'], lb['points']))
        assert all(points[u] >= previous.get(u, 0) for u in points), "points went down!"
        previous = points
        print(f"{k:>3}/{total} matches | {msg:<34} | commit: {'yes' if len(commits) > before else 'no '} | "
              + ', '.join(f"{u}={p}" for u, p in points.items()))

    progress = A.get_player_progress(1)
    print(f"\nFinal: {int((progress['eliminated'] == 0).sum())} player not eliminated (the champion), "
          f"{len(commits)} commits over {len(steps)} syncs.")


if __name__ == '__main__':
    main()
