"""
Trivote Tennis Prediction Game - Streamlit App

Casual multiplayer pick'em: people type their name, submit "Seeding Mode"
picks (one roster of 9 players across seed groups) for a tournament, and
scores are computed automatically by re-scraping the Tennis Abstract
forecast page as the real tournament progresses.

Results tracking insight: the forecast page only lists round-probability
columns for rounds that are still undecided for the players still shown.
Once a round finishes, the now-secured round's column disappears for
everyone who survived it, and anyone who lost simply vanishes from the
table. So a player's furthest secured round can be inferred purely from
repeated scrapes of that one page - no separate results feed is needed.

Storage: instead of a real database, this app reads/writes a handful of CSV
files committed directly into this GitHub repo via the GitHub Contents API.
That's the one thing that survives Streamlit Community Cloud's restarts.
There's no login/password system either - this is a casual game among
friends, so "logging in" is just typing a display name (honor system).
"""

import base64
import io
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

from forecast_scraper import scrape_tennis_forecast, rearrange_player_name

ROUND_ORDER = ['R64', 'R32', 'R16', 'QF', 'SF', 'F', 'W']
ROUND_POINTS = {'R64': 1, 'R32': 1, 'R16': 1, 'QF': 1, 'SF': 2, 'F': 2, 'W': 2}
GROUPS_CONFIG = [
    ('1-2', 1),
    ('3-4', 1),
    ('5-8', 1),
    ('9-16', 2),
    ('17-32', 3),
    ('Unseeded', 2),
]

GITHUB_API = "https://api.github.com"
DATA_DIR = "data"
TABLE_COLUMNS = {
    'users': ['id', 'username', 'created_at'],
    'tournaments': ['id', 'name', 'tour', 'tournament_slug', 'year', 'lock_time', 'created_by', 'created_at'],
    'picks': ['id', 'user_id', 'tournament_id', 'group_name', 'slot_index', 'player_name'],
    'player_progress': ['tournament_id', 'player_name', 'secured_round', 'eliminated', 'draw_position', 'seeding_group'],
}


# ---------------------------------------------------------------------------
# Time helpers - everything is stored and compared in UTC so the lock
# deadline and "last synced" timestamp mean the same thing for every player
# regardless of their own time zone. Each is also shown as a relative
# countdown/age, which needs no time zone math on the reader's part at all.
# ---------------------------------------------------------------------------

def utc_now():
    return datetime.now(timezone.utc)


def format_utc(dt):
    return dt.strftime('%Y-%m-%d %H:%M UTC')


def format_relative(dt, reference=None):
    reference = reference or utc_now()
    delta = dt - reference
    seconds = delta.total_seconds()
    suffix = "from now" if seconds >= 0 else "ago"
    seconds = abs(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, _ = divmod(seconds, 60)
    parts = []
    if days >= 1:
        parts.append(f"{int(days)}d")
    if hours >= 1 or days >= 1:
        parts.append(f"{int(hours)}h")
    parts.append(f"{int(minutes)}m")
    return f"{' '.join(parts)} {suffix}"


# ---------------------------------------------------------------------------
# Seeding helpers (duplicated from fantasy_app.py in the tennis_ratings repo
# - that module calls st.set_page_config() at import time, so it can't be
# imported directly from another Streamlit script without clashing).
# ---------------------------------------------------------------------------

def estimate_seeding_from_ranking(df):
    import re

    def extract_rank(player_name):
        match = re.search(r'\((\d+)\)', player_name)
        if match:
            return int(match.group(1))
        return 999

    df['Estimated_Rank'] = df['Player'].apply(extract_rank)
    df = df.sort_values('Estimated_Rank')
    df['Seed'] = df['Estimated_Rank'].apply(lambda x: x if x <= 32 else 999)
    return df


def add_seeding_group(df, seeding_col='Seed'):
    def get_group(seed):
        if pd.isna(seed) or seed > 32:
            return 'Unseeded'
        elif seed <= 2:
            return '1-2'
        elif seed <= 4:
            return '3-4'
        elif seed <= 8:
            return '5-8'
        elif seed <= 16:
            return '9-16'
        elif seed <= 32:
            return '17-32'
        else:
            return 'Unseeded'

    df['Seeding_Group'] = df[seeding_col].apply(get_group)
    return df


# ---------------------------------------------------------------------------
# Storage layer - CSV "tables" committed to this GitHub repo via the
# Contents API, instead of a real database. Reads are cached briefly so
# normal page interaction doesn't hit the GitHub API on every rerun; writes
# always re-fetch fresh and retry on a 409 (someone else wrote in between).
# ---------------------------------------------------------------------------

def _github_config():
    return (
        st.secrets["github_repo"],
        st.secrets["github_token"],
        st.secrets.get("github_branch", "main"),
    )


def _github_headers(token):
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }


def _table_path(name):
    return f"{DATA_DIR}/{name}.csv"


def _github_get_file(path):
    """Returns (content_str, sha), or (None, None) if the file doesn't exist yet."""
    repo, token, branch = _github_config()
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    resp = requests.get(url, headers=_github_headers(token), params={"ref": branch}, timeout=10)
    if resp.status_code == 404:
        return None, None
    resp.raise_for_status()
    data = resp.json()
    content = base64.b64decode(data["content"]).decode("utf-8")
    return content, data["sha"]


def _github_put_file(path, content_str, sha, message):
    repo, token, branch = _github_config()
    url = f"{GITHUB_API}/repos/{repo}/contents/{path}"
    body = {
        "message": message,
        "content": base64.b64encode(content_str.encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if sha:
        body["sha"] = sha
    return requests.put(url, headers=_github_headers(token), json=body, timeout=10)


def _parse_csv(content, name):
    if content is None:
        return pd.DataFrame(columns=TABLE_COLUMNS[name])
    return pd.read_csv(io.StringIO(content), dtype=str, keep_default_na=False)


@st.cache_data(ttl=30)
def _read_table(name):
    content, _ = _github_get_file(_table_path(name))
    return _parse_csv(content, name)


def _update_table(name, mutate_fn, message):
    """Fetches the current file fresh, applies mutate_fn(df) -> new_df, writes
    it back. Retries on a 409 (stale sha) by re-fetching and re-applying."""
    path = _table_path(name)
    last_error = None
    for _ in range(3):
        content, sha = _github_get_file(path)
        df = _parse_csv(content, name)
        new_df = mutate_fn(df)
        resp = _github_put_file(path, new_df.to_csv(index=False), sha, message)
        if resp.status_code in (200, 201):
            _read_table.clear()
            return new_df
        if resp.status_code == 409:
            last_error = "conflict"
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed to update {name}.csv after retries ({last_error})")


def _next_id(df):
    if df.empty:
        return 1
    return int(pd.to_numeric(df['id'], errors='coerce').max()) + 1


# ---------------------------------------------------------------------------
# Data access (mirrors what used to be plain SQL queries)
# ---------------------------------------------------------------------------

def get_or_create_user(username):
    username = username.strip()
    if not username:
        raise ValueError("Name is required")

    users_df = _read_table('users')
    existing = users_df[users_df['username'] == username]
    if not existing.empty:
        return int(existing.iloc[0]['id'])

    result = {}

    def mutate(df):
        existing_inner = df[df['username'] == username]
        if not existing_inner.empty:
            result['id'] = int(existing_inner.iloc[0]['id'])
            return df
        new_id = _next_id(df)
        result['id'] = new_id
        new_row = pd.DataFrame(
            [{'id': new_id, 'username': username, 'created_at': utc_now().isoformat()}],
            columns=TABLE_COLUMNS['users'],
        )
        return pd.concat([df, new_row], ignore_index=True)

    _update_table('users', mutate, f"Add player {username}")
    return result['id']


def create_event(name, slug, year, lock_dt_utc, created_by):
    """Creates the ATP and WTA tournament rows for this event together in a
    single commit, so setting up an event only takes one form submission
    instead of one per tour. lock_dt_utc must be a timezone-aware UTC datetime.
    No-ops (returns False) if an event with this name+year already exists -
    guards against double-submitting the form creating duplicate rows."""
    result = {'created': False}

    def mutate(df):
        if not df.empty:
            dup = df[(df['name'] == name) & (pd.to_numeric(df['year'], errors='coerce') == year)]
            if not dup.empty:
                return df
        next_id = _next_id(df)
        new_rows = [{
            'id': next_id + i, 'name': name, 'tour': tour, 'tournament_slug': slug,
            'year': year, 'lock_time': lock_dt_utc.isoformat(),
            'created_by': created_by, 'created_at': utc_now().isoformat(),
        } for i, tour in enumerate(["ATP", "WTA"])]
        result['created'] = True
        new_df = pd.DataFrame(new_rows, columns=TABLE_COLUMNS['tournaments'])
        return pd.concat([df, new_df], ignore_index=True)

    _update_table('tournaments', mutate, f"Create event {name}")
    return result['created']


def delete_event(tournament_ids):
    """Permanently deletes the given tournament rows along with any picks
    and player_progress rows tied to them (e.g. both the ATP and WTA
    tournaments of an event, to wipe the whole event in one go)."""
    tournament_ids = {int(t) for t in tournament_ids}

    def mutate_tournaments(df):
        mask = pd.to_numeric(df['id'], errors='coerce').isin(tournament_ids)
        return df[~mask]

    _update_table('tournaments', mutate_tournaments, f"Delete tournaments {sorted(tournament_ids)}")

    def mutate_dependent(df):
        mask = pd.to_numeric(df['tournament_id'], errors='coerce').isin(tournament_ids)
        return df[~mask]

    picks_df = _read_table('picks')
    if not picks_df.empty and pd.to_numeric(picks_df['tournament_id'], errors='coerce').isin(tournament_ids).any():
        _update_table('picks', mutate_dependent, f"Delete picks for tournaments {sorted(tournament_ids)}")

    progress_df = _read_table('player_progress')
    if not progress_df.empty and pd.to_numeric(progress_df['tournament_id'], errors='coerce').isin(tournament_ids).any():
        _update_table('player_progress', mutate_dependent, f"Delete player_progress for tournaments {sorted(tournament_ids)}")


def list_tournaments():
    df = _read_table('tournaments')
    if df.empty:
        return df
    return df.sort_values('created_at', ascending=False).reset_index(drop=True)


def get_tournament(tournament_id):
    df = _read_table('tournaments')
    if df.empty:
        return None
    match = df[pd.to_numeric(df['id'], errors='coerce') == int(tournament_id)]
    if match.empty:
        return None
    row = match.iloc[0].to_dict()
    row['id'] = int(row['id'])
    row['year'] = int(row['year'])
    row['created_by'] = int(row['created_by'])
    return row


def save_picks(user_id, tournament_id, picks):
    """picks: {group_name: [player_name, ...]}"""
    new_rows = []
    for group_name, players in picks.items():
        for slot_index, player_name in enumerate(players):
            new_rows.append({
                'user_id': int(user_id), 'tournament_id': int(tournament_id),
                'group_name': group_name, 'slot_index': slot_index, 'player_name': player_name,
            })

    def mutate(df):
        if not df.empty:
            mask = (pd.to_numeric(df['user_id'], errors='coerce') == int(user_id)) & \
                   (pd.to_numeric(df['tournament_id'], errors='coerce') == int(tournament_id))
            df = df[~mask]
        next_id = _next_id(df)
        rows_with_ids = [{'id': next_id + i, **r} for i, r in enumerate(new_rows)]
        new_df = pd.DataFrame(rows_with_ids, columns=TABLE_COLUMNS['picks'])
        return pd.concat([df, new_df], ignore_index=True) if not df.empty else new_df

    _update_table('picks', mutate, f"Save picks for user {user_id}, tournament {tournament_id}")


def get_user_picks(user_id, tournament_id):
    df = _read_table('picks')
    if df.empty:
        return df
    mask = (pd.to_numeric(df['user_id'], errors='coerce') == int(user_id)) & \
           (pd.to_numeric(df['tournament_id'], errors='coerce') == int(tournament_id))
    result = df[mask][['group_name', 'slot_index', 'player_name']].copy()
    if not result.empty:
        result['slot_index'] = pd.to_numeric(result['slot_index']).astype(int)
    return result


def get_all_picks(tournament_id):
    picks_df = _read_table('picks')
    empty = pd.DataFrame(columns=['user_id', 'username', 'group_name', 'slot_index', 'player_name'])
    if picks_df.empty:
        return empty
    picks_df = picks_df[pd.to_numeric(picks_df['tournament_id'], errors='coerce') == int(tournament_id)]
    if picks_df.empty:
        return empty

    users_df = _read_table('users')
    merged = picks_df.merge(users_df[['id', 'username']], left_on='user_id', right_on='id', how='left')
    merged['user_id'] = pd.to_numeric(merged['user_id']).astype(int)
    merged['slot_index'] = pd.to_numeric(merged['slot_index']).astype(int)
    return merged[['user_id', 'username', 'group_name', 'slot_index', 'player_name']]


def get_player_progress(tournament_id):
    df = _read_table('player_progress')
    empty = pd.DataFrame(columns=['player_name', 'secured_round', 'eliminated', 'draw_position', 'seeding_group'])
    if df.empty:
        return empty
    df = df[pd.to_numeric(df['tournament_id'], errors='coerce') == int(tournament_id)]
    if df.empty:
        return empty
    df = df.copy()
    df['secured_round'] = pd.to_numeric(df['secured_round']).astype(int)
    df['eliminated'] = pd.to_numeric(df['eliminated']).astype(int)
    return df[['player_name', 'secured_round', 'eliminated', 'draw_position', 'seeding_group']]


# ---------------------------------------------------------------------------
# Results sync - the core "automatic scoring" logic
# ---------------------------------------------------------------------------

def sync_tournament(tournament_id):
    """Re-scrape the forecast page and update each player's furthest secured
    round. All player-progress updates for this tournament are batched into
    a single commit, regardless of how many players are in the draw."""
    tournament = get_tournament(tournament_id)
    if not tournament:
        return False, "Tournament not found"

    try:
        df = scrape_tennis_forecast(
            tour=tournament['tour'],
            tournament=tournament['tournament_slug'],
            year=tournament['year'],
        )
        df = estimate_seeding_from_ranking(df)
        df = add_seeding_group(df, 'Seed')
    except Exception as e:
        return False, f"Scrape failed: {e}"

    cols_present = [r for r in ROUND_ORDER if f'{r}%' in df.columns]
    if not cols_present:
        return False, "No round columns found in the forecast table"

    # Every player still listed has already banked every round before the
    # first column still being shown - that column is the only one left
    # undecided for them.
    secured_prefix_idx = ROUND_ORDER.index(cols_present[0])

    current_players = {}
    for draw_pos, (_, row) in enumerate(df.iterrows()):
        player = row['Player']
        secured_idx = secured_prefix_idx - 1
        for i, r in enumerate(ROUND_ORDER):
            col = f'{r}%'
            if col in df.columns and row[col] >= 99.99:
                secured_idx = max(secured_idx, i)
        current_players[player] = {
            'secured_idx': secured_idx,
            'draw_position': draw_pos,
            'seeding_group': row['Seeding_Group'],
        }

    def mutate(progress_df):
        if progress_df.empty:
            other_rows, existing_map = progress_df, {}
        else:
            mask = pd.to_numeric(progress_df['tournament_id'], errors='coerce') == int(tournament_id)
            existing_rows = progress_df[mask]
            other_rows = progress_df[~mask]
            existing_map = {
                r['player_name']: {
                    'secured_round': int(r['secured_round']),
                    'draw_position': int(r.get('draw_position', -1)),
                    'seeding_group': r.get('seeding_group', 'Unseeded'),
                }
                for _, r in existing_rows.iterrows()
            }

        new_rows = []
        for player, info in current_players.items():
            prev = existing_map.get(player, {})
            secured_idx = max(info['secured_idx'], prev.get('secured_round', -1))
            new_rows.append({
                'tournament_id': int(tournament_id),
                'player_name': player,
                'secured_round': secured_idx,
                'eliminated': 0,
                'draw_position': info['draw_position'],
                'seeding_group': info['seeding_group'],
            })

        for player, info in existing_map.items():
            if player not in current_players:
                new_rows.append({
                    'tournament_id': int(tournament_id),
                    'player_name': player,
                    'secured_round': info['secured_round'],
                    'eliminated': 1,
                    'draw_position': info['draw_position'],
                    'seeding_group': info['seeding_group'],
                })

        new_for_tournament = pd.DataFrame(new_rows, columns=TABLE_COLUMNS['player_progress'])
        if other_rows.empty:
            return new_for_tournament
        return pd.concat([other_rows, new_for_tournament], ignore_index=True)

    _update_table('player_progress', mutate, f"Sync tournament {tournament_id}")
    return True, f"Synced {len(current_players)} active players"


@st.cache_data(ttl=600)
def auto_sync(tournament_id):
    """Cached for 10 min - returns (ok, message, synced_at_utc). The timestamp
    is what's actually cached, so 'X ago' below stays accurate on every
    rerun even between re-syncs."""
    ok, msg = sync_tournament(tournament_id)
    return ok, msg, utc_now()


def get_leaderboard(tournament_ids):
    """tournament_ids: iterable of tournament ids to pool together (e.g. the
    ATP and WTA tournaments of the same event), so one player's total is
    their points summed across every tour they picked for that event."""
    all_picks = []
    for tournament_id in tournament_ids:
        picks_df = get_all_picks(tournament_id)
        if picks_df.empty:
            continue

        progress_df = get_player_progress(tournament_id)
        secured_map = dict(zip(progress_df['player_name'], progress_df['secured_round']))
        elim_map = dict(zip(progress_df['player_name'], progress_df['eliminated']))

        picks_df = picks_df.copy()
        picks_df['tournament_id'] = tournament_id
        picks_df['secured_round'] = picks_df['player_name'].map(secured_map).fillna(-1).astype(int)
        picks_df['round_label'] = picks_df['secured_round'].apply(lambda i: ROUND_ORDER[i] if i >= 0 else '-')
        picks_df['status'] = picks_df['player_name'].apply(
            lambda p: 'Eliminated' if elim_map.get(p) else ('Active' if p in secured_map else 'Not started')
        )
        GROUP_LEVEL = {'1-2': 0, '3-4': 1, '5-8': 2, '9-16': 3, '17-32': 4, 'Unseeded': 5}

        pos_to_group = {
            int(r['draw_position']): r['seeding_group']
            for _, r in progress_df.iterrows()
            if str(r.get('draw_position', '')).lstrip('-').isdigit() and int(r['draw_position']) >= 0
        }

        player_to_pos = dict(zip(
            progress_df['player_name'],
            progress_df['draw_position'].astype(int),
        ))

        player_to_group = dict(zip(
            progress_df['player_name'],
            progress_df['seeding_group'],
        ))

        def compute_points(secured_round_idx, player_name, own_group):
            if secured_round_idx < 0:
                return 0
            base = sum(ROUND_POINTS[r] for r in ROUND_ORDER[:secured_round_idx + 1])
            draw_pos = player_to_pos.get(player_name, -1)
            if draw_pos < 0:
                return base
            bonus = 0
            own_level = GROUP_LEVEL.get(own_group, 5)
            for round_idx in range(secured_round_idx + 1):
                # In a draw listed in order, the opponent in round r is at pos XOR 2^r:
                # round 0 (R64): flip bit 0 ? adjacent player
                # round 1 (R32): flip bit 1 ? other pair in group of 4
                # round 2 (R16): flip bit 2 ? other group of 4 in group of 8, etc.
                opponent_pos = draw_pos ^ (1 << round_idx)
                opponent_group = pos_to_group.get(opponent_pos)
                if opponent_group is not None and GROUP_LEVEL.get(opponent_group, 5) < own_level:
                    bonus += 1
            return base + bonus

        picks_df['points'] = picks_df.apply(
            lambda row: compute_points(
                row['secured_round'],
                row['player_name'],
                player_to_group.get(row['player_name'], row['group_name']),
            ),
            axis=1,
        )

        all_picks.append(picks_df)

    if not all_picks:
        return pd.DataFrame(columns=['user_id', 'username', 'points']), pd.DataFrame()

    combined_picks = pd.concat(all_picks, ignore_index=True)
    leaderboard = (
        combined_picks.groupby(['user_id', 'username'])['points']
        .sum()
        .reset_index()
        .sort_values('points', ascending=False)
        .reset_index(drop=True)
    )
    return leaderboard, combined_picks


@st.cache_data(ttl=600)
def load_entrants(tour, slug, year):
    """Entrant list with seed/group only - no probabilities, so picks stay blind."""
    df = scrape_tennis_forecast(tour=tour, tournament=slug, year=year)
    df = estimate_seeding_from_ranking(df)
    df = add_seeding_group(df, 'Seed')
    return df[['Player', 'Seed', 'Seeding_Group']].reset_index(drop=True)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def login_section():
    if 'user_id' in st.session_state:
        st.sidebar.markdown(f"Playing as **{st.session_state.username}**")
        if st.sidebar.button("Switch name"):
            st.session_state.clear()
            st.rerun()
        return

    st.sidebar.subheader("Who's playing?")
    username = st.sidebar.text_input("Your name")
    if st.sidebar.button("Continue", use_container_width=True):
        try:
            user_id = get_or_create_user(username)
            st.session_state.user_id = user_id
            st.session_state.username = username.strip()
            st.rerun()
        except ValueError as e:
            st.sidebar.error(str(e))
    st.stop()


def tournament_section():
    """Tournaments are grouped into one "event" per (display name, year), so
    an ATP and a WTA tournament created with the same name/year (e.g. both
    named "Wimbledon" 2026) share a single combined leaderboard."""
    tournaments = list_tournaments()
    st.sidebar.subheader("Tournament")

    selected_event = None
    if not tournaments.empty:
        groups = {}
        order = []
        for _, row in tournaments.iterrows():
            key = (row['name'], int(row['year']))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(int(row['id']))

        options = {}
        for key in order:
            tours = "+".join(sorted(get_tournament(tid)['tour'] for tid in groups[key]))
            options[f"{key[0]} ({key[1]}) - {tours}"] = key
        choice = st.sidebar.selectbox("Select tournament", list(options.keys()))
        key = options[choice]
        selected_event = {
            'name': key[0],
            'year': key[1],
            'tournaments': [get_tournament(tid) for tid in groups[key]],
        }

        confirm_key = f"confirm_delete_{key[0]}_{key[1]}"
        with st.sidebar.expander("Delete this tournament"):
            st.warning(
                f"This permanently deletes \"{selected_event['name']}\" ({selected_event['year']}) "
                "and every player's picks for it. This cannot be undone."
            )
            if not st.session_state.get(confirm_key):
                if st.button("Delete tournament", key=f"delete_btn_{key[0]}_{key[1]}"):
                    st.session_state[confirm_key] = True
                    st.rerun()
            else:
                st.error("Are you sure? This cannot be undone.")
                col_confirm, col_cancel = st.columns(2)
                with col_confirm:
                    if st.button("Yes, delete", key=f"confirm_delete_btn_{key[0]}_{key[1]}"):
                        delete_event([t['id'] for t in selected_event['tournaments']])
                        st.session_state.pop(confirm_key, None)
                        st.success("Tournament deleted")
                        st.rerun()
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_delete_btn_{key[0]}_{key[1]}"):
                        st.session_state.pop(confirm_key, None)
                        st.rerun()
    else:
        st.sidebar.info("No tournaments yet - create one below.")

    with st.sidebar.expander("+ Create tournament"):
        name = st.text_input("Display name", key="new_t_name", placeholder="e.g. Wimbledon 2026")
        st.caption("Creates both the ATP and WTA draws for this event, combined into one leaderboard.")
        slug = st.text_input(
            "Tennis Abstract slug", key="new_t_slug", placeholder="Wimbledon",
            help="e.g. Wimbledon, RolandGarros, AustralianOpen, USOpen, IndianWells, Rome",
        )
        year = st.number_input("Year", min_value=2020, max_value=2030, value=utc_now().year, key="new_t_year")
        st.caption(f"Current time is {format_utc(utc_now())} - enter the lock deadline below in UTC too.")
        lock_date = st.date_input("Picks lock date (UTC)", value=utc_now().date(), key="new_t_lock_date")
        lock_time = st.time_input("Picks lock time (UTC)", key="new_t_lock_time")
        if st.button("Create tournament"):
            if not name or not slug:
                st.error("Display name and Tennis Abstract slug are required")
            else:
                lock_dt_utc = datetime.combine(lock_date, lock_time, tzinfo=timezone.utc)
                if create_event(name, slug, int(year), lock_dt_utc, st.session_state.user_id):
                    st.success("Tournament created")
                    st.rerun()
                else:
                    st.warning(f"An event named '{name}' ({year}) already exists - select it from the dropdown above.")

    return selected_event


def render_picks_tab(tournament, entrants_df, user_id):
    lock_dt = datetime.fromisoformat(tournament['lock_time'])
    locked = utc_now() > lock_dt
    if locked:
        st.info(f"Picks locked {format_utc(lock_dt)} ({format_relative(lock_dt)}) - you can still see your roster below.")
    else:
        st.caption(f"Picks lock {format_utc(lock_dt)} ({format_relative(lock_dt)})")

    existing = get_user_picks(user_id, tournament['id'])
    existing_map = {}
    for _, row in existing.iterrows():
        existing_map.setdefault(row['group_name'], {})[row['slot_index']] = row['player_name']

    selections = {}
    all_chosen = set()

    for group_name, num_picks in GROUPS_CONFIG:
        group_players = entrants_df[entrants_df['Seeding_Group'] == group_name].copy()
        if group_name == 'Unseeded':
            group_players = group_players.sort_values('Player')
        else:
            group_players = group_players.sort_values('Seed')
        options = group_players['Player'].tolist()
        labels = {p: rearrange_player_name(p) for p in options}

        st.markdown(f"**Seeds {group_name}** - pick {num_picks}")
        chosen_for_group = []
        cols = st.columns(num_picks)
        for slot in range(num_picks):
            default_player = existing_map.get(group_name, {}).get(slot)
            available = [p for p in options if p not in all_chosen or p == default_player]
            display_options = ["-"] + available
            default_index = display_options.index(default_player) if default_player in display_options else 0
            with cols[slot]:
                pick = st.selectbox(
                    f"{group_name} slot {slot + 1}",
                    display_options,
                    index=default_index,
                    format_func=lambda p: labels.get(p, p),
                    key=f"pick_{tournament['id']}_{group_name}_{slot}",
                    disabled=locked,
                )
            if pick != "-":
                chosen_for_group.append(pick)
                all_chosen.add(pick)
        selections[group_name] = chosen_for_group

    if not locked and st.button("Save picks", key=f"save_picks_{tournament['id']}"):
        total_slots = sum(n for _, n in GROUPS_CONFIG)
        total_selected = sum(len(v) for v in selections.values())
        if total_selected < total_slots:
            st.warning(f"Fill every slot before saving ({total_selected}/{total_slots} filled).")
        else:
            save_picks(user_id, tournament['id'], selections)
            st.success("Picks saved!")
            st.rerun()


def render_leaderboard_tab(tournaments, user_id):
    """tournaments: every tour's tournament for this event (e.g. ATP + WTA),
    whose points get pooled into one combined leaderboard."""
    _, refresh_col = st.columns([4, 1])
    with refresh_col:
        if st.button("Refresh now"):
            auto_sync.clear()
            for t in tournaments:
                ok, msg, synced_at = auto_sync(t['id'])
                st.toast(f"{t['tour']}: {msg}" if ok else f"{t['tour']}: {msg}")
            st.rerun()

    locked = True
    for t in tournaments:
        lock_dt = datetime.fromisoformat(t['lock_time'])
        locked = locked and utc_now() > lock_dt
        ok, msg, synced_at = auto_sync(t['id'])
        status = msg if ok else f":warning: {msg}"
        st.caption(
            f"**{t['tour']}** picks lock {format_utc(lock_dt)} ({format_relative(lock_dt)}) - "
            f"{status}, last synced {format_relative(synced_at)} (auto-refreshes at most every 10 min)"
        )

    leaderboard, picks_df = get_leaderboard([t['id'] for t in tournaments])
    if leaderboard.empty:
        st.info("No picks submitted yet.")
        return

    st.dataframe(
        leaderboard.rename(columns={'username': 'Player', 'points': 'Points'})[['Player', 'Points']],
        hide_index=True, use_container_width=True,
    )

    st.markdown("### Pick details")
    for _, row in leaderboard.iterrows():
        is_self = row['user_id'] == user_id
        if not is_self and not locked:
            continue
        with st.expander(f"{row['username']} - {row['points']} pts"):
            user_picks = picks_df[picks_df['user_id'] == row['user_id']].copy()
            tour_tabs = st.tabs([t['tour'] for t in tournaments])
            for tour_tab, t in zip(tour_tabs, tournaments):
                with tour_tab:
                    tour_picks = user_picks[user_picks['tournament_id'] == t['id']][
                        ['group_name', 'player_name', 'round_label', 'status', 'points']
                    ].rename(columns={
                        'group_name': 'Group', 'player_name': 'Player',
                        'round_label': 'Secured Round', 'status': 'Status', 'points': 'Points',
                    }).sort_values('Group')
                    st.dataframe(tour_picks, hide_index=True, use_container_width=True)

    if not locked:
        st.caption("Other players' pick breakdowns are hidden until all picks lock.")


def main():
    st.set_page_config(page_title="Trivote Tennis Prediction Game", page_icon=":trophy:", layout="wide")

    if "github_repo" not in st.secrets or "github_token" not in st.secrets:
        st.error(
            "Missing GitHub secrets. Copy .streamlit/secrets.toml.example to "
            ".streamlit/secrets.toml and fill in github_repo/github_token."
        )
        st.stop()

    st.title(":trophy: Trivote Tennis Prediction Game")

    login_section()
    event = tournament_section()

    if event is None:
        st.info("Create a tournament in the sidebar to get started.")
        return

    tournaments = event['tournaments']
    st.header(f"{event['name']} ({event['year']})")

    tab_labels = [f"Make {t['tour']} Picks" for t in tournaments] + ["Leaderboard"]
    tabs = st.tabs(tab_labels)
    for tab, tournament in zip(tabs[:-1], tournaments):
        with tab:
            try:
                entrants_df = load_entrants(tournament['tour'], tournament['tournament_slug'], tournament['year'])
            except Exception as e:
                st.error(f"Could not load entrants from Tennis Abstract: {e}")
                continue
            render_picks_tab(tournament, entrants_df, st.session_state.user_id)
    with tabs[-1]:
        render_leaderboard_tab(tournaments, st.session_state.user_id)


if __name__ == "__main__":
    main()
