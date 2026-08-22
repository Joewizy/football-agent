from collections import defaultdict

from langchain_core.tools import tool

from services.football_api import (
    LEAGUES,
    list_countries,
    search_team,
    get_squad,
    get_player,
    get_team_details,
    get_current_season_id,
    get_standings,
    get_top_scorers,
)

# SportMonks type 208 = "Goal Topscorer" (83=redcards, 84=yellowcards, 209=assists).
GOAL_TYPE_ID = 208

# Standings-row stat type_ids: a team's season totals live in each row's `details`.
STANDING_GOALS_FOR = 133
STANDING_GOALS_AGAINST = 134
STANDING_GOAL_DIFF = 179


def _standing_stat(row: dict, type_id: int):
    """Pull one stat value out of a standings row's details list by its type_id."""
    for detail in row.get("details") or []:
        if detail.get("type_id") == type_id:
            return detail.get("value")
    return None

def _resolve_season(league_name: str) -> int:
    """Map a free-plan league name to its current season id."""
    league_id = LEAGUES[league_name.strip().lower()]
    return get_current_season_id(league_id)


# SportMonks search matches the stored name literally, so a trailing club-type
# token ("Aberdeen FC") misses. Strip it as a fallback query.
_CLUB_SUFFIXES = {"fc", "afc", "cf", "sc", "fk", "if", "bk", "sk"}


def _search_variants(team_name: str) -> list:
    """The name as given, then a suffix-stripped fallback."""
    name = " ".join(team_name.split())
    variants = [name]
    tokens = name.split()
    if len(tokens) > 1 and tokens[-1].lower() in _CLUB_SUFFIXES:
        variants.append(" ".join(tokens[:-1]))
    return variants


@tool
def football_countries() -> list:
    """Return a list of football countries."""
    return list_countries()


@tool
def find_team(team_name: str) -> list:
    """Find football teams by name. Returns each match's id and name."""
    for query in _search_variants(team_name):
        teams = search_team(query).get("data") or []
        if teams:
            return [{"id": team["id"], "name": team["name"]} for team in teams]

    # Explicit message (not []) so the model reports unavailability instead of inventing an id.
    return [{"message": (
        f"No team named '{team_name}' is available on the free plan "
        "(only Scottish Premiership and Danish Superliga teams are covered)."
    )}]


@tool
def team_squad(team_id: int) -> list:
    """Return the player ids in a team's current squad."""
    raw = get_squad(team_id)
    players = raw["data"] or []

    return [{"player_id": player["player_id"]} for player in players]


@tool
def player_details(player_id: int) -> dict:
    """Return a single player's id and name."""
    raw = get_player(player_id)

    # A bad/unknown id returns a body with no "data" key instead of raising.
    player = raw.get("data")
    if not player:
        return {"error": f"No player found for id {player_id}."}

    return {"id": player["id"], "name": player["name"]}


@tool
def team_details(team_id: int) -> dict:
    """Return a team's profile: stadium, country, founding year, last and next match."""
    team = get_team_details(team_id).get("data")
    if not team:
        return {"error": f"No team found for id {team_id}."}

    latest = team.get("latest") or []
    upcoming = team.get("upcoming") or []

    return {
        "name": team.get("name"),
        "country": (team.get("country") or {}).get("name"),
        "stadium": (team.get("venue") or {}).get("name"),
        "founded": team.get("founded"),
        "last_match": latest[-1].get("name") if latest else None,
        "next_match": upcoming[0].get("name") if upcoming else None,
    }


@tool
def league_table(league_name: str) -> list:
    """Show the current standings for a free-plan league: each team's position,
    points, goals_for, goals_against and goal_difference. Use this to answer how
    many goals a team has scored or conceded this season.
    Valid names: "Scottish Premiership", "Danish Superliga".
    """
    if league_name.strip().lower() not in LEAGUES:
        return [{"error": f"League '{league_name}' is not on the free plan."}]

    season_id = _resolve_season(league_name)
    rows = get_standings(season_id).get("data") or []

    # A season can span several groups (Scottish football splits top-6/bottom-6).
    # The full-league table is simply the biggest group.
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("stage_id"), row.get("group_id"))].append(row)

    if not groups:
        return [{"error": f"No standings available yet for {league_name}."}]

    table = max(groups.values(), key=len)
    table.sort(key=lambda r: r.get("position", 0))

    return [
        {
            "position": r.get("position"),
            "team": (r.get("participant") or {}).get("name"),
            "points": r.get("points"),
            "goals_for": _standing_stat(r, STANDING_GOALS_FOR),
            "goals_against": _standing_stat(r, STANDING_GOALS_AGAINST),
            "goal_difference": _standing_stat(r, STANDING_GOAL_DIFF),
        }
        for r in table
    ]


@tool
def top_scorers(league_name: str) -> list:
    """Show the leading goalscorers for a free-plan league.
    Valid names: "Scottish Premiership", "Danish Superliga".
    """
    if league_name.strip().lower() not in LEAGUES:
        return [{"error": f"League '{league_name}' is not on the free plan."}]

    season_id = _resolve_season(league_name)
    rows = get_top_scorers(season_id).get("data") or []

    scorers = [r for r in rows if r.get("type_id") == GOAL_TYPE_ID]

    if not scorers:
        return [{"message": f"No top-scorer data available yet for {league_name} this season."}]

    return [
        {
            "position": r.get("position"),
            "player": (r.get("player") or {}).get("name"),
            "team": (r.get("participant") or {}).get("name"),
            "goals": r.get("total"),
        }
        for r in scorers[:10]
    ]
