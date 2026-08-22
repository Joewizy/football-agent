import requests

from config import FOOTBALL_API_KEY as API_KEY

BASE_URL = "https://api.sportmonks.com/v3/football"
CORE_URL = "https://api.sportmonks.com/v3/core"
TEAMS_URL = "https://api.sportmonks.com/v3/football/teams"

# The only leagues our free plan covers.
LEAGUES = {
    "scottish premiership": 501,
    "danish superliga": 271,
}

def get_countries():
    url = f"{CORE_URL}/countries"

    response = requests.get(
        url,
        params={
            "api_token": API_KEY
        }
    )

    response.raise_for_status()

    return response.json()


def list_countries():
    """Return just id + name for each country — nothing else."""
    raw = get_countries()                 # the whole blob (dict)
    countries = raw["data"]               # dig out the list

    result = []
    for country in countries:             # each `country` is a dict
        result.append({
            "id": country["id"],
            "name": country["name"],
        })

    return result

def check_if_pagination_exist():
    """Check if pagination exists in the response."""
    raw = get_countries()
    return raw["pagination"]["has_more"] 

def search_teams(country_id: int):
    """Search for teams in a given country."""
    response = requests.get(
        TEAMS_URL,
        params={
            "api_token": API_KEY,
            "country_id": country_id,
        },
    )

    response.raise_for_status()

    return response.json()


def search_team(team_name: str):
    """Find teams whose name matches `team_name`."""
    url = f"{BASE_URL}/teams/search/{team_name}"

    response = requests.get(
        url,
        params={
            "api_token": API_KEY,
        },
    )

    response.raise_for_status()

    return response.json()


def get_squad(team_id: int):
    """Return the current squad (players) of a team."""
    url = f"{BASE_URL}/squads/teams/{team_id}"

    response = requests.get(
        url,
        params={
            "api_token": API_KEY,
        },
    )

    response.raise_for_status()

    return response.json()


def get_player(player_id: int):
    """Return one player, including their statistics."""
    url = f"{BASE_URL}/players/{player_id}"

    response = requests.get(
        url,
        params={
            "api_token": API_KEY,
            "include": "statistics",
        },
    )

    response.raise_for_status()

    return response.json()


def get_team_details(team_id: int):
    """Return one team with venue, country, coaches and recent/next fixtures."""
    url = f"{BASE_URL}/teams/{team_id}"

    response = requests.get(
        url,
        params={
            "api_token": API_KEY,
            "include": "venue;country;coaches;latest;upcoming",
        },
    )

    response.raise_for_status()

    return response.json()


def get_current_season_id(league_id: int) -> int:
    """Resolve a league's active season id so we never hardcode one."""
    url = f"{BASE_URL}/leagues/{league_id}"

    response = requests.get(
        url,
        params={
            "api_token": API_KEY,
            "include": "currentSeason",
        },
    )

    response.raise_for_status()

    return response.json()["data"]["currentseason"]["id"]


def get_standings(season_id: int):
    """Return raw league standings for a season (each team's row)."""
    url = f"{BASE_URL}/standings/seasons/{season_id}"

    response = requests.get(
        url,
        params={
            "api_token": API_KEY,
            # details carries per-team stats (goals for/against, GD) as type_id rows.
            "include": "participant;details",
        },
    )

    response.raise_for_status()

    return response.json()


def get_top_scorers(season_id: int):
    """Return raw top-scorer rows for a season."""
    url = f"{BASE_URL}/topscorers/seasons/{season_id}"

    response = requests.get(
        url,
        params={
            "api_token": API_KEY,
            "include": "player;participant",
            # The endpoint returns ALL stat types (cards, assists, goals) sorted by
            # type_id, so goals (208) get pushed past page 1. Filter to goals only.
            "filters": "seasonTopscorerTypes:208",
        },
    )

    response.raise_for_status()

    return response.json()