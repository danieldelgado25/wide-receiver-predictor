from __future__ import annotations

import polars as pl
import nflreadpy as nfl


def load_player_weekly_stats(seasons: list[int]) -> pl.DataFrame:
    """
    Load weekly player stats from nflreadpy.
    """
    data_frame = nfl.load_player_stats(seasons)
    return _to_polars(data_frame)


def load_players() -> pl.DataFrame:
    """
    Load player metadata.
    """
    data_frame = nfl.load_players()
    return _to_polars(data_frame)


def load_schedules(seasons: list[int]) -> pl.DataFrame:
    """
    Load NFL schedules/results.
    """
    data_frame = nfl.load_schedules(seasons)
    return _to_polars(data_frame)


def load_injuries(seasons: list[int]) -> pl.DataFrame:
    """
    Load injuries/practice participation.
    """
    data_frame = nfl.load_injuries(seasons)
    return _to_polars(data_frame)


def load_ff_opportunity(seasons: list[int]) -> pl.DataFrame:
    """
    Load fantasy opportunity / expected production data.
    Safe fallback if unavailable for your current setup.
    """
    try:
        data_frame = nfl.load_ff_opportunity(seasons)
        return _to_polars(data_frame)
    except Exception:
        return pl.DataFrame()


def _to_polars(data_frame) -> pl.DataFrame:
    """
    nflreadpy is Polars-native, but this keeps the wrapper defensive.
    """
    if isinstance(data_frame, pl.DataFrame):
        return data_frame
    if hasattr(data_frame, "to_pandas"):
        return pl.from_pandas(data_frame.to_pandas())
    return pl.DataFrame(data_frame)


def filter_wide_receivers(data_frame: pl.DataFrame) -> pl.DataFrame:
    """
    Filter the player stats dataframe to only include wide receivers.
    """
    return data_frame.filter(pl.col("position") == "WR")


def select_wr_columns(data_frame: pl.DataFrame) -> pl.DataFrame:
    """
    Select only the columns relevant for wide receiver analysis.
    Drops QB stats, FG stats, PAT stats, etc.
    """
    keep_columns = [
        "player_id", "player_name", "player_display_name", "position", "position_group",
        "headshot_url", "season", "week", "season_type", "game_id", "team", "opponent_team",
        "carries", "rushing_yards", "rushing_tds", "rushing_fumbles", "rushing_fumbles_lost",
        "rushing_first_downs", "rushing_epa", "rushing_2pt_conversions",
        "receptions", "receiving_yards", "receiving_tds", "receiving_fumbles", "receiving_fumbles_lost",
        "receiving_air_yards", "receiving_yards_after_catch", "receiving_first_downs", "receiving_epa",
        "receiving_2pt_conversions", "target_share", "air_yards_share", "wopr", "ryoe", "ryoe_perception",
        "adot", "ryoe_allowed", "ryoe_pass", "ryoe_run", "pacr", "racr",
        "kickoff_return_yards", "punt_return_yards", "punt_return_tds",
        "lateral_receptions", "lateral_rush_yards", "lateral_rush_tds", "lateral_return_yards", "lateral_return_tds",
        "fantasy_points", "fantasy_points_ppr"
    ]
    # Filter to only existing columns
    existing_columns = [col for col in keep_columns if col in data_frame.columns]
    return data_frame.select(existing_columns)