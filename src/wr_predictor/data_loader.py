from __future__ import annotations

import polars as pl
import nflreadpy as nfl


def load_player_weekly_stats(seasons: list[int]) -> pl.DataFrame:
    """
    Load weekly player stats from nflreadpy.
    """
    df = nfl.load_player_stats(seasons)
    return _to_polars(df)


def load_players() -> pl.DataFrame:
    """
    Load player metadata.
    """
    df = nfl.load_players()
    return _to_polars(df)


def load_schedules(seasons: list[int]) -> pl.DataFrame:
    """
    Load NFL schedules/results.
    """
    df = nfl.load_schedules(seasons)
    return _to_polars(df)


def load_injuries(seasons: list[int]) -> pl.DataFrame:
    """
    Load injuries/practice participation.
    """
    df = nfl.load_injuries(seasons)
    return _to_polars(df)


def load_ff_opportunity(seasons: list[int]) -> pl.DataFrame:
    """
    Load fantasy opportunity / expected production data.
    Safe fallback if unavailable for your current setup.
    """
    try:
        df = nfl.load_ff_opportunity(seasons)
        return _to_polars(df)
    except Exception:
        return pl.DataFrame()


def _to_polars(df) -> pl.DataFrame:
    """
    nflreadpy is Polars-native, but this keeps the wrapper defensive.
    """
    if isinstance(df, pl.DataFrame):
        return df
    if hasattr(df, "to_pandas"):
        return pl.from_pandas(df.to_pandas())
    return pl.DataFrame(df)