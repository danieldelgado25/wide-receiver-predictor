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