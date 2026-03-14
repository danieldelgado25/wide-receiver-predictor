from __future__ import annotations

import os
import polars as pl

from src.wr_predictor import data_loader, features

"""
File imported by main.py to build the training dataset.
Pulls data using data_loader.py, Filters to WRs using filters.py,
Adds fantasy points using features.py, created targets using targets.py,
Writes final dataset to a CSV file and returns the Polars DataFrame.
"""


def build_training_dataset(
    seasons: list[int],
    min_games_for_player: int = 4,
    output_path: str | None = None,
    position: str = "WR",
) -> pl.DataFrame:
    """
    Build the WR training dataset: load stats, filter to position, add features
    and next-week target, apply min-games filter, optionally write CSV.
    """
    # Load weekly stats and player metadata
    weekly = data_loader.load_player_weekly_stats(seasons)
    players = data_loader.load_players()

    # Normalize column names from nflreadpy (receptions, receiving_yards, receiving_tds) to our names
    weekly = _normalize_receiving_columns(weekly)

    # Filter to position (WR): join with players if position not in weekly
    player_col = _find_first_existing(weekly.columns, ["player_id", "gsis_id"])
    if player_col is None:
        raise ValueError("Weekly stats must have player_id or gsis_id.")
    if "position" not in weekly.columns and not players.is_empty():
        pos_col = _find_first_existing(players.columns, ["position", "pos"])
        id_in_players = _find_first_existing(players.columns, ["player_id", "gsis_id"])
        if pos_col and id_in_players:
            players = players.select([id_in_players, pos_col]).unique()
            if pos_col != "position":
                players = players.rename({pos_col: "position"})
            weekly = weekly.join(
                players,
                left_on=player_col,
                right_on=id_in_players,
                how="inner",
            )
    if "position" in weekly.columns:
        weekly = weekly.filter(pl.col("position") == position)

    # PPR fantasy points, then target and drop rows without target
    weekly = features.add_basic_fantasy_points(weekly)
    weekly = add_next_week_target(weekly)
    weekly = drop_rows_without_target(weekly)

    # Optional min games per player per season.
    # This can be used to restrict the training set, but is disabled
    # when min_games_for_player is 0 or None so that we avoid
    # discarding potentially informative rows.
    season_col = _find_first_existing(weekly.columns, ["season"])
    if season_col and min_games_for_player and min_games_for_player > 0:
        game_counts = (
            weekly.group_by([player_col, season_col])
            .agg(pl.len().alias("_games"))
            .filter(pl.col("_games") >= min_games_for_player)
        )
        weekly = weekly.join(
            game_counts.select([player_col, season_col]),
            on=[player_col, season_col],
            how="inner",
        )

    # Lag and rolling features, then model columns
    weekly = features.add_lag_features(weekly)
    weekly = features.add_rolling_features(weekly)
    weekly = features.select_model_columns(weekly)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        weekly.write_csv(output_path)

    return weekly


def add_next_week_target(data_frame: pl.DataFrame) -> pl.DataFrame:
    """
    Create next week's fantasy output as the supervised-learning target.
    """
    player_col = _find_first_existing(data_frame.columns, ["player_id", "gsis_id", "player_name"])
    season_col = _find_first_existing(data_frame.columns, ["season"])
    week_col = _find_first_existing(data_frame.columns, ["week"])

    if not all([player_col, season_col, week_col]):
        raise ValueError("Missing player/season/week columns needed for target creation.")

    if "ppr_points" not in data_frame.columns:
        raise ValueError("ppr_points must exist before creating the target.")

    return (
        data_frame.sort([player_col, season_col, week_col])
        .with_columns(
            pl.col("ppr_points")
            .shift(-1)
            .over(player_col)
            .alias("next_week_ppr_points")
        )
    )


def drop_rows_without_target(data_frame: pl.DataFrame) -> pl.DataFrame:
    return data_frame.filter(pl.col("next_week_ppr_points").is_not_null())


def _normalize_receiving_columns(data_frame: pl.DataFrame) -> pl.DataFrame:
    """Rename nflreadpy receiving columns to rec, rec_yds, rec_td if present."""
    renames = {}
    if "receptions" in data_frame.columns and "rec" not in data_frame.columns:
        renames["receptions"] = "rec"
    if "receiving_yards" in data_frame.columns and "rec_yds" not in data_frame.columns:
        renames["receiving_yards"] = "rec_yds"
    if "receiving_tds" in data_frame.columns and "rec_td" not in data_frame.columns:
        renames["receiving_tds"] = "rec_td"
    if renames:
        return data_frame.rename(renames)
    return data_frame


def _find_first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None