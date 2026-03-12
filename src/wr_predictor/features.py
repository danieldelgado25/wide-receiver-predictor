from __future__ import annotations

import polars as pl


def add_basic_fantasy_points(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add PPR fantasy points if enough receiving columns exist.
    """
    required = ["rec", "rec_yds", "rec_td"]
    if not all(col in df.columns for col in required):
        return df

    return df.with_columns(
        (
            pl.col("rec") * 1.0
            + pl.col("rec_yds") * 0.1
            + pl.col("rec_td") * 6.0
        ).alias("ppr_points")
    )


def add_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add previous-week features per player.
    """
    player_col = _find_first_existing(df.columns, ["player_id", "gsis_id", "player_name"])
    season_col = _find_first_existing(df.columns, ["season"])
    week_col = _find_first_existing(df.columns, ["week"])

    if not all([player_col, season_col, week_col]):
        raise ValueError("Missing player/season/week columns needed for lag features.")

    sort_cols = [player_col, season_col, week_col]
    df = df.sort(sort_cols)

    lag_candidates = ["ppr_points", "targets", "rec", "rec_yds", "rec_td"]

    expressions = []
    for col in lag_candidates:
        if col in df.columns:
            expressions.append(
                pl.col(col)
                .shift(1)
                .over(player_col)
                .alias(f"{col}_prev_week")
            )

    return df.with_columns(expressions)


def add_rolling_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add simple trailing rolling means.
    """
    player_col = _find_first_existing(df.columns, ["player_id", "gsis_id", "player_name"])
    if player_col is None:
        raise ValueError("Missing player identifier for rolling features.")

    rolling_candidates = ["ppr_points", "targets", "rec_yds"]

    expressions = []
    for col in rolling_candidates:
        if col in df.columns:
            expressions.append(
                pl.col(col)
                .shift(1)
                .rolling_mean(window_size=3, min_samples=1)
                .over(player_col)
                .alias(f"{col}_rolling_3")
            )
            expressions.append(
                pl.col(col)
                .shift(1)
                .rolling_mean(window_size=5, min_samples=1)
                .over(player_col)
                .alias(f"{col}_rolling_5")
            )

    return df.with_columns(expressions)


def select_model_columns(df: pl.DataFrame) -> pl.DataFrame:
    keep_if_exists = [
        "player_id",
        "player_name",
        "season",
        "week",
        "team",
        "opponent_team",
        "home_away",
        "ppr_points",
        "ppr_points_prev_week",
        "targets_prev_week",
        "rec_prev_week",
        "rec_yds_prev_week",
        "rec_td_prev_week",
        "ppr_points_rolling_3",
        "ppr_points_rolling_5",
        "targets_rolling_3",
        "targets_rolling_5",
        "rec_yds_rolling_3",
        "rec_yds_rolling_5",
        "next_week_ppr_points",
    ]
    keep = [col for col in keep_if_exists if col in df.columns]
    return df.select(keep)


def _find_first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None