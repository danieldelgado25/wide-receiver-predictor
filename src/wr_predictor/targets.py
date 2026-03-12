from __future__ import annotations

import polars as pl


def add_next_week_target(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create next week's fantasy output as the supervised-learning target.
    """
    player_col = _find_first_existing(df.columns, ["player_id", "gsis_id", "player_name"])
    season_col = _find_first_existing(df.columns, ["season"])
    week_col = _find_first_existing(df.columns, ["week"])

    if not all([player_col, season_col, week_col]):
        raise ValueError("Missing player/season/week columns needed for target creation.")

    if "ppr_points" not in df.columns:
        raise ValueError("ppr_points must exist before creating the target.")

    return (
        df.sort([player_col, season_col, week_col])
        .with_columns(
            pl.col("ppr_points")
            .shift(-1)
            .over(player_col)
            .alias("next_week_ppr_points")
        )
    )


def drop_rows_without_target(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("next_week_ppr_points").is_not_null())


def _find_first_existing(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None