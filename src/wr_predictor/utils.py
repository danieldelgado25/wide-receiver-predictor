from __future__ import annotations

import polars as pl


def print_shape_and_columns(df: pl.DataFrame, label: str = "DataFrame") -> None:
    print(f"{label} shape: {df.shape}")
    print(f"{label} columns:")
    for col in df.columns:
        print(f"  - {col}")