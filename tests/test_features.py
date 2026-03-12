import polars as pl

from src.wr_predictor.features import add_basic_fantasy_points


def test_add_basic_fantasy_points() -> None:
    df = pl.DataFrame(
        {
            "rec": [5],
            "rec_yds": [80],
            "rec_td": [1],
        }
    )

    result = add_basic_fantasy_points(df)
    assert "ppr_points" in result.columns
    assert result["ppr_points"][0] == 19.0