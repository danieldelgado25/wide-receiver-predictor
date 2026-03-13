from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    """
    Defines project configuration dataclass with default values.
    """
    seasons: list[int] = field(default_factory=lambda: [2021, 2022, 2023, 2024])
    position: str = "WR"
    min_games_for_player: int = 4
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    target_col: str = "next_week_ppr_points"