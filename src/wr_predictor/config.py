from dataclasses import dataclass, field

# DEPRECATED: not currently imported/used anywhere. dataset_builder.build_training_dataset()
# owns the live defaults (seasons, position, min_games_for_player) via its own kwargs instead.
# Revisit wiring this in once there's a second consumer (CLI, web endpoint) that needs the
# same defaults, rather than duplicating them by hand. Flagged for manual removal if still
# unused when that day comes.


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