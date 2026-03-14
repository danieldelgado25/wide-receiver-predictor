import sys

from src.wr_predictor.dataset_builder import build_training_dataset


def main() -> None:
    """
    Script to build the training dataset.
    """
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    training_data_frame = build_training_dataset(
        seasons=[2021, 2022, 2023, 2024],
        min_games_for_player=0,
        output_path="data/processed/wr_training_dataset_2021-2024.csv",
    )

    """
    Evaluation / Prediction data 2025 season
    Same filters as training data, but only 2025 season
    """
    evaluation_2025_data_frame = build_training_dataset(
        seasons=[2025],
        min_games_for_player=0,
        output_path="data/processed/wr_evaluation_dataset_2025.csv",
    )

    print(training_data_frame.head())
    print(f"\nRows: {training_data_frame.shape[0]}")
    print(f"Cols: {training_data_frame.shape[1]}")

    print(evaluation_2025_data_frame.head())
    print(f"\nRows: {evaluation_2025_data_frame.shape[0]}")
    print(f"Cols: {evaluation_2025_data_frame.shape[1]}")


if __name__ == "__main__":
    main()