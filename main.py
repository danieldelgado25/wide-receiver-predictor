from src.wr_predictor.dataset_builder import build_training_dataset


def main() -> None:
    data_frame = build_training_dataset(
        seasons=[2022, 2023, 2024],
        min_games_for_player=3,
        output_path="data/processed/wr_training_dataset.csv",
    )
    print(data_frame.head())
    print(f"\nRows: {data_frame.shape[0]}")
    print(f"Cols: {data_frame.shape[1]}")


if __name__ == "__main__":
    main()