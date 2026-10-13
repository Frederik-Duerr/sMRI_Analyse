from pathlib import Path
import pandas as pd

def concat_feature_frames(path,):
    feature_dir = Path(
        path
    )

    parquet_files = list(feature_dir.glob("*.parquet"))

    all_features = pd.concat(
        [pd.read_parquet(file) for file in parquet_files],
        ignore_index=True
    )

    output_file = feature_dir / "all_features.parquet"
    all_features.to_parquet(output_file, index=False)

    print(f"Number of Files: {len(parquet_files)}")
    print(f"Whole Dimensions: {all_features.shape}")
    print(f"Saved under: {output_file}")
    return output_file