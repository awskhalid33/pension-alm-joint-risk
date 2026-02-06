from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import settings
from src.io_utils import ensure_dirs_exist
from src.datasets.toy_mortality import generate_toy_mortality


def main() -> None:
    ensure_dirs_exist([settings.processed_dir])

    df = generate_toy_mortality(
        ages=range(50, 111),
        years=range(settings.start_year, settings.end_year + 1),
        seed=123,
    )

    out_path: Path = settings.processed_dir / "toy_mortality_uk.csv"
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print("Rows:", len(df))
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
