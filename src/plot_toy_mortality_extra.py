from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.config import settings
from src.io_utils import ensure_dirs_exist


def main() -> None:
    ensure_dirs_exist([settings.output_dir])

    path = settings.processed_dir / "toy_mortality_uk.csv"
    df = pd.read_csv(path)

    ages_to_plot = [55, 65, 75, 85]

    # -----------------------------
    # Plot 1: m_x over time
    # -----------------------------
    for age in ages_to_plot:
        sub = df[df["age"] == age].sort_values("year")
        plt.plot(sub["year"], sub["mx"], label=f"age {age}")

    plt.title("Toy mortality: m_x over time")
    plt.xlabel("Year")
    plt.ylabel("m_x")
    plt.legend()
    out1 = settings.output_dir / "toy_mortality_mx_over_time.png"
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # Plot 2: improvement (delta log m_x)
    # delta log m_{x,t} = log m_{x,t} - log m_{x,t-1}
    # Negative values = improving mortality (mortality falling).
    # -----------------------------
    for age in ages_to_plot:
        sub = df[df["age"] == age].sort_values("year").copy()
        sub["delta_log_mx"] = sub["log_mx"].diff()
        plt.plot(sub["year"], sub["delta_log_mx"], label=f"age {age}")

    plt.axhline(0.0)  # reference line
    plt.title("Toy mortality: year-on-year improvement (Δ log m_x)")
    plt.xlabel("Year")
    plt.ylabel("Δ log m_x")
    plt.legend()
    out2 = settings.output_dir / "toy_mortality_improvement_delta_logmx.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved plot:", out1)
    print("Saved plot:", out2)


if __name__ == "__main__":
    main()
