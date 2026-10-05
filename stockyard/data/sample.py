"""Public synthetic scenarios. Row order is bottom-to-top within each source."""

import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
from stockyard.data.generator import Plate


def generate(seed=42, sources=4, destinations=4, plates_per_source=6, dynamic=False):
    if not 1 <= sources <= 30 or not 1 <= destinations <= 30 or plates_per_source < 1:
        raise ValueError("Use 1–30 piles and at least one plate per source")
    rng = random.Random(seed)
    rows = []
    for i in range(sources):
        for j in range(plates_per_source):
            arrival = rng.randint(1, 12) if dynamic and j % 3 == 0 else 1
            rows.append(
                dict(
                    pileno=f"SYN_S{i:02d}",
                    pileseq=j + 1,
                    markno=f"SYN_{i:02d}_{j:03d}",
                    unitw=round(rng.uniform(1, 10), 2),
                    inbound=arrival,
                    outbound=arrival + rng.randint(1, 60),
                    topile=f"SYN_D{(i + j) % destinations:02d}",
                )
            )
    return pd.DataFrame(rows)


def load(path):
    return validate(
        pd.read_csv(path, dtype={"pileno": str, "markno": str, "topile": str})
    )


def validate(df):
    """Check the public schema before sorting or converting fields to plates."""
    df = df.copy()
    required = {"pileno", "pileseq", "markno", "unitw", "inbound", "outbound", "topile"}
    if (
        not required.issubset(df.columns)
        or df.empty
        or df[list(required)].isna().any().any()
    ):
        raise ValueError("Missing or empty scenario fields")
    for column in ["pileno", "markno", "topile"]:
        df[column] = df[column].astype(str).str.strip()
        if df[column].eq("").any():
            raise ValueError(f"{column} must contain nonempty identifiers")
    for column in ["pileseq", "inbound", "outbound", "unitw"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(df[column]).all():
            raise ValueError(f"{column} must contain finite numeric values")
    for column in ["pileseq", "inbound", "outbound"]:
        if (df[column] % 1 != 0).any():
            raise ValueError(f"{column} must contain integer values")
        df[column] = df[column].astype("int64")
    if (df.pileseq < 1).any() or (df.inbound < 0).any() or (df.unitw < 0).any():
        raise ValueError("Require positive pile sequences and nonnegative days/weights")
    if df.duplicated(["pileno", "pileseq"]).any():
        raise ValueError("Duplicate pile sequence within a source")
    if df.markno.duplicated().any() or (df.outbound < df.inbound).any():
        raise ValueError("Duplicate plate IDs or outbound before inbound")
    if df.pileno.nunique() > 30 or df.topile.nunique() > 30:
        raise ValueError("Model supports at most 30 source and 30 destination piles")
    if "confirm_time" in df and df.confirm_time.notna().any():
        raise ValueError(
            "Scheduled outbound-date confirmations are not applied by the public environment"
        )
    # Experimental metadata must not change one baseline's capacity or obstacles.
    return (
        df[sorted(required)].sort_values(["pileno", "pileseq"]).reset_index(drop=True)
    )


def to_plates(df):
    plates = []
    for row in df.to_dict("records"):
        p = Plate(
            row["markno"],
            int(row["inbound"]),
            int(row["outbound"]),
            float(row["unitw"]),
            row.get("planned_outbound"),
            row.get("confirmed_outbound"),
            row.get("confirm_time"),
        )
        p.from_pile = str(row["pileno"])
        p.topile = str(row["topile"])
        plates.append(p)
    return plates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="data/sample/scenario.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sources", type=int, default=4)
    p.add_argument("--destinations", type=int, default=4)
    p.add_argument("--plates-per-source", type=int, default=6)
    p.add_argument("--dynamic", action="store_true")
    a = p.parse_args()
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    generate(a.seed, a.sources, a.destinations, a.plates_per_source, a.dynamic).to_csv(
        dest, index=False
    )
    print(f"Saved synthetic scenario: {dest}")


if __name__ == "__main__":
    main()
