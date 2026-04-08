"""
LEVEL 1: Data Loading, Parsing, Cleaning, and Sanity Gate
No analysis or model is allowed before passing the Data Sanity Gate.
"""
import pandas as pd
import numpy as np
import sys


# ── 1.1 Load Raw Data ──
def load_data(filepath: str, sep: str = ";", decimal: str = ",") -> pd.DataFrame:
    """
    Load steady-state simulation data (e.g., from Aspen).
    Handles European CSV formats (semicolon separator, comma decimal).
    """
    df = pd.read_csv(filepath, sep=sep, decimal=decimal)
    print(f"[DATA] Loaded {len(df)} rows, columns: {list(df.columns)}")
    return df


# ── 1.2 Parsing & Cleaning ──
REQUIRED_COLUMNS = ["HeatDuty", "XD", "TTOP", "TMID", "TBOTTOM"]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Verify required columns exist
    - Convert all to float
    - Drop NaN rows
    - Normalize units (e.g., W → kW if needed)
    """
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"[DATA ERROR] Missing column: {col}")

    df = df[REQUIRED_COLUMNS].copy()

    # Force numeric conversion
    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    df.dropna(inplace=True)
    n_after = len(df)
    if n_before != n_after:
        print(f"[DATA] Dropped {n_before - n_after} rows with NaN/non-numeric values")

    # Unit normalization example: if HeatDuty is in W, convert to kW
    # df["HeatDuty"] = df["HeatDuty"] / 1000.0  # Uncomment if needed

    df.reset_index(drop=True, inplace=True)
    return df


# ── 1.3 Data Sanity Gate ──
def data_sanity_gate(df: pd.DataFrame,
                     min_rows: int = 20,
                     heat_duty_range: tuple = (0.1, 1e6)) -> bool:
    """
    Hard gate: If this fails, the ENTIRE project stops.
    """
    errors = []

    if len(df) < min_rows:
        errors.append(f"Too few rows: {len(df)} < {min_rows}")

    hd_min, hd_max = df["HeatDuty"].min(), df["HeatDuty"].max()
    if hd_min < heat_duty_range[0] or hd_max > heat_duty_range[1]:
        errors.append(f"HeatDuty range [{hd_min}, {hd_max}] outside expected {heat_duty_range}")

    if hd_min == hd_max:
        errors.append("HeatDuty has zero variance — cannot build model")

    if not (0.0 <= df["XD"].min() and df["XD"].max() <= 1.0):
        errors.append(f"XD out of [0,1]: [{df['XD'].min()}, {df['XD'].max()}]")

    if errors:
        print("🚫 DATA SANITY GATE FAILED:")
        for e in errors:
            print(f"   ✗ {e}")
        sys.exit(1)

    print("✅ Data Sanity Gate PASSED")
    print(f"   Rows: {len(df)} | HeatDuty: [{hd_min:.2f}, {hd_max:.2f}]")
    print(f"   XD: [{df['XD'].min():.4f}, {df['XD'].max():.4f}]")
    return True