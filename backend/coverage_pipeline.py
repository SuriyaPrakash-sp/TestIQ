"""
Coverage Optimization Pipeline  v2.0
======================================
Schema-native pipeline for:
    Seed, Bin_Name, Hits, Total, Coverage_Before, Coverage_After,
    Bin_Weight, Times_Executed

Column semantics
────────────────
  Seed             - unique seed identifier
  Bin_Name         - coverage bin name / ID
  Hits             - times this bin was successfully triggered in simulation
  Total            - total simulation attempts for this bin
  Coverage_Before  - hit-rate before seeding  (≈ Hits / Times_Executed)
  Coverage_After   - hit-rate after seeding
  Bin_Weight       - business importance / criticality of this bin (higher = more critical)
  Times_Executed   - how many times this seed was attempted on this bin

Derived critical metrics
────────────────────────
  Improvement          = Coverage_After - Coverage_Before
  Coverage_Gap         = 1 - Coverage_Before
  Weighted_Priority    = Coverage_Gap * Bin_Weight          ← drives ranking
  Hit_Rate_Consistency = Hits / Times_Executed              ← validates Coverage_Before
  Execution_Efficiency = Improvement / Times_Executed       ← improvement per attempt

Usage
─────
  python coverage_pipeline.py                          # built-in sample data
  python coverage_pipeline.py --input data.csv         # your CSV
  python coverage_pipeline.py --input data.csv --output results.json
  python coverage_pipeline.py --input data.csv --bin BIN_A BIN_B --top-seeds 8
  python coverage_pipeline.py --threshold 0.70         # custom low-coverage cutoff
  python coverage_pipeline.py --max-bins-output 5 --max-seeds-per-bin 3  # limit output size

Requirements
────────────
  pip install scikit-learn pandas numpy scipy xgboost
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("pipeline")


# ══════════════════════════════════════════════════════════════
# 0.  CONSTANTS & CONFIG
# ══════════════════════════════════════════════════════════════

REQUIRED_COLUMNS = [
    "Seed", "Bin_Name", "Hits", "Total",
    "Coverage_Before", "Coverage_After",
    "Bin_Weight", "Times_Executed",
]

LOW_COVERAGE_THRESHOLD     = 0.60   # Coverage_Before < 60 % → flagged as gap
HIGH_IMPROVEMENT_THRESHOLD = 0.05   # avg Improvement > 5 % → "strong" seed
TOP_N_BINS                 = 10
TOP_N_SEEDS                = 5
RANDOM_STATE               = 42


# ══════════════════════════════════════════════════════════════
# 1.  RESULT DATA CLASSES
# ══════════════════════════════════════════════════════════════

@dataclass
class SeedRecommendation:
    seed: str
    predicted_improvement: float
    predicted_coverage_after: float
    weighted_score: float       # improvement × bin_weight — the "real" value
    confidence_score: float     # 0–1 derived from CV variance + execution count
    rank: int


@dataclass
class BinAnalysis:
    bin_name: str
    bin_weight: float
    coverage_before: float
    coverage_gap: float
    weighted_priority: float        # Coverage_Gap × Bin_Weight
    avg_hits: float
    avg_times_executed: float
    hit_rate_consistency: float     # Hits / Times_Executed alignment
    best_historical_seed: str
    historical_improvement: float
    recommendations: list[SeedRecommendation] = field(default_factory=list)
    simulated_coverage_after: float = 0.0
    simulated_new_hits: float = 0.0  # projected Hits after best seed applied


@dataclass
class PipelineResult:
    # Summary
    total_bins: int
    total_seeds: int
    total_records: int
    avg_coverage_before: float
    avg_coverage_after: float
    avg_improvement: float
    total_hits: int
    total_executions: int
    avg_bin_weight: float
    # Gap analysis
    low_coverage_bins: list[dict]
    weakest_bins_by_weighted_priority: list[dict]
    # Seed intelligence
    best_seeds_global: list[dict]
    seed_bin_matrix_sample: list[dict]
    # Model
    model_name: str
    model_metrics: dict[str, float]
    feature_importances: dict[str, float]
    # Recommendations
    bin_recommendations: list[dict]
    # Meta
    pipeline_version: str = "2.0.0"


# ══════════════════════════════════════════════════════════════
# 2.  SAMPLE DATA GENERATOR
# ══════════════════════════════════════════════════════════════

def generate_sample_data(
    n_seeds: int = 20,
    n_bins: int = 30,
    n_records: int = 1500,
) -> pd.DataFrame:
    """
    Realistic synthetic data matching the exact 8-column schema.
    Bin_Weight is log-normal to simulate real criticality skew.
    """
    log.info("Generating synthetic sample dataset (%d records)…", n_records)
    rng = np.random.default_rng(RANDOM_STATE)

    seeds = [f"SEED_{i:03d}" for i in range(1, n_seeds + 1)]
    bins  = [f"BIN_{i:03d}"  for i in range(1, n_bins + 1)]

    seed_potency    = rng.uniform(0.02, 0.18, n_seeds)
    bin_base_cov    = rng.uniform(0.25, 0.85, n_bins)
    bin_weights_raw = np.exp(rng.normal(0, 0.8, n_bins))
    bin_weights     = np.round(bin_weights_raw / bin_weights_raw.max() * 10, 2)

    rows = []
    for _ in range(n_records):
        s_idx = rng.integers(0, n_seeds)
        b_idx = rng.integers(0, n_bins)

        times_executed = int(rng.integers(50, 500))
        cb = float(np.clip(rng.normal(bin_base_cov[b_idx], 0.06), 0.05, 0.99))
        hits = int(round(cb * times_executed))
        lift = float(np.clip(rng.normal(seed_potency[s_idx], 0.025), -0.05, 0.40))
        ca   = float(np.clip(cb + lift, 0.0, 1.0))
        total = int(rng.integers(times_executed, times_executed * 2))

        rows.append({
            "Seed":           seeds[s_idx],
            "Bin_Name":       bins[b_idx],
            "Hits":           hits,
            "Total":          total,
            "Coverage_Before": round(cb, 4),
            "Coverage_After":  round(ca, 4),
            "Bin_Weight":     float(bin_weights[b_idx]),
            "Times_Executed": times_executed,
        })

    df = pd.DataFrame(rows)

    # Inject ~3 % missing + outliers to test cleaning
    for col in ["Coverage_Before", "Coverage_After", "Hits", "Times_Executed"]:
        mask = rng.random(n_records) < 0.03
        df.loc[mask, col] = np.nan
    df.loc[rng.choice(n_records, 8, replace=False), "Coverage_Before"] = -0.3
    df.loc[rng.choice(n_records, 8, replace=False), "Coverage_After"]  =  1.5
    df.loc[rng.choice(n_records, 5, replace=False), "Bin_Weight"]      = -1.0

    log.info("Sample dataset ready: %d rows × %d columns", *df.shape)
    return df


# ══════════════════════════════════════════════════════════════
# 3.  LOAD
# ══════════════════════════════════════════════════════════════

def load_dataset(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return generate_sample_data()

    path = Path(path)
    log.info("Loading dataset from %s …", path)
    loaders = {
        ".csv":     pd.read_csv,
        ".xlsx":    pd.read_excel,
        ".xls":     pd.read_excel,
        ".json":    pd.read_json,
        ".parquet": pd.read_parquet,
    }
    suffix = path.suffix.lower()
    if suffix not in loaders:
        raise ValueError(f"Unsupported format '{suffix}'. Use CSV, Excel, JSON, or Parquet.")
    df = loaders[suffix](path)
    log.info("Loaded %d rows × %d columns", *df.shape)
    return df


# ══════════════════════════════════════════════════════════════
# 4.  VALIDATE
# ══════════════════════════════════════════════════════════════

def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Returns validation messages. Fatal errors start with [FATAL]."""
    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"[FATAL] Missing required columns: {missing}")
        return errors
    if df.empty:
        errors.append("[FATAL] Dataset is empty.")
        return errors

    for col in ["Coverage_Before", "Coverage_After"]:
        n_bad = ((df[col].dropna() < 0) | (df[col].dropna() > 1)).sum()
        if n_bad:
            errors.append(f"{col}: {n_bad} values outside [0,1] → will be clipped.")

    if (df["Bin_Weight"].dropna() < 0).any():
        errors.append("Bin_Weight: negative values detected → will be set to 0.")
    if (df["Hits"].dropna() < 0).any():
        errors.append("Hits: negative values → will be clipped to 0.")
    if (df["Times_Executed"].dropna() <= 0).any():
        errors.append("Times_Executed: zero/negative values → will be clipped to 1.")

    valid = df[["Hits", "Times_Executed"]].dropna()
    inconsistent = (valid["Hits"] > valid["Times_Executed"]).sum()
    if inconsistent:
        errors.append(
            f"Hits > Times_Executed in {inconsistent} rows (Hits will be capped)."
        )

    for col, cnt in df[REQUIRED_COLUMNS].isnull().sum().items():
        if cnt:
            errors.append(f"{col}: {cnt} nulls ({100*cnt/len(df):.1f}%) → will be imputed.")

    dup = df.duplicated().sum()
    if dup:
        errors.append(f"{dup} fully duplicate rows → will be dropped.")

    return errors


# ══════════════════════════════════════════════════════════════
# 5.  CLEAN
# ══════════════════════════════════════════════════════════════

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning dataset…")
    n0 = len(df)
    df = df.drop_duplicates().copy()
    log.info("  Dropped %d duplicate rows", n0 - len(df))

    for col in ["Coverage_Before", "Coverage_After"]:
        bad = ((df[col] < 0) | (df[col] > 1)).sum()
        df[col] = df[col].clip(0.0, 1.0)
        if bad:
            log.info("  Clipped %d out-of-range values in %s", bad, col)

    bad_w = (df["Bin_Weight"] < 0).sum()
    df["Bin_Weight"] = df["Bin_Weight"].clip(lower=0.0)
    if bad_w:
        log.info("  Fixed %d negative Bin_Weight values → 0", bad_w)

    df["Times_Executed"] = df["Times_Executed"].clip(lower=1)
    df["Hits"] = df["Hits"].clip(lower=0)
    overcapped = (df["Hits"] > df["Times_Executed"]).sum()
    df["Hits"] = df[["Hits", "Times_Executed"]].min(axis=1)
    if overcapped:
        log.info("  Capped Hits ≤ Times_Executed in %d rows", overcapped)

    # Impute nulls: per-bin median, then global median
    for col in ["Coverage_Before", "Coverage_After", "Bin_Weight",
                "Hits", "Total", "Times_Executed"]:
        if df[col].isnull().any():
            bin_med    = df.groupby("Bin_Name")[col].transform("median")
            global_med = df[col].median()
            df[col]    = df[col].fillna(bin_med).fillna(global_med)

    for col in ["Seed", "Bin_Name"]:
        df[col] = df[col].astype(str).str.strip().str.upper()

    log.info("  Clean dataset: %d rows", len(df))
    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# 6.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Engineering features…")

    # ── Core derived metrics (critical) ─────────────────────
    df["Improvement"]          = df["Coverage_After"] - df["Coverage_Before"]
    df["Coverage_Gap"]         = 1.0 - df["Coverage_Before"]
    df["Weighted_Priority"]    = df["Coverage_Gap"] * df["Bin_Weight"]
    df["Hit_Rate_Computed"]    = (df["Hits"] / df["Times_Executed"]).clip(0.0, 1.0)
    df["Hit_Rate_Consistency"] = 1.0 - (df["Coverage_Before"] - df["Hit_Rate_Computed"]).abs()
    df["Execution_Efficiency"] = df["Improvement"] / df["Times_Executed"].clip(lower=1)
    df["Relative_Gain"]        = np.where(
        df["Coverage_Before"] > 0,
        df["Improvement"] / df["Coverage_Before"],
        0.0,
    )
    df["Weighted_Improvement"] = df["Improvement"] * df["Bin_Weight"]
    df["Total_Hit_Rate"]       = (df["Hits"] / df["Total"].clip(lower=1)).clip(0.0, 1.0)

    # ── Bin-level aggregates ─────────────────────────────────
    bin_stats = df.groupby("Bin_Name").agg(
        Bin_Avg_Coverage    = ("Coverage_Before",    "mean"),
        Bin_Std_Coverage    = ("Coverage_Before",    "std"),
        Bin_Avg_Weight      = ("Bin_Weight",         "mean"),
        Bin_Total_Hits      = ("Hits",               "sum"),
        Bin_Total_Exec      = ("Times_Executed",     "sum"),
        Bin_Record_Count    = ("Improvement",        "count"),
        Bin_Avg_Improvement = ("Improvement",        "mean"),
    ).reset_index()
    bin_stats["Bin_Std_Coverage"] = bin_stats["Bin_Std_Coverage"].fillna(0)
    df = df.merge(bin_stats, on="Bin_Name", how="left")

    # ── Seed-level aggregates ────────────────────────────────
    seed_stats = df.groupby("Seed").agg(
        Seed_Avg_Improvement     = ("Improvement",          "mean"),
        Seed_Std_Improvement     = ("Improvement",          "std"),
        Seed_Max_Improvement     = ("Improvement",          "max"),
        Seed_Run_Count           = ("Improvement",          "count"),
        Seed_Avg_Weighted_Imp    = ("Weighted_Improvement", "mean"),
        Seed_Avg_Exec_Efficiency = ("Execution_Efficiency", "mean"),
        Seed_Total_Hits          = ("Hits",                 "sum"),
        Seed_Total_Exec          = ("Times_Executed",       "sum"),
        Seed_Bins_Covered        = ("Bin_Name",             "nunique"),
    ).reset_index()
    seed_stats["Seed_Std_Improvement"] = seed_stats["Seed_Std_Improvement"].fillna(0)
    df = df.merge(seed_stats, on="Seed", how="left")

    # ── Seed × Bin interaction features ─────────────────────
    sb_stats = df.groupby(["Seed", "Bin_Name"]).agg(
        SeedBin_Avg_Improvement = ("Improvement",          "mean"),
        SeedBin_Count           = ("Improvement",          "count"),
        SeedBin_Avg_Hits        = ("Hits",                 "mean"),
        SeedBin_Avg_Exec        = ("Times_Executed",       "mean"),
        SeedBin_Weighted_Imp    = ("Weighted_Improvement", "mean"),
    ).reset_index()
    df = df.merge(sb_stats, on=["Seed", "Bin_Name"], how="left")

    # ── Label-encode categoricals ────────────────────────────
    for col in ["Seed", "Bin_Name"]:
        le = LabelEncoder()
        df[f"{col}_Encoded"] = le.fit_transform(df[col].astype(str))

    log.info("  Feature engineering complete → %d columns", df.shape[1])
    return df


# ══════════════════════════════════════════════════════════════
# 7.  STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════

def analyse_coverage(df: pd.DataFrame) -> dict[str, Any]:
    log.info("Running statistical analysis…")

    summary = {
        "total_records":            len(df),
        "total_bins":               df["Bin_Name"].nunique(),
        "total_seeds":              df["Seed"].nunique(),
        "avg_coverage_before":      round(df["Coverage_Before"].mean(),       4),
        "avg_coverage_after":       round(df["Coverage_After"].mean(),        4),
        "avg_improvement":          round(df["Improvement"].mean(),           4),
        "std_improvement":          round(df["Improvement"].std(),            4),
        "total_hits":               int(df["Hits"].sum()),
        "total_executions":         int(df["Times_Executed"].sum()),
        "avg_bin_weight":           round(df["Bin_Weight"].mean(),            4),
        "avg_weighted_priority":    round(df["Weighted_Priority"].mean(),     4),
        "pct_improved":             round((df["Improvement"] > 0).mean() * 100, 2),
        "avg_exec_efficiency":      round(df["Execution_Efficiency"].mean(),  6),
        "avg_hit_rate_consistency": round(df["Hit_Rate_Consistency"].mean(),  4),
    }

    bin_agg = df.groupby("Bin_Name").agg(
        Avg_Coverage_Before   = ("Coverage_Before",     "mean"),
        Avg_Coverage_After    = ("Coverage_After",      "mean"),
        Avg_Improvement       = ("Improvement",         "mean"),
        Avg_Coverage_Gap      = ("Coverage_Gap",        "mean"),
        Avg_Bin_Weight        = ("Bin_Weight",          "mean"),
        Avg_Weighted_Priority = ("Weighted_Priority",   "mean"),
        Total_Hits            = ("Hits",                "sum"),
        Total_Executions      = ("Times_Executed",      "sum"),
        Record_Count          = ("Improvement",         "count"),
        Avg_Exec_Efficiency   = ("Execution_Efficiency","mean"),
    ).reset_index()

    # Low-coverage bins: below threshold, sorted by worst first
    low_cov_bins = (
        bin_agg[bin_agg["Avg_Coverage_Before"] < LOW_COVERAGE_THRESHOLD]
        .sort_values("Avg_Coverage_Before")
        .head(TOP_N_BINS)
    )

    # High-priority bins: sorted by Weighted_Priority (gap × importance)
    weakest_bins = (
        bin_agg.sort_values("Avg_Weighted_Priority", ascending=False)
        .head(TOP_N_BINS)
    )

    # Seed intelligence
    seed_agg = df.groupby("Seed").agg(
        Avg_Improvement     = ("Improvement",          "mean"),
        Std_Improvement     = ("Improvement",          "std"),
        Max_Improvement     = ("Improvement",          "max"),
        Run_Count           = ("Improvement",          "count"),
        Avg_Weighted_Imp    = ("Weighted_Improvement", "mean"),
        Avg_Exec_Efficiency = ("Execution_Efficiency", "mean"),
        Bins_Covered        = ("Bin_Name",             "nunique"),
        Total_Hits          = ("Hits",                 "sum"),
        Total_Exec          = ("Times_Executed",       "sum"),
    ).reset_index()
    seed_agg["Std_Improvement"]     = seed_agg["Std_Improvement"].fillna(0)
    seed_agg["Reliability"]         = (
        seed_agg["Avg_Improvement"] / (seed_agg["Std_Improvement"] + 1e-9)
    )
    seed_agg["Weighted_Reliability"] = (
        seed_agg["Avg_Weighted_Imp"] / (seed_agg["Std_Improvement"] + 1e-9)
    )

    best_seeds = (
        seed_agg[seed_agg["Run_Count"] >= 3]
        .sort_values("Weighted_Reliability", ascending=False)
        .head(TOP_N_BINS)
    )

    seed_bin = (
        df.groupby(["Seed", "Bin_Name"])["Improvement"]
        .mean().reset_index()
        .sort_values("Improvement", ascending=False)
        .head(50)
    )

    n_strong = (seed_agg["Avg_Improvement"] > HIGH_IMPROVEMENT_THRESHOLD).sum()
    log.info(
        "  Analysis: %d low-cov bins | %d high-priority (weighted) bins | %d strong seeds",
        len(low_cov_bins), len(weakest_bins), n_strong,
    )

    return {
        "summary":         summary,
        "bin_agg":         bin_agg,
        "low_cov_bins":    low_cov_bins,
        "weakest_bins":    weakest_bins,
        "seed_agg":        seed_agg,
        "best_seeds":      best_seeds,
        "seed_bin_matrix": seed_bin,
    }


# ══════════════════════════════════════════════════════════════
# 8.  MODEL TRAINING
# ══════════════════════════════════════════════════════════════

FEATURE_COLS = [
    # Categorical encodings
    "Seed_Encoded", "Bin_Name_Encoded",
    # Core schema inputs
    "Coverage_Before", "Coverage_Gap", "Bin_Weight", "Weighted_Priority",
    "Hits", "Total", "Times_Executed",
    # Derived from raw counts
    "Hit_Rate_Computed", "Hit_Rate_Consistency", "Total_Hit_Rate",
    # Bin-level context
    "Bin_Avg_Coverage", "Bin_Std_Coverage", "Bin_Avg_Weight",
    "Bin_Total_Hits", "Bin_Total_Exec", "Bin_Record_Count",
    # Seed-level intelligence
    "Seed_Avg_Improvement", "Seed_Std_Improvement", "Seed_Max_Improvement",
    "Seed_Run_Count", "Seed_Avg_Weighted_Imp", "Seed_Avg_Exec_Efficiency",
    "Seed_Total_Hits", "Seed_Total_Exec", "Seed_Bins_Covered",
    # Seed × Bin interaction
    "SeedBin_Avg_Improvement", "SeedBin_Count",
    "SeedBin_Avg_Hits", "SeedBin_Avg_Exec", "SeedBin_Weighted_Imp",
]
TARGET_COL = "Improvement"


def _available_features(df: pd.DataFrame) -> list[str]:
    return [c for c in FEATURE_COLS if c in df.columns]


def _try_xgboost():
    try:
        from xgboost import XGBRegressor  # type: ignore
        return XGBRegressor(
            n_estimators=500, learning_rate=0.04, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=3, gamma=0.1,
            reg_alpha=0.05, reg_lambda=1.0,
            random_state=RANDOM_STATE, verbosity=0,
        ), "XGBoostRegressor"
    except ImportError:
        return None, None


def build_model():
    model, name = _try_xgboost()
    if model:
        log.info("  Using XGBoostRegressor")
        return model, name
    log.info("  xgboost not installed → GradientBoostingRegressor (sklearn fallback)")
    return GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.04, max_depth=5,
        subsample=0.8, min_samples_leaf=5, random_state=RANDOM_STATE,
    ), "GradientBoostingRegressor"


def train_model(df: pd.DataFrame) -> tuple[Any, list[str], dict[str, float], str]:
    log.info("Training ML model…")
    features = _available_features(df)
    log.info("  Feature set: %d features", len(features))
    X = df[features].fillna(0)
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model, model_name = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics: dict[str, float] = {
        "MAE":  round(float(mean_absolute_error(y_test, y_pred)),         4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        "R2":   round(float(r2_score(y_test, y_pred)),                    4),
    }
    cv = cross_val_score(model, X, y, cv=5, scoring="r2")
    metrics["CV_R2_Mean"] = round(float(cv.mean()), 4)
    metrics["CV_R2_Std"]  = round(float(cv.std()),  4)

    log.info(
        "  %s | MAE=%.4f | RMSE=%.4f | R²=%.4f | CV_R²=%.4f±%.4f",
        model_name, metrics["MAE"], metrics["RMSE"], metrics["R2"],
        metrics["CV_R2_Mean"], metrics["CV_R2_Std"],
    )
    return model, features, metrics, model_name


# ══════════════════════════════════════════════════════════════
# 9.  RECOMMENDATION & SIMULATION ENGINE
# ══════════════════════════════════════════════════════════════

def recommend_seeds(
    target_bin: str,
    df: pd.DataFrame,
    model: Any,
    features: list[str],
    analysis: dict[str, Any],
    cv_std: float,
    top_n: int = TOP_N_SEEDS,
) -> BinAnalysis:
    """
    For a given bin:
      1. Build a feature row for every known seed.
      2. Predict Improvement with the trained model.
      3. Compute Weighted_Score = Predicted_Improvement × Bin_Weight.
      4. Rank seeds; build SeedRecommendation list.
      5. Simulate projected Hits after applying the top seed.
    """
    bin_data = df[df["Bin_Name"] == target_bin]
    if bin_data.empty:
        raise ValueError(f"Bin '{target_bin}' not found in dataset.")

    ref = bin_data.iloc[0]
    coverage_before     = float(bin_data["Coverage_Before"].mean())
    coverage_gap        = float(1.0 - coverage_before)
    bin_weight          = float(bin_data["Bin_Weight"].mean())
    weighted_priority   = round(coverage_gap * bin_weight, 4)
    avg_hits            = float(bin_data["Hits"].mean())
    avg_times_executed  = float(bin_data["Times_Executed"].mean())
    hit_rate_consistency = float(bin_data["Hit_Rate_Consistency"].mean())

    hist = (
        bin_data.groupby("Seed")["Improvement"]
        .mean().reset_index()
        .sort_values("Improvement", ascending=False)
    )
    best_hist_seed   = str(hist.iloc[0]["Seed"]) if not hist.empty else "N/A"
    hist_improvement = float(hist.iloc[0]["Improvement"]) if not hist.empty else 0.0

    rows: list[dict] = []
    for seed in df["Seed"].unique():
        sd   = df[df["Seed"] == seed]
        sbdf = df[(df["Seed"] == seed) & (df["Bin_Name"] == target_bin)]

        rows.append({
            "Seed":                     seed,
            "Seed_Encoded":             int(sd["Seed_Encoded"].iloc[0]) if len(sd) else 0,
            "Bin_Name_Encoded":         int(ref.get("Bin_Name_Encoded", 0)),
            "Coverage_Before":          coverage_before,
            "Coverage_Gap":             coverage_gap,
            "Bin_Weight":               bin_weight,
            "Weighted_Priority":        weighted_priority,
            "Hits":                     avg_hits,
            "Total":                    float(bin_data["Total"].mean()),
            "Times_Executed":           avg_times_executed,
            "Hit_Rate_Computed":        float(bin_data["Hit_Rate_Computed"].mean()),
            "Hit_Rate_Consistency":     hit_rate_consistency,
            "Total_Hit_Rate":           float(bin_data["Total_Hit_Rate"].mean()),
            "Bin_Avg_Coverage":         float(ref.get("Bin_Avg_Coverage",    coverage_before)),
            "Bin_Std_Coverage":         float(ref.get("Bin_Std_Coverage",    0.0)),
            "Bin_Avg_Weight":           float(ref.get("Bin_Avg_Weight",      bin_weight)),
            "Bin_Total_Hits":           float(ref.get("Bin_Total_Hits",      avg_hits)),
            "Bin_Total_Exec":           float(ref.get("Bin_Total_Exec",      avg_times_executed)),
            "Bin_Record_Count":         len(bin_data),
            "Seed_Avg_Improvement":     float(sd["Improvement"].mean())          if len(sd) else 0.0,
            "Seed_Std_Improvement":     float(sd["Improvement"].std())           if len(sd) > 1 else 0.0,
            "Seed_Max_Improvement":     float(sd["Improvement"].max())           if len(sd) else 0.0,
            "Seed_Run_Count":           len(sd),
            "Seed_Avg_Weighted_Imp":    float(sd["Weighted_Improvement"].mean()) if len(sd) else 0.0,
            "Seed_Avg_Exec_Efficiency": float(sd["Execution_Efficiency"].mean()) if len(sd) else 0.0,
            "Seed_Total_Hits":          float(sd["Hits"].sum())                  if len(sd) else 0.0,
            "Seed_Total_Exec":          float(sd["Times_Executed"].sum())        if len(sd) else 0.0,
            "Seed_Bins_Covered":        int(sd["Bin_Name"].nunique())            if len(sd) else 0,
            "SeedBin_Avg_Improvement":  float(sbdf["Improvement"].mean())        if len(sbdf) else 0.0,
            "SeedBin_Count":            len(sbdf),
            "SeedBin_Avg_Hits":         float(sbdf["Hits"].mean())               if len(sbdf) else avg_hits,
            "SeedBin_Avg_Exec":         float(sbdf["Times_Executed"].mean())     if len(sbdf) else avg_times_executed,
            "SeedBin_Weighted_Imp":     float(sbdf["Weighted_Improvement"].mean())if len(sbdf) else 0.0,
        })

    pred_df = pd.DataFrame(rows)
    avail   = [f for f in features if f in pred_df.columns]
    pred_df["Predicted_Improvement"]    = model.predict(pred_df[avail].fillna(0))
    pred_df["Predicted_Coverage_After"] = np.clip(
        coverage_before + pred_df["Predicted_Improvement"], 0.0, 1.0
    )
    pred_df["Weighted_Score"] = pred_df["Predicted_Improvement"] * bin_weight

    run_counts = pred_df["Seed_Run_Count"].clip(lower=1)
    pred_df["Confidence"] = np.clip(
        1.0 - (cv_std / (1.0 + np.log1p(run_counts))), 0.0, 1.0
    )

    top = pred_df.sort_values("Weighted_Score", ascending=False).head(top_n)
    recommendations = [
        SeedRecommendation(
            seed=str(row["Seed"]),
            predicted_improvement=round(float(row["Predicted_Improvement"]),     4),
            predicted_coverage_after=round(float(row["Predicted_Coverage_After"]),4),
            weighted_score=round(float(row["Weighted_Score"]),                   4),
            confidence_score=round(float(row["Confidence"]),                     4),
            rank=i + 1,
        )
        for i, (_, row) in enumerate(top.iterrows())
    ]

    if recommendations:
        best = recommendations[0]
        simulated_ca   = best.predicted_coverage_after
        simulated_hits = round(simulated_ca * avg_times_executed, 1)
    else:
        simulated_ca   = coverage_before
        simulated_hits = avg_hits

    return BinAnalysis(
        bin_name=target_bin,
        bin_weight=round(bin_weight,              4),
        coverage_before=round(coverage_before,    4),
        coverage_gap=round(coverage_gap,          4),
        weighted_priority=weighted_priority,
        avg_hits=round(avg_hits,                  2),
        avg_times_executed=round(avg_times_executed, 2),
        hit_rate_consistency=round(hit_rate_consistency, 4),
        best_historical_seed=best_hist_seed,
        historical_improvement=round(hist_improvement, 4),
        recommendations=recommendations,
        simulated_coverage_after=round(simulated_ca,   4),
        simulated_new_hits=simulated_hits,
    )


# ══════════════════════════════════════════════════════════════
# 10.  PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════

def run_pipeline(
    input_path:  str | None       = None,
    target_bins: list[str] | None = None,
    top_seeds:   int               = TOP_N_SEEDS,
    max_bins_output: int           = 10,
    max_seeds_per_bin: int         = 5,
) -> PipelineResult:

    log.info("═" * 65)
    log.info("COVERAGE OPTIMIZATION PIPELINE  v2.0  (8-column schema)")
    log.info("═" * 65)

    df_raw   = load_dataset(input_path)

    log.info("Validating dataset…")
    errors = validate_dataset(df_raw)
    for err in errors:
        if err.startswith("[FATAL]"):
            log.error("  %s", err)
        else:
            log.warning("  VALIDATION: %s", err)
    if any(e.startswith("[FATAL]") for e in errors):
        log.error("Fatal validation error — aborting.")
        sys.exit(1)

    df_clean = clean_dataset(df_raw)
    df_feat  = engineer_features(df_clean)
    analysis = analyse_coverage(df_feat)
    model, features, metrics, model_name = train_model(df_feat)
    cv_std   = metrics.get("CV_R2_Std", 0.05)

    if target_bins is None:
        target_bins = analysis["weakest_bins"]["Bin_Name"].tolist()[:TOP_N_BINS]
    if not target_bins:
        target_bins = analysis["low_cov_bins"]["Bin_Name"].tolist()[:TOP_N_BINS]

    log.info("Generating recommendations for %d bins…", len(target_bins))
    bin_results: list[BinAnalysis] = []
    for bin_id in target_bins:
        try:
            bin_results.append(
                recommend_seeds(bin_id, df_feat, model, features, analysis, cv_std, top_seeds)
            )
        except ValueError as exc:
            log.warning("  Skipping '%s': %s", bin_id, exc)

    s = analysis["summary"]
    fi: dict[str, float] = {}
    if hasattr(model, "feature_importances_"):
        fi = dict(sorted(
            {f: round(float(v), 4) for f, v in zip(features, model.feature_importances_)}.items(),
            key=lambda x: x[1], reverse=True,
        ))

    # Limit output size for frontend consumption
    bin_results = bin_results[:max_bins_output]
    for br in bin_results:
        br.recommendations = br.recommendations[:max_seeds_per_bin]

    result = PipelineResult(
        total_bins=s["total_bins"],
        total_seeds=s["total_seeds"],
        total_records=s["total_records"],
        avg_coverage_before=s["avg_coverage_before"],
        avg_coverage_after=s["avg_coverage_after"],
        avg_improvement=s["avg_improvement"],
        total_hits=s["total_hits"],
        total_executions=s["total_executions"],
        avg_bin_weight=s["avg_bin_weight"],
        low_coverage_bins=analysis["low_cov_bins"].round(4).to_dict("records")[:max_bins_output],
        weakest_bins_by_weighted_priority=analysis["weakest_bins"].round(4).to_dict("records")[:max_bins_output],
        best_seeds_global=analysis["best_seeds"].round(4).to_dict("records")[:max_bins_output],
        seed_bin_matrix_sample=analysis["seed_bin_matrix"].round(4).to_dict("records")[:max_bins_output * max_seeds_per_bin],
        model_name=model_name,
        model_metrics=metrics,
        feature_importances=fi,
        bin_recommendations=[asdict(b) for b in bin_results],
    )

    log.info("═" * 65)
    log.info("PIPELINE COMPLETE")
    log.info("  Records processed     : %d", result.total_records)
    log.info("  Bins                  : %d  |  Seeds: %d",
             result.total_bins, result.total_seeds)
    log.info("  Low-coverage bins     : %d", len(result.low_coverage_bins))
    log.info("  High-priority bins    : %d", len(result.weakest_bins_by_weighted_priority))
    log.info("  Recommendations built : %d bins", len(result.bin_recommendations))
    log.info("  Model  %-30s  R²=%.4f", result.model_name, result.model_metrics["R2"])
    log.info("═" * 65)
    return result


# ══════════════════════════════════════════════════════════════
# 11.  PRETTY PRINT
# ══════════════════════════════════════════════════════════════

def generate_compact_output(result: PipelineResult) -> dict[str, Any]:
    """Generate compact JSON output for frontend consumption."""
    # Summary
    summary = {
        "avg_coverage_before": result.avg_coverage_before,
        "avg_coverage_after": result.avg_coverage_after,
        "avg_improvement": result.avg_improvement,
    }

    # Priority bins (top 3 from weakest_bins_by_weighted_priority)
    priority_bins = [
        {
            "bin_name": b["Bin_Name"],
            "coverage_before": b["Avg_Coverage_Before"],
            "coverage_gap": b["Avg_Coverage_Gap"],
            "bin_weight": b["Avg_Bin_Weight"],
            "weighted_priority": b["Avg_Weighted_Priority"],
        }
        for b in result.weakest_bins_by_weighted_priority[:3]
    ]

    # Recommendations (top recommendation per bin for top 3 bins)
    recommendations = []
    for br in result.bin_recommendations[:3]:
        if br["recommendations"]:
            rec = br["recommendations"][0]  # top recommendation
            confidence = "High" if rec["confidence_score"] > 0.8 else "Medium" if rec["confidence_score"] > 0.5 else "Low"
            recommendations.append({
                "bin_name": br["bin_name"],
                "coverage_before": br["coverage_before"],
                "weighted_priority": br["weighted_priority"],
                "recommended_seed": rec["seed"],
                "predicted_improvement": rec["predicted_improvement"],
                "predicted_coverage_after": rec["predicted_coverage_after"],
                "confidence": confidence,
            })

    # Top seeds (top 2 from best_seeds_global)
    top_seeds = [
        {
            "seed": s["Seed"],
            "avg_improvement": s["Avg_Improvement"],
        }
        for s in result.best_seeds_global[:2]
    ]

    # Next best action (best overall: highest weighted_priority bin with its top seed)
    if result.bin_recommendations:
        best_bin = max(result.bin_recommendations, key=lambda x: x["weighted_priority"])
        if best_bin["recommendations"]:
            best_rec = best_bin["recommendations"][0]
            next_best_action = {
                "bin": best_bin["bin_name"],
                "seed": best_rec["seed"],
                "expected_gain": best_rec["predicted_improvement"],
            }
        else:
            next_best_action = {}
    else:
        next_best_action = {}

    return {
        "summary": summary,
        "priority_bins": priority_bins,
        "recommendations": recommendations,
        "top_seeds": top_seeds,
        "next_best_action": next_best_action,
    }


def _print_summary(result: PipelineResult) -> None:
    W = 65
    print(f"\n{'═' * W}")
    print("  COVERAGE OPTIMIZATION PIPELINE  v2.0  —  SUMMARY")
    print(f"{'═' * W}")
    print(f"  Records          : {result.total_records:,}")
    print(f"  Bins             : {result.total_bins}  |  Seeds: {result.total_seeds}")
    print(f"  Total Hits       : {result.total_hits:,}  |  "
          f"Total Executions: {result.total_executions:,}")
    print(f"  Avg Bin Weight   : {result.avg_bin_weight:.3f}")
    print(f"  Avg Coverage Before : {result.avg_coverage_before:.2%}")
    print(f"  Avg Coverage After  : {result.avg_coverage_after:.2%}")
    print(f"  Avg Improvement     : {result.avg_improvement:+.2%}")
    print(f"  Model  : {result.model_name}")
    print(f"  R²={result.model_metrics['R2']:.4f}  "
          f"MAE={result.model_metrics['MAE']:.4f}  "
          f"RMSE={result.model_metrics['RMSE']:.4f}  "
          f"CV_R²={result.model_metrics['CV_R2_Mean']:.4f}±{result.model_metrics['CV_R2_Std']:.4f}")

    print("\n── Top Feature Importances ─────────────────────────────────")
    for feat, imp in list(result.feature_importances.items())[:8]:
        print(f"  {feat:<38} {'█' * int(imp * 40)} {imp:.4f}")

    print("\n── High-Priority Bins (Coverage_Gap × Bin_Weight) ──────────")
    print(f"  {'Bin':<14} {'Cov_Before':>10} {'Gap':>7} {'Weight':>8} {'W_Priority':>11}")
    for b in result.weakest_bins_by_weighted_priority[:6]:
        print(
            f"  {b['Bin_Name']:<14}"
            f"  {b['Avg_Coverage_Before']:>9.2%}"
            f"  {b['Avg_Coverage_Gap']:>6.2%}"
            f"  {b['Avg_Bin_Weight']:>7.2f}"
            f"  {b['Avg_Weighted_Priority']:>10.4f}"
        )

    print("\n── Best Seeds (weighted reliability) ───────────────────────")
    for s in result.best_seeds_global[:5]:
        print(
            f"  {s['Seed']:<12}"
            f"  avg_imp={s['Avg_Improvement']:+.2%}"
            f"  w_imp={s['Avg_Weighted_Imp']:.4f}"
            f"  w_rely={s['Weighted_Reliability']:.3f}"
            f"  runs={int(s['Run_Count'])}"
        )

    print("\n── Recommendations (top 2 bins) ────────────────────────────")
    for br in result.bin_recommendations[:2]:
        print(f"\n  Bin          : {br['bin_name']}")
        print(f"  Bin Weight   : {br['bin_weight']}")
        print(f"  Cov Before   : {br['coverage_before']:.2%}  "
              f"Gap: {br['coverage_gap']:.2%}  "
              f"W-Priority: {br['weighted_priority']}")
        print(f"  Avg Hits     : {br['avg_hits']:.0f} / {br['avg_times_executed']:.0f} executions")
        print(f"  Hit Consist. : {br['hit_rate_consistency']:.4f}")
        print(f"  Best Hist.   : {br['best_historical_seed']} "
              f"(+{br['historical_improvement']:.2%})")
        print(f"  ── Seed Recommendations ──────────")
        for rec in br["recommendations"]:
            print(
                f"    #{rec['rank']} {rec['seed']:<12}"
                f"  pred_imp={rec['predicted_improvement']:+.2%}"
                f"  after={rec['predicted_coverage_after']:.2%}"
                f"  w_score={rec['weighted_score']:.4f}"
                f"  conf={rec['confidence_score']:.2%}"
            )
        print(f"  ── Simulation (best seed applied) ─")
        print(f"     Coverage : {br['coverage_before']:.2%} → {br['simulated_coverage_after']:.2%}")
        print(f"     Hits     : {br['avg_hits']:.0f} → {br['simulated_new_hits']:.0f}"
              f"  (per {br['avg_times_executed']:.0f} executions)")

    print(f"\n{'═' * W}\n")


# ══════════════════════════════════════════════════════════════
# 12.  CLI
# ══════════════════════════════════════════════════════════════

def main() -> PipelineResult:
    parser = argparse.ArgumentParser(
        description="Coverage Optimization Pipeline v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  python coverage_pipeline.py
  python coverage_pipeline.py --input data.csv --output results.json
  python coverage_pipeline.py --input data.csv --bin BIN_008 BIN_017
  python coverage_pipeline.py --input data.csv --top-seeds 8
  python coverage_pipeline.py --threshold 0.70
  python coverage_pipeline.py --max-bins-output 5 --max-seeds-per-bin 3 --output compact.json --compact
        """,
    )
    parser.add_argument("--input",     "-i", default=None,
        help="Path to CSV (or Excel/JSON/Parquet). Omit to use built-in sample data.")
    parser.add_argument("--output",    "-o", default=None,
        help="Save full results as JSON to this path.")
    parser.add_argument("--frontend-output", default=None,
        help="Save compact frontend JSON to this path (default: project root ./data.json).")
    parser.add_argument("--bin",       "-b", nargs="*",
        help="Specific Bin_Name values to recommend for.")
    parser.add_argument("--top-seeds", "-t", type=int, default=TOP_N_SEEDS,
        help=f"Seed recommendations per bin (default {TOP_N_SEEDS}).")
    parser.add_argument("--threshold", "-T", type=float, default=LOW_COVERAGE_THRESHOLD,
        help=f"Low-coverage threshold 0–1 (default {LOW_COVERAGE_THRESHOLD}).")
    parser.add_argument("--max-bins-output", type=int, default=10,
        help="Maximum number of bins to include in output (default 10).")
    parser.add_argument("--max-seeds-per-bin", type=int, default=5,
        help="Maximum seed recommendations per bin in output (default 5).")
    parser.add_argument("--compact", action="store_true",
        help="Generate compact JSON output for frontend (default: full detailed output).")
    parser.add_argument("--quiet",     "-q", action="store_true",
        help="Suppress INFO logs.")
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger("pipeline").setLevel(logging.WARNING)

    import coverage_pipeline as _self  # noqa: PLC0415
    _self.LOW_COVERAGE_THRESHOLD = args.threshold

    result = run_pipeline(
        input_path=args.input,
        target_bins=args.bin,
        top_seeds=args.top_seeds,
        max_bins_output=args.max_bins_output,
        max_seeds_per_bin=args.max_seeds_per_bin,
    )

    _print_summary(result)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            if args.compact:
                json.dump(generate_compact_output(result), fh, indent=2)
            else:
                json.dump(asdict(result), fh, indent=2, default=str)
        log.info("Results saved → %s", out)

    # Write direct frontend JSON for index.html load path
    frontend_path = Path(args.frontend_output) if args.frontend_output else Path(__file__).resolve().parents[2] / "data.json"
    frontend_path.parent.mkdir(parents=True, exist_ok=True)
    with open(frontend_path, "w", encoding="utf-8") as fh:
        json.dump(generate_compact_output(result), fh, indent=2)
    log.info("Frontend data JSON saved → %s", frontend_path)

    return result


if __name__ == "__main__":
    main()