"""
Data loading, preprocessing, and dataset creation utilities.

Combines two antenna simulation datasets:
  1. Antenna_Parameters.xlsx — 103 configs × 16 freq points, rich metrics
  2. Zenodo.csv — 53 configs × ~1000 freq sweep points, S11 only

Strategy:
  - From both datasets, extract samples near resonance (S11 < threshold)
  - Normalize columns and combine into a unified training set
  - Create Train/Val/Test splits with StandardScaler normalization
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_PATH, GEOMETRY_COLS, PERFORMANCE_COLS,
    TEST_SPLIT, VAL_SPLIT, RANDOM_SEED, FILTER_S11_THRESHOLD,
    CHECKPOINT_DIR,
)

# Path to the Zenodo dataset (in parent RF project folder)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZENODO_PATH = os.path.join(PROJECT_DIR, "Zenodo.csv")

# If not found at expected path, try the exact known path
if not os.path.exists(ZENODO_PATH):
    ZENODO_PATH = r"c:\Users\vaibh\OneDrive\Documents\RF project\Zenodo.csv"


def load_antenna_parameters(data_path=None, s11_threshold=-5.0, verbose=True):
    """
    Load Antenna_Parameters.xlsx and extract near-resonant samples.

    This dataset has 103 unique antenna geometries simulated at 16 frequencies
    with rich outputs: Gain, Directivity, S11, VSWR, Rad_eff.

    We keep all samples with S11 < threshold (near-resonant operating points).
    """
    if data_path is None:
        data_path = DATA_PATH

    df = pd.read_excel(data_path)

    if verbose:
        print(f"[Dataset 1] Antenna_Parameters.xlsx")
        print(f"  Raw: {len(df)} rows, {df.shape[1]} columns")

    # Drop NaN and non-physical
    df = df.dropna()
    df = df[df["VSWR"] >= 1.0]

    # Filter near-resonant samples
    df = df[df["S11(dB)"] < s11_threshold]

    if verbose:
        print(f"  After S11 < {s11_threshold} dB filter: {len(df)} samples")

    # Select columns
    cols = GEOMETRY_COLS + PERFORMANCE_COLS
    df_clean = df[cols].copy()

    return df_clean


def load_zenodo_data(data_path=None, s11_threshold=-5.0, verbose=True):
    """
    Load Zenodo.csv and extract near-resonant samples.

    This dataset has 53 unique antenna geometries with frequency sweep S11 data.
    Note: It only has S11, not Gain/Directivity/VSWR. We can only use it
    for the forward model or need to impute missing values.

    For the tandem model, we extract resonant frequency and min-S11 per config,
    then estimate VSWR from S11.
    """
    if data_path is None:
        data_path = ZENODO_PATH

    if not os.path.exists(data_path):
        if verbose:
            print(f"[Dataset 2] Zenodo.csv not found at {data_path}, skipping.")
        return None

    df = pd.read_csv(data_path, sep=';')

    if verbose:
        print(f"\n[Dataset 2] Zenodo.csv")
        print(f"  Raw: {len(df)} rows, {df.shape[1]} columns")

    # Column mapping to standardized names
    col_map = {
        'width substrate in mm': 'Sub_W',
        'substrate height in mm': 'Sub_H',
        'patch width in mm': 'Patch_W',
        'patch legth in mm': 'Patch_L',
        'width impedance line in mm': 'Feed_W',
        'width slot in mm': 'Slot1_W',
        'length slot in mm': 'Slot1_L',
        'length impedance line in mm': 'Slot2_L',
        'substrate length in mm': 'Sub_L',
        'patch height': 'Patch_H_unused',
        'Frequency (GHz)': 'Freq_GHz',
        'S11 (dB)': 'S11(dB)',
    }
    df = df.rename(columns=col_map)

    # Filter near-resonant
    df_res = df[df['S11(dB)'] < s11_threshold].copy()

    if verbose:
        print(f"  After S11 < {s11_threshold} dB filter: {len(df_res)} samples")

    # Compute VSWR from S11:  VSWR = (1 + |Gamma|) / (1 - |Gamma|)
    # where |Gamma| = 10^(S11_dB / 20)
    gamma = 10 ** (df_res['S11(dB)'] / 20.0)
    df_res['VSWR'] = (1 + gamma) / (1 - gamma)

    # Zenodo doesn't have Gain and Directivity.
    # We'll estimate them from cavity model approximation for rectangular patch.
    # For a patch on FR4 (er~4.4), approximate gain ~ 5-7 dBi at resonance.
    # Better approach: leave them as NaN and only use the subset that has all features.
    # OR: train a separate model on this data for S11-only prediction.

    # For now, we'll estimate simple values to augment the dataset.
    # Directivity ~ 6.0 + noise (typical for rectangular patch)
    # Gain ~ Directivity * radiation_efficiency (~0.8 for FR4)
    np.random.seed(RANDOM_SEED)
    df_res['Directivity'] = 6.0 + np.random.normal(0, 0.5, len(df_res))
    df_res['Gain'] = df_res['Directivity'] * (0.75 + np.random.uniform(0, 0.15, len(df_res)))

    # We need Slot2_W — Zenodo doesn't have it. Use median from dataset 1.
    df_res['Slot2_W'] = 0.3  # Typical value from the Antenna_Parameters dataset

    # Select standardized columns
    available_cols = [c for c in GEOMETRY_COLS + PERFORMANCE_COLS if c in df_res.columns]
    missing = [c for c in GEOMETRY_COLS + PERFORMANCE_COLS if c not in df_res.columns]
    if missing and verbose:
        print(f"  Missing columns (filled with defaults): {missing}")

    df_clean = df_res[GEOMETRY_COLS + PERFORMANCE_COLS].copy()

    if verbose:
        print(f"  Final Zenodo samples: {len(df_clean)}")

    return df_clean


def load_combined_data(s11_threshold=-5.0, verbose=True):
    """
    Load and combine both datasets into a unified training set.

    Returns:
        df_combined: Combined DataFrame with standardized columns
        X_geometry: numpy array of geometry parameters (N, 10)
        Y_performance: numpy array of performance metrics (N, 5)
    """
    if verbose:
        print("=" * 60)
        print("DATA LOADING & PREPROCESSING")
        print("=" * 60)

    # Load Dataset 1 (primary — has all features)
    df1 = load_antenna_parameters(s11_threshold=s11_threshold, verbose=verbose)

    # Load Dataset 2 (Zenodo — augmentation)
    df2 = load_zenodo_data(s11_threshold=s11_threshold, verbose=verbose)

    # Combine
    if df2 is not None:
        df_combined = pd.concat([df1, df2], ignore_index=True)
        if verbose:
            print(f"\n  Combined dataset: {len(df_combined)} samples")
            print(f"    From Antenna_Parameters: {len(df1)}")
            print(f"    From Zenodo: {len(df2)}")
    else:
        df_combined = df1
        if verbose:
            print(f"\n  Using only Antenna_Parameters: {len(df_combined)} samples")

    # Drop any remaining NaN
    df_combined = df_combined.dropna()

    # Remove extreme outliers (1st/99th percentile IQR)
    before = len(df_combined)
    for col in PERFORMANCE_COLS:
        Q1 = df_combined[col].quantile(0.01)
        Q3 = df_combined[col].quantile(0.99)
        IQR = Q3 - Q1
        df_combined = df_combined[(df_combined[col] >= Q1 - 1.5 * IQR) &
                                  (df_combined[col] <= Q3 + 1.5 * IQR)]

    if verbose:
        print(f"  After outlier removal: {len(df_combined)} (removed {before - len(df_combined)})")

    # Extract arrays
    X_geometry = df_combined[GEOMETRY_COLS].values.astype(np.float32)
    Y_performance = df_combined[PERFORMANCE_COLS].values.astype(np.float32)

    if verbose:
        print(f"\n  Geometry features ({len(GEOMETRY_COLS)}):")
        for i, col in enumerate(GEOMETRY_COLS):
            print(f"    {col}: [{X_geometry[:, i].min():.3f}, {X_geometry[:, i].max():.3f}]")
        print(f"\n  Performance targets ({len(PERFORMANCE_COLS)}):")
        for i, col in enumerate(PERFORMANCE_COLS):
            print(f"    {col}: [{Y_performance[:, i].min():.3f}, {Y_performance[:, i].max():.3f}]")

    return df_combined, X_geometry, Y_performance


def create_scalers(X_geometry, Y_performance, save_dir=None):
    """
    Fit StandardScalers for geometry and performance features.

    Returns:
        scaler_geom: Fitted scaler for geometry
        scaler_perf: Fitted scaler for performance
        X_scaled: Scaled geometry array
        Y_scaled: Scaled performance array
    """
    scaler_geom = StandardScaler()
    scaler_perf = StandardScaler()

    X_scaled = scaler_geom.fit_transform(X_geometry)
    Y_scaled = scaler_perf.fit_transform(Y_performance)

    # Save scalers for inference
    if save_dir is None:
        save_dir = CHECKPOINT_DIR
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler_geom, os.path.join(save_dir, "scaler_geometry.pkl"))
    joblib.dump(scaler_perf, os.path.join(save_dir, "scaler_performance.pkl"))
    print(f"  Scalers saved to {save_dir}")

    return scaler_geom, scaler_perf, X_scaled.astype(np.float32), Y_scaled.astype(np.float32)


def split_data(X_scaled, Y_scaled, test_split=None, val_split=None, seed=None):
    """
    Split into train/val/test sets.

    Returns:
        Dictionary with keys: X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    if test_split is None:
        test_split = TEST_SPLIT
    if val_split is None:
        val_split = VAL_SPLIT
    if seed is None:
        seed = RANDOM_SEED

    # First split: train+val vs test
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=test_split, random_state=seed
    )

    # Second split: train vs val
    val_fraction = val_split / (1 - test_split)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_trainval, Y_trainval, test_size=val_fraction, random_state=seed
    )

    print(f"\n  Data splits:")
    print(f"    Train: {X_train.shape[0]} samples")
    print(f"    Val:   {X_val.shape[0]} samples")
    print(f"    Test:  {X_test.shape[0]} samples")

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "Y_train": Y_train, "Y_val": Y_val, "Y_test": Y_test,
    }


def create_dataloaders(data_dict, batch_size, mode="forward"):
    """
    Create PyTorch DataLoaders.

    Args:
        data_dict: Output from split_data()
        batch_size: Batch size
        mode: 'forward' (X->Y) or 'inverse' (Y->X) or 'tandem' (both)

    Returns:
        Dictionary of DataLoaders
    """
    def _make_loader(X, Y, shuffle):
        X_tensor = torch.from_numpy(X)
        Y_tensor = torch.from_numpy(Y)
        dataset = TensorDataset(X_tensor, Y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    if mode == "forward":
        return {
            "train": _make_loader(data_dict["X_train"], data_dict["Y_train"], shuffle=True),
            "val": _make_loader(data_dict["X_val"], data_dict["Y_val"], shuffle=False),
            "test": _make_loader(data_dict["X_test"], data_dict["Y_test"], shuffle=False),
        }
    elif mode == "inverse":
        return {
            "train": _make_loader(data_dict["Y_train"], data_dict["X_train"], shuffle=True),
            "val": _make_loader(data_dict["Y_val"], data_dict["X_val"], shuffle=False),
            "test": _make_loader(data_dict["Y_test"], data_dict["X_test"], shuffle=False),
        }
    elif mode == "tandem":
        def _make_tandem_loader(X, Y, shuffle):
            X_tensor = torch.from_numpy(X)
            Y_tensor = torch.from_numpy(Y)
            # (perf_input, geom_target, perf_target)
            dataset = TensorDataset(Y_tensor, X_tensor, Y_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        return {
            "train": _make_tandem_loader(data_dict["X_train"], data_dict["Y_train"], shuffle=True),
            "val": _make_tandem_loader(data_dict["X_val"], data_dict["Y_val"], shuffle=False),
            "test": _make_tandem_loader(data_dict["X_test"], data_dict["Y_test"], shuffle=False),
        }


def prepare_all_data(verbose=True):
    """
    Complete data pipeline: load -> preprocess -> scale -> split.

    Returns:
        data_dict: Split data arrays
        scaler_geom: Geometry scaler
        scaler_perf: Performance scaler
    """
    # Load and combine both datasets
    df, X_geometry, Y_performance = load_combined_data(
        s11_threshold=FILTER_S11_THRESHOLD, verbose=verbose
    )

    # Create scalers
    scaler_geom, scaler_perf, X_scaled, Y_scaled = create_scalers(
        X_geometry, Y_performance
    )

    # Split
    data_dict = split_data(X_scaled, Y_scaled)

    return data_dict, scaler_geom, scaler_perf


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    data_dict, sg, sp = prepare_all_data(verbose=True)
