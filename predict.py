"""
Inference Script for Inverse Antenna Design.

Given desired performance specifications, predict the optimal antenna geometry
using the trained Tandem Network.

Usage:
    python predict.py
    python predict.py --freq 28.0 --gain 6.0 --directivity 7.5 --s11 -15.0 --vswr 1.5

The script:
  1. Loads the trained inverse model and scalers
  2. Takes desired performance specs as input
  3. Predicts the optimal geometry parameters
  4. Validates the prediction through the forward model
"""

import argparse
import os
import sys
import torch
import numpy as np
import joblib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    GEOMETRY_COLS, PERFORMANCE_COLS, CHECKPOINT_DIR, DEVICE,
)
from models.networks import ForwardModel, InverseModel, TandemNetwork


def load_trained_models(checkpoint_dir=None):
    """Load trained models and scalers from disk."""
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR

    n_geom = len(GEOMETRY_COLS)
    n_perf = len(PERFORMANCE_COLS)

    # Load scalers
    scaler_geom = joblib.load(os.path.join(checkpoint_dir, "scaler_geometry.pkl"))
    scaler_perf = joblib.load(os.path.join(checkpoint_dir, "scaler_performance.pkl"))

    # Load models
    forward_model = ForwardModel(input_dim=n_geom, output_dim=n_perf)
    inverse_model = InverseModel(input_dim=n_perf, output_dim=n_geom)

    forward_model.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "forward_model_final.pt"),
                    map_location=DEVICE, weights_only=True)
    )
    inverse_model.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "inverse_model_final.pt"),
                    map_location=DEVICE, weights_only=True)
    )

    tandem = TandemNetwork(inverse_model, forward_model, freeze_forward=True)
    tandem.eval()

    return tandem, scaler_geom, scaler_perf


def predict_geometry(performance_spec, tandem_model, scaler_geom, scaler_perf):
    """
    Predict antenna geometry from desired performance specs.

    Args:
        performance_spec: dict with keys from PERFORMANCE_COLS
            Example: {"Freq_GHz": 28.0, "Gain": 6.0, "Directivity": 7.5,
                      "S11(dB)": -15.0, "VSWR": 1.5}
        tandem_model: Loaded TandemNetwork
        scaler_geom: Fitted geometry scaler
        scaler_perf: Fitted performance scaler

    Returns:
        geometry: dict with predicted geometry parameter values
        recon_perf: dict with reconstructed performance (for validation)
    """
    # Build input array
    perf_array = np.array([[performance_spec[col] for col in PERFORMANCE_COLS]],
                          dtype=np.float32)

    # Scale
    perf_scaled = scaler_perf.transform(perf_array)
    perf_tensor = torch.from_numpy(perf_scaled).to(DEVICE)

    # Predict
    tandem_model.eval()
    with torch.no_grad():
        pred_geom_scaled, recon_perf_scaled = tandem_model(perf_tensor)

    # Inverse transform
    pred_geom = scaler_geom.inverse_transform(pred_geom_scaled.cpu().numpy())
    recon_perf = scaler_perf.inverse_transform(recon_perf_scaled.cpu().numpy())

    # Build output dicts
    geometry = {col: float(pred_geom[0, i]) for i, col in enumerate(GEOMETRY_COLS)}
    reconstructed_perf = {col: float(recon_perf[0, i]) for i, col in enumerate(PERFORMANCE_COLS)}

    return geometry, reconstructed_perf


def print_results(desired_perf, predicted_geom, reconstructed_perf):
    """Pretty-print the prediction results."""
    print("\n" + "=" * 60)
    print("  🎯 INVERSE ANTENNA DESIGN PREDICTION")
    print("=" * 60)

    print("\n📋 Desired Performance Specifications:")
    print("-" * 40)
    for col, val in desired_perf.items():
        unit = "GHz" if "Freq" in col else "dB" if "dB" in col or "Gain" in col or "Directivity" in col else ""
        print(f"  {col:<15}: {val:>10.4f} {unit}")

    print("\n📐 Predicted Antenna Geometry:")
    print("-" * 40)
    for col, val in predicted_geom.items():
        print(f"  {col:<15}: {val:>10.4f} mm")

    print("\n🔄 Reconstructed Performance (Validation via Forward Model):")
    print("-" * 40)
    for col in desired_perf:
        desired = desired_perf[col]
        recon = reconstructed_perf[col]
        error_pct = abs((recon - desired) / desired) * 100 if abs(desired) > 1e-6 else 0
        status = "✅" if error_pct < 10 else "⚠️" if error_pct < 20 else "❌"
        print(f"  {col:<15}: {recon:>10.4f}  (target: {desired:.4f}, err: {error_pct:.1f}%) {status}")

    print("\n" + "=" * 60)


def interactive_mode(tandem_model, scaler_geom, scaler_perf):
    """Interactive loop for predictions."""
    print("\n🎮 Interactive Mode — Enter desired performance specs")
    print("   Type 'quit' to exit\n")

    while True:
        try:
            specs = {}
            for col in PERFORMANCE_COLS:
                val = input(f"  {col}: ")
                if val.lower() == 'quit':
                    print("👋 Goodbye!")
                    return
                specs[col] = float(val)

            geometry, recon_perf = predict_geometry(
                specs, tandem_model, scaler_geom, scaler_perf
            )
            print_results(specs, geometry, recon_perf)

        except (ValueError, KeyboardInterrupt):
            print("\n👋 Goodbye!")
            return


def main():
    parser = argparse.ArgumentParser(
        description="Predict antenna geometry from desired performance specs"
    )
    parser.add_argument("--freq", type=float, help="Target frequency (GHz)")
    parser.add_argument("--gain", type=float, help="Target gain (dBi)")
    parser.add_argument("--directivity", type=float, help="Target directivity (dBi)")
    parser.add_argument("--s11", type=float, help="Target S11 return loss (dB)")
    parser.add_argument("--vswr", type=float, help="Target VSWR")
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive mode")

    args = parser.parse_args()

    # Load models
    print("🔄 Loading trained models...")
    tandem_model, scaler_geom, scaler_perf = load_trained_models()
    print("✅ Models loaded successfully!")

    if args.interactive:
        interactive_mode(tandem_model, scaler_geom, scaler_perf)
    elif args.freq is not None:
        # Command-line mode
        specs = {
            "Freq_GHz": args.freq,
            "Gain": args.gain if args.gain else 6.0,
            "Directivity": args.directivity if args.directivity else 7.0,
            "S11(dB)": args.s11 if args.s11 else -15.0,
            "VSWR": args.vswr if args.vswr else 1.5,
        }
        geometry, recon_perf = predict_geometry(
            specs, tandem_model, scaler_geom, scaler_perf
        )
        print_results(specs, geometry, recon_perf)
    else:
        # Demo with sample predictions
        print("\n📌 Running demo predictions...")
        demo_specs = [
            {"Freq_GHz": 28.0, "Gain": 6.5, "Directivity": 7.5, "S11(dB)": -20.0, "VSWR": 1.2},
            {"Freq_GHz": 38.0, "Gain": 5.0, "Directivity": 6.0, "S11(dB)": -15.0, "VSWR": 1.5},
            {"Freq_GHz": 45.0, "Gain": 7.0, "Directivity": 8.0, "S11(dB)": -25.0, "VSWR": 1.1},
        ]
        for specs in demo_specs:
            geometry, recon_perf = predict_geometry(
                specs, tandem_model, scaler_geom, scaler_perf
            )
            print_results(specs, geometry, recon_perf)


if __name__ == "__main__":
    main()
