"""
Main Training Pipeline for Inverse Antenna Design using Tandem Network.

Usage:
    python train.py

This script runs the complete 3-phase training pipeline:
  Phase 1: Train forward surrogate model (geometry → performance)
  Phase 2: Pre-train inverse model (performance → geometry)
  Phase 3: Tandem fine-tuning (inverse → frozen forward)

All hyperparameters are configured in config.py
"""

import os
import sys
import time
import torch
import numpy as np

# Fix Windows console encoding for emoji/unicode
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    DEVICE, CHECKPOINT_DIR, RESULTS_DIR,
    FORWARD_BATCH_SIZE, INVERSE_BATCH_SIZE, TANDEM_BATCH_SIZE,
    GEOMETRY_COLS, PERFORMANCE_COLS,
)
from utils.data_loader import prepare_all_data, create_dataloaders
from models.networks import ForwardModel, InverseModel, TandemNetwork, count_parameters
from models.trainer import train_forward_model, train_inverse_model, train_tandem_network
from utils.evaluation import (
    evaluate_forward_model, evaluate_tandem_model,
    plot_training_history, plot_tandem_history,
    plot_predictions_scatter, plot_error_distribution,
)


def main():
    print("=" * 70)
    print("  🛰️  INVERSE ANTENNA DESIGN USING TANDEM NEURAL NETWORK")
    print("  📡  Microstrip Patch Antenna — 5G Band")
    print("=" * 70)
    print(f"\n  Device: {DEVICE}")
    print(f"  Geometry params: {len(GEOMETRY_COLS)} → {GEOMETRY_COLS}")
    print(f"  Performance params: {len(PERFORMANCE_COLS)} → {PERFORMANCE_COLS}")

    start_time = time.time()

    # ================================================================
    # STEP 1: PREPARE DATA
    # ================================================================
    print("\n" + "=" * 60)
    print("📦 STEP 1: Data Preparation")
    print("=" * 60)

    data_dict, scaler_geom, scaler_perf = prepare_all_data(verbose=True)

    # Create DataLoaders for each phase
    loaders_forward = create_dataloaders(data_dict, FORWARD_BATCH_SIZE, mode="forward")
    loaders_inverse = create_dataloaders(data_dict, INVERSE_BATCH_SIZE, mode="inverse")
    loaders_tandem = create_dataloaders(data_dict, TANDEM_BATCH_SIZE, mode="tandem")

    n_geom = len(GEOMETRY_COLS)
    n_perf = len(PERFORMANCE_COLS)

    # ================================================================
    # STEP 2: PHASE 1 — TRAIN FORWARD MODEL
    # ================================================================
    forward_model = ForwardModel(input_dim=n_geom, output_dim=n_perf)
    print(f"\n  Forward Model Parameters: {count_parameters(forward_model):,}")

    forward_model, forward_history = train_forward_model(
        forward_model, loaders_forward, device=DEVICE
    )

    # Evaluate forward model
    fwd_metrics, fwd_preds, fwd_targets = evaluate_forward_model(
        forward_model, loaders_forward["test"], scaler_geom, scaler_perf
    )

    # Save plots
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_training_history(
        forward_history, "Forward Model Training",
        os.path.join(RESULTS_DIR, "forward_training_history.png")
    )
    plot_predictions_scatter(
        fwd_targets, fwd_preds, PERFORMANCE_COLS,
        "Forward Model", os.path.join(RESULTS_DIR, "forward_scatter.png")
    )

    # ================================================================
    # STEP 3: PHASE 2 — PRE-TRAIN INVERSE MODEL
    # ================================================================
    inverse_model = InverseModel(input_dim=n_perf, output_dim=n_geom)
    print(f"\n  Inverse Model Parameters: {count_parameters(inverse_model):,}")

    inverse_model, inverse_history = train_inverse_model(
        inverse_model, loaders_inverse, device=DEVICE
    )

    plot_training_history(
        inverse_history, "Inverse Model Pre-Training",
        os.path.join(RESULTS_DIR, "inverse_pretrain_history.png")
    )

    # ================================================================
    # STEP 4: PHASE 3 — TANDEM FINE-TUNING
    # ================================================================
    tandem_model = TandemNetwork(
        inverse_model=inverse_model,
        forward_model=forward_model,
        freeze_forward=True
    )
    print(f"\n  Tandem trainable params: {count_parameters(tandem_model):,} "
          f"(only inverse model)")

    tandem_model, tandem_history = train_tandem_network(
        tandem_model, loaders_tandem, device=DEVICE
    )

    # ================================================================
    # STEP 5: FINAL EVALUATION
    # ================================================================
    print("\n" + "=" * 60)
    print("🏆 FINAL EVALUATION ON TEST SET")
    print("=" * 60)

    (geom_metrics, perf_metrics, pred_geometry, true_geometry,
     recon_performance, true_performance) = evaluate_tandem_model(
        tandem_model, loaders_tandem["test"], scaler_geom, scaler_perf
    )

    # Save all plots
    plot_tandem_history(
        tandem_history, os.path.join(RESULTS_DIR, "tandem_training_history.png")
    )
    plot_predictions_scatter(
        true_geometry, pred_geometry, GEOMETRY_COLS,
        "Tandem — Geometry", os.path.join(RESULTS_DIR, "tandem_geometry_scatter.png")
    )
    plot_predictions_scatter(
        true_performance, recon_performance, PERFORMANCE_COLS,
        "Tandem — Performance Reconstruction",
        os.path.join(RESULTS_DIR, "tandem_performance_scatter.png")
    )
    plot_error_distribution(
        true_geometry, pred_geometry, GEOMETRY_COLS,
        "Tandem — Geometry", os.path.join(RESULTS_DIR, "tandem_geometry_errors.png")
    )
    plot_error_distribution(
        true_performance, recon_performance, PERFORMANCE_COLS,
        "Tandem — Performance", os.path.join(RESULTS_DIR, "tandem_performance_errors.png")
    )

    # Save final models
    torch.save(forward_model.state_dict(),
               os.path.join(CHECKPOINT_DIR, "forward_model_final.pt"))
    torch.save(tandem_model.inverse_model.state_dict(),
               os.path.join(CHECKPOINT_DIR, "inverse_model_final.pt"))

    elapsed = time.time() - start_time
    print(f"\n⏱️  Total training time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"💾 Models saved to: {CHECKPOINT_DIR}")
    print(f"📊 Plots saved to: {RESULTS_DIR}")
    print("\n✅ Training complete! Run `python predict.py` for inference.")


if __name__ == "__main__":
    main()
