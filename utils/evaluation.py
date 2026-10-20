"""
Evaluation and visualization utilities for the Tandem Inverse Antenna Design system.

Provides:
  - Per-parameter error metrics (MAE, MAPE, R²)
  - Scatter plots: predicted vs. true
  - Training history plots
  - Tandem performance analysis
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEOMETRY_COLS, PERFORMANCE_COLS, RESULTS_DIR, DEVICE


def evaluate_forward_model(model, loader, scaler_geom, scaler_perf, device=None):
    """
    Evaluate forward model on test set.

    Returns:
        metrics: Dict with per-parameter MAE, MAPE, R²
        predictions: (N, 5) array of predicted performance (unscaled)
        targets: (N, 5) array of true performance (unscaled)
    """
    if device is None:
        device = DEVICE
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_geom, batch_perf in loader:
            batch_geom = batch_geom.to(device)
            pred = model(batch_geom).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(batch_perf.numpy())

    preds_scaled = np.concatenate(all_preds, axis=0)
    targets_scaled = np.concatenate(all_targets, axis=0)

    # Inverse transform to original scale
    preds = scaler_perf.inverse_transform(preds_scaled)
    targets = scaler_perf.inverse_transform(targets_scaled)

    metrics = {}
    print("\n📊 Forward Model Test Results:")
    print("-" * 65)
    print(f"{'Parameter':<15} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'MAPE%':>8}")
    print("-" * 65)
    for i, col in enumerate(PERFORMANCE_COLS):
        mae = mean_absolute_error(targets[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(targets[:, i], preds[:, i]))
        r2 = r2_score(targets[:, i], preds[:, i])
        # MAPE (avoid division by zero)
        mask = np.abs(targets[:, i]) > 1e-6
        if mask.sum() > 0:
            mape = np.mean(np.abs((targets[mask, i] - preds[mask, i]) / targets[mask, i])) * 100
        else:
            mape = 0
        metrics[col] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
        print(f"{col:<15} {mae:>10.4f} {rmse:>10.4f} {r2:>8.4f} {mape:>7.2f}%")
    print("-" * 65)

    return metrics, preds, targets


def evaluate_tandem_model(tandem_model, loader, scaler_geom, scaler_perf, device=None):
    """
    Evaluate the tandem model on test set.

    Returns:
        geom_metrics: Per-geometry-parameter metrics
        perf_metrics: Reconstructed performance metrics
        pred_geometry: (N, 10) predicted geometry (unscaled)
        true_geometry: (N, 10) true geometry (unscaled)
        recon_performance: (N, 5) reconstructed performance (unscaled)
        true_performance: (N, 5) true performance (unscaled)
    """
    if device is None:
        device = DEVICE
    tandem_model = tandem_model.to(device)
    tandem_model.eval()

    all_pred_geom = []
    all_true_geom = []
    all_recon_perf = []
    all_true_perf = []

    with torch.no_grad():
        for batch_perf_in, batch_geom_target, batch_perf_target in loader:
            batch_perf_in = batch_perf_in.to(device)

            pred_geom, recon_perf = tandem_model(batch_perf_in)

            all_pred_geom.append(pred_geom.cpu().numpy())
            all_true_geom.append(batch_geom_target.numpy())
            all_recon_perf.append(recon_perf.cpu().numpy())
            all_true_perf.append(batch_perf_target.numpy())

    pred_geom_scaled = np.concatenate(all_pred_geom, axis=0)
    true_geom_scaled = np.concatenate(all_true_geom, axis=0)
    recon_perf_scaled = np.concatenate(all_recon_perf, axis=0)
    true_perf_scaled = np.concatenate(all_true_perf, axis=0)

    # Inverse transform
    pred_geometry = scaler_geom.inverse_transform(pred_geom_scaled)
    true_geometry = scaler_geom.inverse_transform(true_geom_scaled)
    recon_performance = scaler_perf.inverse_transform(recon_perf_scaled)
    true_performance = scaler_perf.inverse_transform(true_perf_scaled)

    # Geometry metrics
    print("\n📐 Tandem Model — Geometry Prediction:")
    print("-" * 65)
    print(f"{'Parameter':<15} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'MAPE%':>8}")
    print("-" * 65)
    geom_metrics = {}
    for i, col in enumerate(GEOMETRY_COLS):
        mae = mean_absolute_error(true_geometry[:, i], pred_geometry[:, i])
        rmse = np.sqrt(mean_squared_error(true_geometry[:, i], pred_geometry[:, i]))
        r2 = r2_score(true_geometry[:, i], pred_geometry[:, i])
        mask = np.abs(true_geometry[:, i]) > 1e-6
        mape = np.mean(np.abs((true_geometry[mask, i] - pred_geometry[mask, i]) / true_geometry[mask, i])) * 100 if mask.sum() > 0 else 0
        geom_metrics[col] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
        print(f"{col:<15} {mae:>10.4f} {rmse:>10.4f} {r2:>8.4f} {mape:>7.2f}%")
    print("-" * 65)

    # Performance reconstruction metrics
    print("\n📡 Tandem Model — Performance Reconstruction:")
    print("-" * 65)
    print(f"{'Parameter':<15} {'MAE':>10} {'RMSE':>10} {'R²':>8} {'MAPE%':>8}")
    print("-" * 65)
    perf_metrics = {}
    for i, col in enumerate(PERFORMANCE_COLS):
        mae = mean_absolute_error(true_performance[:, i], recon_performance[:, i])
        rmse = np.sqrt(mean_squared_error(true_performance[:, i], recon_performance[:, i]))
        r2 = r2_score(true_performance[:, i], recon_performance[:, i])
        mask = np.abs(true_performance[:, i]) > 1e-6
        mape = np.mean(np.abs((true_performance[mask, i] - recon_performance[mask, i]) / true_performance[mask, i])) * 100 if mask.sum() > 0 else 0
        perf_metrics[col] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}
        print(f"{col:<15} {mae:>10.4f} {rmse:>10.4f} {r2:>8.4f} {mape:>7.2f}%")
    print("-" * 65)

    return (geom_metrics, perf_metrics, pred_geometry, true_geometry,
            recon_performance, true_performance)


def plot_training_history(history, title, save_path):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: {save_path}")


def plot_tandem_history(history, save_path):
    """Plot tandem training history with geometry and performance losses."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Total loss
    axes[0].plot(history["train_loss"], label="Train", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val", linewidth=2)
    axes[0].set_title("Total Tandem Loss", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')

    # Geometry loss
    axes[1].plot(history["train_geom_loss"], label="Train", linewidth=2, color='green')
    axes[1].plot(history["val_geom_loss"], label="Val", linewidth=2, color='darkgreen')
    axes[1].set_title("Geometry Loss", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    # Performance loss
    axes[2].plot(history["train_perf_loss"], label="Train", linewidth=2, color='orange')
    axes[2].plot(history["val_perf_loss"], label="Val", linewidth=2, color='red')
    axes[2].set_title("Performance Recon. Loss", fontsize=13)
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')

    plt.suptitle("Tandem Network Training", fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: {save_path}")


def plot_predictions_scatter(true_vals, pred_vals, col_names, title_prefix, save_path):
    """
    Scatter plot of predicted vs. true values for each parameter.
    Perfect prediction lies on the diagonal.
    """
    n_cols = len(col_names)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]

    for i, col in enumerate(col_names):
        ax = axes[i]
        ax.scatter(true_vals[:, i], pred_vals[:, i], alpha=0.5, s=20, edgecolors='none')

        # Diagonal line
        lims = [
            min(true_vals[:, i].min(), pred_vals[:, i].min()),
            max(true_vals[:, i].max(), pred_vals[:, i].max()),
        ]
        ax.plot(lims, lims, 'r--', linewidth=1.5, label='Ideal')

        r2 = r2_score(true_vals[:, i], pred_vals[:, i])
        ax.set_title(f"{col}  (R² = {r2:.4f})", fontsize=11)
        ax.set_xlabel("True", fontsize=10)
        ax.set_ylabel("Predicted", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_cols, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"{title_prefix} — Predicted vs True", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: {save_path}")


def plot_error_distribution(true_vals, pred_vals, col_names, title_prefix, save_path):
    """Histogram of prediction errors for each parameter."""
    n_cols = len(col_names)
    n_rows = (n_cols + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_cols > 1 else [axes]

    for i, col in enumerate(col_names):
        ax = axes[i]
        errors = pred_vals[:, i] - true_vals[:, i]
        ax.hist(errors, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
        ax.axvline(0, color='red', linewidth=1.5, linestyle='--')
        ax.set_title(f"{col}  (μ={errors.mean():.4f}, σ={errors.std():.4f})", fontsize=10)
        ax.set_xlabel("Error (Predicted - True)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.grid(True, alpha=0.3)

    for j in range(n_cols, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"{title_prefix} — Error Distribution", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  📈 Saved: {save_path}")
