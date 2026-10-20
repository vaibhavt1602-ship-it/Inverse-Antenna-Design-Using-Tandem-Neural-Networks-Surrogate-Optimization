"""
Training engine for the Tandem Inverse Antenna Design system.

Three-phase training pipeline:
  Phase 1: Forward model training (geometry → performance)
  Phase 2: Inverse model pre-training (performance → geometry)
  Phase 3: Tandem fine-tuning (inverse → frozen forward, end-to-end)
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FORWARD_LR, FORWARD_WEIGHT_DECAY, FORWARD_EPOCHS, FORWARD_BATCH_SIZE,
    FORWARD_PATIENCE, FORWARD_LR_FACTOR, FORWARD_LR_PATIENCE,
    INVERSE_LR, INVERSE_WEIGHT_DECAY, INVERSE_EPOCHS, INVERSE_BATCH_SIZE,
    INVERSE_PATIENCE,
    TANDEM_ALPHA, TANDEM_BETA, TANDEM_LR, TANDEM_WEIGHT_DECAY,
    TANDEM_EPOCHS, TANDEM_BATCH_SIZE, TANDEM_PATIENCE,
    DEVICE, CHECKPOINT_DIR,
)


class EarlyStopping:
    """Early stopping with patience and best model checkpoint."""
    def __init__(self, patience=30, delta=1e-6, path="best_model.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(self.best_state, self.path)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_forward_model(model, loaders, device=None, verbose=True):
    """
    Phase 1: Train the forward surrogate model.

    Learns the mapping: geometry → performance
    This replaces expensive EM simulations with a fast neural network.
    """
    if device is None:
        device = DEVICE
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=FORWARD_LR, weight_decay=FORWARD_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=FORWARD_LR_FACTOR,
        patience=FORWARD_LR_PATIENCE,
    )
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(
        patience=FORWARD_PATIENCE,
        path=os.path.join(CHECKPOINT_DIR, "forward_model_best.pt")
    )

    history = defaultdict(list)

    print("\n" + "=" * 60)
    print("📡 PHASE 1: Training Forward Model (Geometry → Performance)")
    print("=" * 60)

    for epoch in range(FORWARD_EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0
        for batch_geom, batch_perf in loaders["train"]:
            batch_geom = batch_geom.to(device)
            batch_perf = batch_perf.to(device)

            optimizer.zero_grad()
            pred_perf = model(batch_geom)
            loss = criterion(pred_perf, batch_perf)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_geom.size(0)

        train_loss /= len(loaders["train"].dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_geom, batch_perf in loaders["val"]:
                batch_geom = batch_geom.to(device)
                batch_perf = batch_perf.to(device)
                pred_perf = model(batch_geom)
                loss = criterion(pred_perf, batch_perf)
                val_loss += loss.item() * batch_geom.size(0)
        val_loss /= len(loaders["val"].dataset)

        scheduler.step(val_loss)
        early_stop(val_loss, model)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % 25 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:4d}/{FORWARD_EPOCHS} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.1e}")

        if early_stop.early_stop:
            print(f"  ⏹ Early stopping at epoch {epoch+1} (best val: {early_stop.best_loss:.6f})")
            break

    early_stop.load_best(model)
    print(f"  ✅ Best forward model val loss: {early_stop.best_loss:.6f}")
    return model, history


def train_inverse_model(model, loaders, device=None, verbose=True):
    """
    Phase 2: Pre-train the inverse model.

    Learns the mapping: performance → geometry
    This gives a good initialization before tandem fine-tuning.
    """
    if device is None:
        device = DEVICE
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=INVERSE_LR, weight_decay=INVERSE_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20,
    )
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(
        patience=INVERSE_PATIENCE,
        path=os.path.join(CHECKPOINT_DIR, "inverse_model_best.pt")
    )

    history = defaultdict(list)

    print("\n" + "=" * 60)
    print("🔄 PHASE 2: Pre-training Inverse Model (Performance → Geometry)")
    print("=" * 60)

    for epoch in range(INVERSE_EPOCHS):
        # --- Training ---
        model.train()
        train_loss = 0
        for batch_perf, batch_geom in loaders["train"]:
            batch_perf = batch_perf.to(device)
            batch_geom = batch_geom.to(device)

            optimizer.zero_grad()
            pred_geom = model(batch_perf)
            loss = criterion(pred_geom, batch_geom)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_perf.size(0)

        train_loss /= len(loaders["train"].dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_perf, batch_geom in loaders["val"]:
                batch_perf = batch_perf.to(device)
                batch_geom = batch_geom.to(device)
                pred_geom = model(batch_perf)
                loss = criterion(pred_geom, batch_geom)
                val_loss += loss.item() * batch_perf.size(0)
        val_loss /= len(loaders["val"].dataset)

        scheduler.step(val_loss)
        early_stop(val_loss, model)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and (epoch + 1) % 25 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:4d}/{INVERSE_EPOCHS} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.1e}")

        if early_stop.early_stop:
            print(f"  ⏹ Early stopping at epoch {epoch+1} (best val: {early_stop.best_loss:.6f})")
            break

    early_stop.load_best(model)
    print(f"  ✅ Best inverse model val loss: {early_stop.best_loss:.6f}")
    return model, history


def train_tandem_network(tandem_model, loaders, device=None, verbose=True):
    """
    Phase 3: Tandem fine-tuning.

    Connects the inverse model to the frozen forward model:
        performance_spec → InverseModel → geometry → FrozenForward → recon_performance

    Loss = α * MSE(predicted_geometry, true_geometry)
         + β * MSE(reconstructed_performance, true_performance)

    Only the inverse model weights are updated.
    """
    if device is None:
        device = DEVICE
    tandem_model = tandem_model.to(device)

    # Only optimize inverse model parameters (forward is frozen)
    optimizer = torch.optim.AdamW(
        tandem_model.inverse_model.parameters(),
        lr=TANDEM_LR, weight_decay=TANDEM_WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    criterion = nn.MSELoss()
    early_stop = EarlyStopping(
        patience=TANDEM_PATIENCE,
        path=os.path.join(CHECKPOINT_DIR, "tandem_inverse_best.pt")
    )

    history = defaultdict(list)
    alpha = TANDEM_ALPHA
    beta = TANDEM_BETA

    print("\n" + "=" * 60)
    print("🔗 PHASE 3: Tandem Fine-Tuning (Inverse → Frozen Forward)")
    print(f"   Loss = {alpha:.1f} × geometry_loss + {beta:.1f} × performance_loss")
    print("=" * 60)

    for epoch in range(TANDEM_EPOCHS):
        # --- Training ---
        tandem_model.inverse_model.train()
        tandem_model.forward_model.eval()  # Keep frozen

        train_loss = 0
        train_geom_loss = 0
        train_perf_loss = 0

        for batch_perf_in, batch_geom_target, batch_perf_target in loaders["train"]:
            batch_perf_in = batch_perf_in.to(device)
            batch_geom_target = batch_geom_target.to(device)
            batch_perf_target = batch_perf_target.to(device)

            optimizer.zero_grad()

            pred_geom, recon_perf = tandem_model(batch_perf_in)

            # Geometry reconstruction loss
            geom_loss = criterion(pred_geom, batch_geom_target)
            # Performance reconstruction loss (through frozen forward)
            perf_loss = criterion(recon_perf, batch_perf_target)
            # Combined tandem loss
            loss = alpha * geom_loss + beta * perf_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                tandem_model.inverse_model.parameters(), max_norm=1.0
            )
            optimizer.step()

            bs = batch_perf_in.size(0)
            train_loss += loss.item() * bs
            train_geom_loss += geom_loss.item() * bs
            train_perf_loss += perf_loss.item() * bs

        n_train = len(loaders["train"].dataset)
        train_loss /= n_train
        train_geom_loss /= n_train
        train_perf_loss /= n_train

        # --- Validation ---
        tandem_model.eval()
        val_loss = 0
        val_geom_loss = 0
        val_perf_loss = 0

        with torch.no_grad():
            for batch_perf_in, batch_geom_target, batch_perf_target in loaders["val"]:
                batch_perf_in = batch_perf_in.to(device)
                batch_geom_target = batch_geom_target.to(device)
                batch_perf_target = batch_perf_target.to(device)

                pred_geom, recon_perf = tandem_model(batch_perf_in)

                geom_loss = criterion(pred_geom, batch_geom_target)
                perf_loss = criterion(recon_perf, batch_perf_target)
                loss = alpha * geom_loss + beta * perf_loss

                bs = batch_perf_in.size(0)
                val_loss += loss.item() * bs
                val_geom_loss += geom_loss.item() * bs
                val_perf_loss += perf_loss.item() * bs

        n_val = len(loaders["val"].dataset)
        val_loss /= n_val
        val_geom_loss /= n_val
        val_perf_loss /= n_val

        scheduler.step()
        early_stop(val_loss, tandem_model.inverse_model)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_geom_loss"].append(train_geom_loss)
        history["val_geom_loss"].append(val_geom_loss)
        history["train_perf_loss"].append(train_perf_loss)
        history["val_perf_loss"].append(val_perf_loss)

        if verbose and (epoch + 1) % 25 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1:4d}/{TANDEM_EPOCHS} | "
                  f"Train: {train_loss:.6f} (G:{train_geom_loss:.4f} P:{train_perf_loss:.4f}) | "
                  f"Val: {val_loss:.6f} | LR: {lr:.1e}")

        if early_stop.early_stop:
            print(f"  ⏹ Early stopping at epoch {epoch+1} (best val: {early_stop.best_loss:.6f})")
            break

    early_stop.load_best(tandem_model.inverse_model)
    print(f"  ✅ Best tandem val loss: {early_stop.best_loss:.6f}")
    return tandem_model, history
