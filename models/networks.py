"""
Neural Network architectures for the Tandem Inverse Antenna Design system.

Three models:
  1. ForwardModel:  geometry → performance  (surrogate for EM simulation)
  2. InverseModel:  performance → geometry  (the inverse design network)
  3. TandemNetwork: Connects InverseModel → frozen ForwardModel for end-to-end training

Architecture Details:
  - Deep fully-connected networks with skip connections
  - Batch normalization for training stability
  - LeakyReLU activation (handles negative values in antenna params)
  - Dropout for regularization
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    FORWARD_HIDDEN_DIMS, FORWARD_DROPOUT, FORWARD_ACTIVATION,
    INVERSE_HIDDEN_DIMS, INVERSE_DROPOUT, INVERSE_ACTIVATION,
)


class ResidualBlock(nn.Module):
    """
    Residual block with BatchNorm, activation, and dropout.
    Adds a skip connection when input/output dims match.
    """
    def __init__(self, in_dim, out_dim, dropout=0.1, activation="LeakyReLU"):
        super().__init__()
        self.use_skip = (in_dim == out_dim)

        act_fn = getattr(nn, activation)() if hasattr(nn, activation) else nn.LeakyReLU(0.1)

        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            act_fn,
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_skip:
            out = out + x
        return out


class ForwardModel(nn.Module):
    """
    Forward Surrogate Model: Geometry Parameters → Performance Metrics.

    This replaces the EM simulator (CST/HFSS) with a fast neural network
    that predicts antenna performance from geometry in milliseconds.

    Input:  [Sub_W, Sub_L, Sub_H, Patch_W, Patch_L, Feed_W,
             Slot1_W, Slot1_L, Slot2_W, Slot2_L]  → 10 features
    Output: [Freq_GHz, Gain, Directivity, S11(dB), VSWR]  → 5 features
    """
    def __init__(self, input_dim=10, output_dim=5, hidden_dims=None,
                 dropout=None, activation=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = FORWARD_HIDDEN_DIMS
        if dropout is None:
            dropout = FORWARD_DROPOUT
        if activation is None:
            activation = FORWARD_ACTIVATION

        layers = []

        # Input projection
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        act_fn = getattr(nn, activation)() if hasattr(nn, activation) else nn.LeakyReLU(0.1)
        layers.append(act_fn)
        layers.append(nn.Dropout(dropout))

        # Hidden layers with residual connections
        for i in range(len(hidden_dims) - 1):
            layers.append(ResidualBlock(
                hidden_dims[i], hidden_dims[i + 1],
                dropout=dropout, activation=activation
            ))

        # Output projection (no activation — regression)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, geometry):
        """
        Args:
            geometry: (batch, 10) scaled geometry parameters
        Returns:
            performance: (batch, 5) predicted performance metrics
        """
        return self.network(geometry)


class InverseModel(nn.Module):
    """
    Inverse Design Model: Performance Specs → Geometry Parameters.

    Given desired antenna performance, predict the geometry that achieves it.
    This is the "inverse" of the EM simulation problem.

    Input:  [Freq_GHz, Gain, Directivity, S11(dB), VSWR]  → 5 features
    Output: [Sub_W, Sub_L, Sub_H, Patch_W, Patch_L, Feed_W,
             Slot1_W, Slot1_L, Slot2_W, Slot2_L]  → 10 features
    """
    def __init__(self, input_dim=5, output_dim=10, hidden_dims=None,
                 dropout=None, activation=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = INVERSE_HIDDEN_DIMS
        if dropout is None:
            dropout = INVERSE_DROPOUT
        if activation is None:
            activation = INVERSE_ACTIVATION

        layers = []

        # Input projection
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        act_fn = getattr(nn, activation)() if hasattr(nn, activation) else nn.LeakyReLU(0.1)
        layers.append(act_fn)
        layers.append(nn.Dropout(dropout))

        # Hidden layers with residual connections
        for i in range(len(hidden_dims) - 1):
            layers.append(ResidualBlock(
                hidden_dims[i], hidden_dims[i + 1],
                dropout=dropout, activation=activation
            ))

        # Output projection
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, performance):
        """
        Args:
            performance: (batch, 5) desired performance specs (scaled)
        Returns:
            geometry: (batch, 10) predicted geometry parameters (scaled)
        """
        return self.network(performance)


class TandemNetwork(nn.Module):
    """
    Tandem Network for Inverse Antenna Design.

    Architecture:
        performance_spec → [InverseModel] → predicted_geometry → [Frozen ForwardModel] → predicted_performance

    Training strategy:
        1. Phase 1: Train ForwardModel alone (geometry → performance)
        2. Phase 2: Train InverseModel alone (performance → geometry)
        3. Phase 3: Connect them in tandem. Freeze ForwardModel weights.
           Loss = α * ||predicted_geometry - true_geometry||²
                + β * ||forward(predicted_geometry) - true_performance||²

    The tandem loss ensures the predicted geometry is both:
        (a) close to known good designs
        (b) actually produces the desired performance when simulated

    This elegantly handles the one-to-many mapping problem in inverse design
    (many different geometries can produce similar performance).
    """
    def __init__(self, inverse_model, forward_model, freeze_forward=True):
        super().__init__()

        self.inverse_model = inverse_model
        self.forward_model = forward_model

        # Freeze forward model weights during tandem training
        if freeze_forward:
            for param in self.forward_model.parameters():
                param.requires_grad = False
            self.forward_model.eval()

    def forward(self, performance_spec):
        """
        Args:
            performance_spec: (batch, 5) desired performance (scaled)

        Returns:
            predicted_geometry: (batch, 10) predicted geometry from inverse model
            reconstructed_performance: (batch, 5) performance predicted by forward model
                                       from the predicted geometry
        """
        # Inverse: performance → geometry
        predicted_geometry = self.inverse_model(performance_spec)

        # Forward: predicted geometry → reconstructed performance
        # Forward model is frozen, so no gradients flow through it
        reconstructed_performance = self.forward_model(predicted_geometry)

        return predicted_geometry, reconstructed_performance

    def predict_geometry(self, performance_spec):
        """Inference: just get the predicted geometry."""
        self.eval()
        with torch.no_grad():
            geometry = self.inverse_model(performance_spec)
        return geometry


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing model architectures...")

    fwd = ForwardModel(input_dim=10, output_dim=5)
    inv = InverseModel(input_dim=5, output_dim=10)
    tandem = TandemNetwork(inv, fwd)

    print(f"\nForward Model: {count_parameters(fwd):,} trainable params")
    print(fwd)

    print(f"\nInverse Model: {count_parameters(inv):,} trainable params")
    print(inv)

    # Test forward pass
    x_geom = torch.randn(4, 10)
    x_perf = torch.randn(4, 5)

    y_perf = fwd(x_geom)
    y_geom = inv(x_perf)
    pred_geom, recon_perf = tandem(x_perf)

    print(f"\nForward:  {x_geom.shape} → {y_perf.shape}")
    print(f"Inverse:  {x_perf.shape} → {y_geom.shape}")
    print(f"Tandem:   {x_perf.shape} → geom {pred_geom.shape}, perf {recon_perf.shape}")
    print("✅ All models working!")
