"""
Surrogate-Assisted Optimization Engine for Inverse Antenna Design.

This module answers the "Which optimizer?" question:
Instead of running computationally prohibitive EM simulations (CST/HFSS: ~15-45 minutes per run),
we use the trained differentiable Neural Forward Surrogate f(g) to evaluate and optimize
antenna geometry in milliseconds.

Optimizers implemented:
  1. GradientDescentOptimizer (Differentiable PyTorch Autograd + Adam/L-BFGS):
     - Calculates analytical gradients d(Loss)/d(geometry) via backpropagation
       through the frozen Forward Surrogate.
     - Extremely fast (~50-100 ms for 200 iterations).
  2. ParticleSwarmOptimizer (PSO / Metaheuristic):
     - Derivative-free global search for non-convex or multi-modal target specs.
  3. HybridTwoStageOptimizer:
     - Stage 1: Inverse Neural Network provides warm-start initial guess g_0 (< 1 ms).
     - Stage 2: Gradient-based fine-tuning refines g_0 to exactly reach target specs (< 50 ms).
"""

import numpy as np
import torch
import torch.nn as nn
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEOMETRY_COLS, PERFORMANCE_COLS, DEVICE


class AntennaLossFunction:
    """
    Multi-objective cost function for antenna parameter optimization:
      Loss(g) = w_freq * (f_pred - f_target)^2
              + w_s11  * max(0, s11_pred - s11_target)^2  (we want s11 <= s11_target)
              + w_gain * max(0, gain_target - gain_pred)^2 (we want gain >= gain_target)
              + physical_penalty(g) (box constraints & manufacturing feasibility)
    """
    def __init__(self, target_specs, scaler_geom, scaler_perf,
                 w_freq=5.0, w_s11=2.0, w_gain=1.0, w_vswr=1.0):
        self.target_specs = target_specs
        self.scaler_geom = scaler_geom
        self.scaler_perf = scaler_perf
        self.w_freq = w_freq
        self.w_s11 = w_s11
        self.w_gain = w_gain
        self.w_vswr = w_vswr

        # Geometry bounds (min, max) in physical mm
        self.geom_mean = torch.tensor(scaler_geom.mean_, dtype=torch.float32, device=DEVICE)
        self.geom_std = torch.tensor(scaler_geom.scale_, dtype=torch.float32, device=DEVICE)
        self.perf_mean = torch.tensor(scaler_perf.mean_, dtype=torch.float32, device=DEVICE)
        self.perf_std = torch.tensor(scaler_perf.scale_, dtype=torch.float32, device=DEVICE)

    def __call__(self, geom_scaled, forward_surrogate):
        """Compute loss for scaled geometry tensor."""
        # Forward pass through surrogate
        pred_perf_scaled = forward_surrogate(geom_scaled)

        # Unscale performance for physical objective evaluation
        pred_perf = pred_perf_scaled * self.perf_std + self.perf_mean
        geom_unscaled = geom_scaled * self.geom_std + self.geom_mean

        # Extract predicted metrics: [Freq_GHz, Gain, Directivity, S11(dB), VSWR]
        f_pred = pred_perf[:, 0]
        gain_pred = pred_perf[:, 1]
        s11_pred = pred_perf[:, 3]
        vswr_pred = pred_perf[:, 4]

        # Target specs
        f_target = self.target_specs.get("Freq_GHz", 28.0)
        s11_target = self.target_specs.get("S11(dB)", -15.0)
        gain_target = self.target_specs.get("Gain", 6.0)
        vswr_target = self.target_specs.get("VSWR", 1.5)

        # Objectives
        loss_f = self.w_freq * torch.mean((f_pred - f_target) ** 2)
        # S11 objective: penalize when S11 is worse (higher) than target
        loss_s11 = self.w_s11 * torch.mean(torch.relu(s11_pred - s11_target) ** 2)
        # Gain objective: penalize when Gain is lower than target
        loss_gain = self.w_gain * torch.mean(torch.relu(gain_target - gain_pred) ** 2)
        # VSWR objective: penalize when VSWR > target
        loss_vswr = self.w_vswr * torch.mean(torch.relu(vswr_pred - vswr_target) ** 2)

        # Physical feasibility penalties (Patch must fit on Substrate, etc.)
        # Sub_W > Patch_W, Sub_L > Patch_L
        sub_w = geom_unscaled[:, 0]
        sub_l = geom_unscaled[:, 1]
        patch_w = geom_unscaled[:, 3]
        patch_l = geom_unscaled[:, 4]

        # Negative dimension penalty
        penalty_pos = 10.0 * torch.sum(torch.relu(-geom_unscaled) ** 2)
        # Overlap penalty (Substrate must be strictly larger than Patch)
        penalty_w = 10.0 * torch.sum(torch.relu(patch_w - sub_w + 0.5) ** 2)
        penalty_l = 10.0 * torch.sum(torch.relu(patch_l - sub_l + 0.5) ** 2)

        total_loss = loss_f + loss_s11 + loss_gain + loss_vswr + penalty_pos + penalty_w + penalty_l

        return total_loss, {
            "loss_f": loss_f.item(),
            "loss_s11": loss_s11.item(),
            "loss_gain": loss_gain.item(),
            "loss_vswr": loss_vswr.item(),
            "f_pred": f_pred.detach().cpu().numpy()[0],
            "s11_pred": s11_pred.detach().cpu().numpy()[0],
            "gain_pred": gain_pred.detach().cpu().numpy()[0],
            "vswr_pred": vswr_pred.detach().cpu().numpy()[0],
        }


class GradientDescentOptimizer:
    """
    Surrogate-Assisted Gradient Optimizer (PyTorch Autograd + Adam / L-BFGS).
    Optimizes geometry parameters directly by backpropagating through
    the frozen forward surrogate.
    """
    def __init__(self, forward_surrogate, scaler_geom, scaler_perf,
                 lr=0.05, max_iter=200, method="Adam"):
        self.forward_surrogate = forward_surrogate
        self.forward_surrogate.eval()
        for p in self.forward_surrogate.parameters():
            p.requires_grad = False

        self.scaler_geom = scaler_geom
        self.scaler_perf = scaler_perf
        self.lr = lr
        self.max_iter = max_iter
        self.method = method

    def optimize(self, initial_geom_unscaled, target_specs, verbose=False):
        """
        Run gradient-based optimization starting from initial_geom_unscaled.

        Args:
            initial_geom_unscaled: np.ndarray (10,) or (1, 10) initial geometry
            target_specs: dict of target performance specs
            verbose: bool

        Returns:
            best_geom_unscaled: np.ndarray (10,) optimal geometry
            history: dict of optimization trajectory
        """
        start_time = time.time()

        # Scale initial geometry
        init_arr = np.array(initial_geom_unscaled, dtype=np.float32).reshape(1, -1)
        init_scaled = self.scaler_geom.transform(init_arr)

        # Optimization variable (requires grad)
        geom_param = torch.tensor(init_scaled, dtype=torch.float32,
                                  device=DEVICE, requires_grad=True)

        loss_fn = AntennaLossFunction(target_specs, self.scaler_geom, self.scaler_perf)

        if self.method == "Adam":
            optimizer = torch.optim.Adam([geom_param], lr=self.lr)
        elif self.method == "SGD":
            optimizer = torch.optim.SGD([geom_param], lr=self.lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam([geom_param], lr=self.lr)

        history = {"loss": [], "f_pred": [], "s11_pred": [], "gain_pred": []}
        best_loss = float("inf")
        best_geom = None

        # Box bounds in scaled space (clamp to +/- 2.5 std devs)
        scaled_min = torch.tensor(self.scaler_geom.transform(np.array([[5.5, 5.5, 0.18, 4.4, 1.5, 0.5, 0.3, 0.8, 0.25, 0.5]]))[0], dtype=torch.float32, device=DEVICE)
        scaled_max = torch.tensor(self.scaler_geom.transform(np.array([[45.0, 42.0, 1.6, 44.0, 29.3, 3.0, 5.0, 10.0, 0.35, 10.0]]))[0], dtype=torch.float32, device=DEVICE)

        for it in range(self.max_iter):
            optimizer.zero_grad()
            total_loss, metrics = loss_fn(geom_param, self.forward_surrogate)
            total_loss.backward()

            # Gradient clipping to prevent erratic geometry jumps
            torch.nn.utils.clip_grad_norm_([geom_param], max_norm=1.0)
            optimizer.step()

            # Project onto box constraints
            with torch.no_grad():
                geom_param.clamp_(min=scaled_min, max=scaled_max)

            cur_loss = total_loss.item()
            history["loss"].append(cur_loss)
            history["f_pred"].append(metrics["f_pred"])
            history["s11_pred"].append(metrics["s11_pred"])
            history["gain_pred"].append(metrics["gain_pred"])

            if cur_loss < best_loss:
                best_loss = cur_loss
                best_geom = geom_param.detach().clone()

            if verbose and (it + 1) % 25 == 0:
                print(f"  [Iter {it+1:3d}/{self.max_iter}] Loss: {cur_loss:.4f} | "
                      f"Freq: {metrics['f_pred']:.2f} GHz | S11: {metrics['s11_pred']:.1f} dB | "
                      f"Gain: {metrics['gain_pred']:.2f} dBi")

        elapsed_ms = (time.time() - start_time) * 1000.0

        # Unscale best geometry
        best_geom_np = best_geom.cpu().numpy()
        best_geom_unscaled = self.scaler_geom.inverse_transform(best_geom_np)[0]

        history["elapsed_ms"] = elapsed_ms
        history["best_loss"] = best_loss

        return best_geom_unscaled, history


class ParticleSwarmOptimizer:
    """
    Particle Swarm Optimization (PSO) on the Surrogate landscape.
    Best for finding global minima across highly multi-modal frequency bands.
    """
    def __init__(self, forward_surrogate, scaler_geom, scaler_perf,
                 n_particles=40, max_iter=100, c1=1.5, c2=1.5, w=0.7):
        self.forward_surrogate = forward_surrogate
        self.forward_surrogate.eval()
        self.scaler_geom = scaler_geom
        self.scaler_perf = scaler_perf
        self.n_particles = n_particles
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def optimize(self, target_specs, bounds=None, warm_start=None, verbose=False):
        """
        Run PSO using forward surrogate evaluations.
        """
        start_time = time.time()
        dim = len(GEOMETRY_COLS)

        if bounds is None:
            # Scaled bounds around [-2.5, 2.5] standard deviations
            lb = -2.5 * np.ones(dim, dtype=np.float32)
            ub = 2.5 * np.ones(dim, dtype=np.float32)
        else:
            lb, ub = bounds

        # Initialize particles
        particles = np.random.uniform(lb, ub, size=(self.n_particles, dim)).astype(np.float32)
        if warm_start is not None:
            # Seed the first particle with the warm-start inverse prediction
            ws_scaled = self.scaler_geom.transform(np.array(warm_start).reshape(1, -1))[0]
            particles[0] = ws_scaled

        velocities = np.zeros_like(particles)
        pbest = particles.copy()
        pbest_scores = np.full(self.n_particles, float("inf"))

        loss_fn = AntennaLossFunction(target_specs, self.scaler_geom, self.scaler_perf)

        gbest = None
        gbest_score = float("inf")
        history = {"loss": []}

        for it in range(self.max_iter):
            t_particles = torch.tensor(particles, dtype=torch.float32, device=DEVICE)
            with torch.no_grad():
                pred_perf = self.forward_surrogate(t_particles)
                pred_perf_unscaled = pred_perf * loss_fn.perf_std + loss_fn.perf_mean

                f_pred = pred_perf_unscaled[:, 0]
                s11_pred = pred_perf_unscaled[:, 3]
                gain_pred = pred_perf_unscaled[:, 1]

                f_target = target_specs.get("Freq_GHz", 28.0)
                s11_target = target_specs.get("S11(dB)", -15.0)
                gain_target = target_specs.get("Gain", 6.0)

                losses = (
                    loss_fn.w_freq * (f_pred - f_target) ** 2
                    + loss_fn.w_s11 * torch.relu(s11_pred - s11_target) ** 2
                    + loss_fn.w_gain * torch.relu(gain_target - gain_pred) ** 2
                ).cpu().numpy()

            for i in range(self.n_particles):
                if losses[i] < pbest_scores[i]:
                    pbest_scores[i] = losses[i]
                    pbest[i] = particles[i].copy()

                if losses[i] < gbest_score:
                    gbest_score = losses[i]
                    gbest = particles[i].copy()

            history["loss"].append(gbest_score)

            # Update velocity and position
            r1 = np.random.rand(self.n_particles, dim)
            r2 = np.random.rand(self.n_particles, dim)
            velocities = (self.w * velocities
                          + self.c1 * r1 * (pbest - particles)
                          + self.c2 * r2 * (gbest - particles))
            particles = np.clip(particles + velocities, lb, ub)

            if verbose and (it + 1) % 25 == 0:
                print(f"  [PSO Iter {it+1:3d}/{self.max_iter}] Best Loss: {gbest_score:.4f}")

        elapsed_ms = (time.time() - start_time) * 1000.0
        best_geom_unscaled = self.scaler_geom.inverse_transform(gbest.reshape(1, -1))[0]
        history["elapsed_ms"] = elapsed_ms
        history["best_loss"] = gbest_score

        return best_geom_unscaled, history


class HybridAntennaSynthesizer:
    """
    Two-Stage Hybrid Optimization Pipeline (The Complete Inverse Solution):
      Stage 1: Inverse Neural Network computes initial geometry guess in <1 ms.
      Stage 2: Differentiable Gradient Optimizer refines geometry to exact targets in <50 ms.
    """
    def __init__(self, tandem_model, scaler_geom, scaler_perf):
        self.tandem_model = tandem_model
        self.scaler_geom = scaler_geom
        self.scaler_perf = scaler_perf
        self.grad_opt = GradientDescentOptimizer(
            self.tandem_model.forward_model, scaler_geom, scaler_perf
        )
        self.pso_opt = ParticleSwarmOptimizer(
            self.tandem_model.forward_model, scaler_geom, scaler_perf
        )

    def synthesize(self, target_specs, optimizer_type="gradient", verbose=True):
        """
        Synthesize antenna geometry from performance specs.

        Args:
            target_specs: dict containing desired RF specifications
            optimizer_type: 'gradient' or 'pso' or 'none' (just raw NN)
            verbose: bool

        Returns:
            result: dict containing initial geometry, refined geometry,
                    simulated performance, and speedup comparison
        """
        # Step 1: Inverse NN Initial Guess
        t0 = time.time()
        perf_array = np.array([[target_specs[col] for col in PERFORMANCE_COLS]], dtype=np.float32)
        perf_scaled = self.scaler_perf.transform(perf_array)
        t_perf = torch.from_numpy(perf_scaled).to(DEVICE)

        with torch.no_grad():
            init_geom_scaled, init_perf_scaled = self.tandem_model(t_perf)

        init_geom = self.scaler_geom.inverse_transform(init_geom_scaled.cpu().numpy())[0]
        init_perf = self.scaler_perf.inverse_transform(init_perf_scaled.cpu().numpy())[0]
        nn_time_ms = (time.time() - t0) * 1000.0

        # Step 2: Optimization Fine-Tuning
        if optimizer_type == "gradient":
            refined_geom, opt_history = self.grad_opt.optimize(
                init_geom, target_specs, verbose=verbose
            )
            opt_time_ms = opt_history["elapsed_ms"]
        elif optimizer_type == "pso":
            refined_geom, opt_history = self.pso_opt.optimize(
                target_specs, warm_start=init_geom, verbose=verbose
            )
            opt_time_ms = opt_history["elapsed_ms"]
        else:
            refined_geom = init_geom
            opt_history = {"elapsed_ms": 0.0, "loss": []}
            opt_time_ms = 0.0

        # Enforce physical constraints / bounds
        # All dimensions must be strictly positive, and Patch must fit on Substrate
        refined_geom = np.maximum(refined_geom, 0.1)
        refined_geom[0] = max(refined_geom[0], refined_geom[3] + 0.5)  # Sub_W > Patch_W + 0.5
        refined_geom[1] = max(refined_geom[1], refined_geom[4] + 0.5)  # Sub_L > Patch_L + 0.5

        # Step 3: Validate refined geometry with forward surrogate
        ref_geom_scaled = self.scaler_geom.transform(refined_geom.reshape(1, -1))
        with torch.no_grad():
            final_perf_scaled = self.tandem_model.forward_model(
                torch.tensor(ref_geom_scaled, dtype=torch.float32, device=DEVICE)
            )
        final_perf = self.scaler_perf.inverse_transform(final_perf_scaled.cpu().numpy())[0]

        total_time_ms = nn_time_ms + opt_time_ms
        # Benchmark comparison: typical CST/HFSS run is ~1200 seconds (20 mins)
        cst_time_s = 1200.0
        speedup = (cst_time_s * 1000.0) / max(total_time_ms, 1.0)

        initial_geom_dict = {col: float(init_geom[i]) for i, col in enumerate(GEOMETRY_COLS)}
        refined_geom_dict = {col: float(refined_geom[i]) for i, col in enumerate(GEOMETRY_COLS)}
        initial_perf_dict = {col: float(init_perf[i]) for i, col in enumerate(PERFORMANCE_COLS)}
        final_perf_dict = {col: float(final_perf[i]) for i, col in enumerate(PERFORMANCE_COLS)}

        return {
            "initial_geometry": initial_geom_dict,
            "refined_geometry": refined_geom_dict,
            "initial_performance": initial_perf_dict,
            "final_performance": final_perf_dict,
            "nn_time_ms": nn_time_ms,
            "opt_time_ms": opt_time_ms,
            "total_time_ms": total_time_ms,
            "speedup_vs_cst": speedup,
            "opt_history": opt_history,
        }
