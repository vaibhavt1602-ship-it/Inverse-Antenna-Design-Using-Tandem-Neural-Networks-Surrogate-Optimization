# Deep Learning-Based Inverse Antenna Design & Optimization

An automated AI-driven framework for **Inverse Antenna Synthesis** that predicts and optimizes microstrip patch antenna geometries in **seconds**, bypassing computationally expensive EM simulations (CST Studio / Ansys HFSS).

---

## 📌 Answers to Key Project Questions

### 1. What Dataset?
* **Primary High-Frequency Dataset**: `Antenna_Parameters.xlsx` (1,648 frequency-sweep simulation samples of a slotted 5G mmWave microstrip patch antenna operating across 28–50 GHz with rich multi-objective metrics: Gain, Directivity, Return Loss $S_{11}$, VSWR, and Radiation Efficiency).
* **Augmentation / Sub-6GHz Dataset**: `Zenodo.csv` (55,053 frequency sweep points for 1.8–2.8 GHz patch antenna S11 profiles).
* **Clean Combined Training Set**: Filtered for near-resonance ($S_{11} < -5$ dB) yielding **6,177 samples** with 10 geometry variables and 5 RF performance targets.

### 2. Which Model?
* **Tandem Neural Network (TNN)**:
  * **Forward Surrogate Model**: $[G_1, \dots, G_{10}] \to [f_0, \text{Gain}, \text{Directivity}, S_{11}, \text{VSWR}]$ (~135k parameters, Residual Blocks, BatchNorm, LeakyReLU). Replaces the slow full-wave 3D Maxwell equation solver.
  * **Inverse Synthesis Model**: $[f_0, \text{Gain}, \text{Directivity}, S_{11}, \text{VSWR}] \to [G_1, \dots, G_{10}]$ (~135k parameters).
  * **Tandem Loss Training**: The Inverse Network is connected to the frozen Forward Model with compound loss:
    $$\mathcal{L} = 0.3 \cdot \mathcal{L}_{\text{geom}} + 0.7 \cdot \mathcal{L}_{\text{perf}}$$
    *Why?* Solves the notorious **one-to-many non-uniqueness** problem in inverse electromagnetic design.

### 3. Which Optimizer?
* **Hybrid Two-Stage Optimization**:
  1. **Stage 1 (Initial Guess)**: Inverse Neural Network computes the starting geometry $\mathbf{g}_0$ in **$< 4$ milliseconds**.
  2. **Stage 2 (Differentiable Surrogate Gradient Descent)**:
     * Optimizer: **Adam / L-BFGS with Autograd**.
     * Loss function backpropagates through the differentiable Forward Surrogate $\nabla_{\mathbf{g}} \mathcal{L}_{\text{RF}}$ in microseconds.
     * Box constraints project $\mathbf{g}$ onto physical manufacturing limits.
     * **Total runtime**: $\approx 3.5$ seconds vs. 15–45 minutes in CST/HFSS (**$>300\times$ speedup**).
  3. **Alternative Global Optimizer**: **Particle Swarm Optimization (PSO)** for non-convex multi-band targets.

---

## 🚀 Quickstart

### 1. Launch Interactive Streamlit Web App (Recommended)
Launch the graphical browser dashboard:
```bash
streamlit run app.py
```
*(Or double-click `run_app.bat` in the root folder).*

### 2. Run Complete End-to-End Synthesis via CLI
Synthesize an antenna for 28 GHz with target $S_{11} = -20$ dB, Gain = 6.5 dBi:
```bash
python main.py --freq 28.0 --s11 -20.0 --gain 6.5 --opt gradient
```

### 2. Interactive Mode
Run the guided CLI prompt to input custom researcher specs:
```bash
python main.py --interactive
```

### 3. Retrain the Neural Networks
Run the full 3-phase training pipeline:
```bash
python train.py
```

### 4. Fast Inference (Raw Neural Prediction)
```bash
python predict.py --freq 28.0 --s11 -20.0 --gain 6.5
```

---

## 📁 Project Structure

```
inverse_antenna_design/
├── config.py              # Central hyperparameters & paths
├── train.py               # 3-phase training pipeline (Forward -> Inverse -> Tandem)
├── predict.py             # Direct neural inference
├── optimizer.py           # Differentiable Gradient & PSO Surrogate Optimizers
├── main.py                # End-to-end synthesis CLI with CAD export & plots
├── test_optimizer.py      # Standalone verification script for the optimizer
├── models/
│   ├── networks.py        # ForwardModel, InverseModel, TandemNetwork architectures
│   └── trainer.py         # PyTorch training loops & early stopping
├── utils/
│   ├── data_loader.py     # Data extraction, normalization, and splits
│   ├── evaluation.py      # Metric calculations, scatter plots & error analysis
│   └── export_cad.py      # CST Studio (*.vba) & Ansys HFSS (*.py) macro exporters
├── checkpoints/           # Saved PyTorch weights (.pt) and standard scalers (.pkl)
└── results/               # Generated plots, CAD macros, and spec sheets
```

---

## 📊 Outputs & Exported Artifacts

Every synthesis run automatically generates:
1. `results/synthesized_antenna_layout.png`: 2D dimensioned engineering layout.
2. `results/optimizer_convergence.png`: Loss convergence curve.
3. `results/antenna_cst_macro.vba`: 1-click importable macro for CST Microwave Studio.
4. `results/antenna_hfss_script.py`: 1-click Python automation script for Ansys HFSS.
5. `results/antenna_specs.json` & `.csv`: Manufacturing parameter tables.
