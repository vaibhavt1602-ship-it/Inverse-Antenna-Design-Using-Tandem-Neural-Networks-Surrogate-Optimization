# 🛰️ AI-Powered Inverse Antenna Design & Optimization

An end-to-end Machine Learning system that predicts and optimizes **5G Microstrip Patch Antenna** physical geometries in **under 4 seconds** — replacing hours of slow, expensive 3D electromagnetic simulations (CST Studio / Ansys HFSS).

---

## 💡 The Problem

Designing an antenna traditionally requires a tedious trial-and-error process:
1. An engineer guesses initial dimensions.
2. They open CST Studio or Ansys HFSS and run a 3D electromagnetic simulation solving Maxwell's equations.
3. **Each simulation takes 20 to 45 minutes.**
4. If an optimizer tests 200–300 parameter variations, **it takes 50 to 100+ hours** of compute time just to design a single antenna.

---

## ⚡ What This Project Does

Instead of running heavy 3D simulations repeatedly, this project uses a **Two-Stage Hybrid AI pipeline**:

1. **Stage 1: Instant Geometry Synthesis (Tandem Neural Network)**
   * You input your target specifications (e.g., `Frequency = 28 GHz`, `S11 = -20 dB`, `Gain = 6.5 dBi`).
   * The **Inverse Neural Model** predicts the required physical millimeter dimensions in **~3.5 milliseconds**.
   * *Why a Tandem Network?* Inverse antenna design suffers from a "one-to-many" problem (multiple geometries can produce the same frequency). Connecting the Inverse model to a frozen Forward model enforces electromagnetic consistency and prevents invalid designs.

2. **Stage 2: Precision Fine-Tuning (Surrogate-Assisted Optimization)**
   * The initial guess is refined using **gradient descent (Adam Autograd)** through our differentiable **Forward Surrogate Model** (trained on thousands of real CST/HFSS runs).
   * It fine-tunes the physical dimensions to hit the exact target frequency with physical manufacturing constraints in **~3.5 seconds**.

### ⏱️ The Result
* **Traditional CST Optimization**: ~100 hours (300 simulations)
* **Our AI Framework**: **~3.5 seconds** (0 heavy simulations during design)
* **Final Step**: The researcher runs CST **exactly once** at the very end to verify the design before sending it to a PCB manufacturer.

---

## 🛠️ Features

* **Interactive Streamlit Web Dashboard**: Sliders for target frequency (24–50 GHz), Return Loss ($S_{11}$), Gain, and VSWR.
* **2D Engineering CAD Blueprint**: Real-time dimensioned schematic showing substrate, radiating patch, 50Ω feed line, and impedance-matching slots.
* **Fabrication Feasibility Checks**: Automatically ensures the patch fits on the board, slot cuts fit inside the patch, and dimensions are within PCB manufacturing tolerances.
* **1-Click CAD Macro Exporter**:
  * ⬇️ **CST Microwave Studio VBA Macro (`.vba`)**: Builds the entire 3D antenna in CST in 1 second.
  * ⬇️ **Ansys HFSS Python Script (`.py`)**: Automates 3D modeling in HFSS via PyAEDT.
  * ⬇️ **Manufacturing Specs (`.json` & `.csv`)**: Clean tables for PCB fabrication.

---

## 🚀 Quickstart: Running Locally

### 1. Clone the repository
```bash
git clone https://github.com/vaibhavt1602-ship-it/Inverse-Antenna-Design-Using-Tandem-Neural-Networks-Surrogate-Optimization.git
cd Inverse-Antenna-Design-Using-Tandem-Neural-Networks-Surrogate-Optimization
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to use the interactive dashboard.

---

## 💻 CLI Usage (For Scripts / Batch Synthesis)

You can also run synthesis directly from the terminal:

```bash
# Synthesize an antenna for 28 GHz mmWave with S11 = -20 dB and Gain = 6.5 dBi
python main.py --freq 28.0 --s11 -20.0 --gain 6.5 --opt gradient
```

Or run fast inference without the optimizer:
```bash
python predict.py --freq 28.0 --s11 -20.0 --gain 6.5
```

---

## 📁 Repository Structure

```
├── app.py                # Interactive Streamlit Web Dashboard
├── config.py             # Hyperparameters, column definitions & physical bounds
├── main.py               # End-to-end synthesis CLI with CAD export & plots
├── optimizer.py          # Differentiable Gradient & PSO Surrogate Optimizers
├── predict.py            # Fast neural inference script
├── train.py              # 3-phase Tandem Network training pipeline
├── requirements.txt      # Python dependencies
├── checkpoints/          # Trained PyTorch model weights & scalers (~2 MB total)
│   ├── forward_model_final.pt
│   ├── inverse_model_final.pt
│   ├── scaler_geometry.pkl
│   └── scaler_performance.pkl
├── models/
│   ├── networks.py       # ForwardModel, InverseModel, TandemNetwork architectures
│   └── trainer.py        # PyTorch training loops with early stopping
└── utils/
    ├── data_loader.py    # Dataset loading & preprocessing
    ├── evaluation.py     # Metrics (R2, MAE) and scatter plotting
    └── export_cad.py     # CST Studio (.vba) and Ansys HFSS (.py) script generators
```

---

## 📊 Dataset & Model Details

* **Training Data**: Combined dataset of **6,177 samples** filtered for operational resonance ($S_{11} < -5$ dB) across 5G mmWave bands (28–50 GHz) and sub-6GHz frequencies.
* **Architecture**: Deep fully connected networks with **Residual Blocks**, **Batch Normalization**, and **LeakyReLU** activations.
* **Model Size**: ~135,000 parameters per network (<1 MB on disk).
* **Accuracy on Unseen Test Data**:
  * Resonance Frequency: **$R^2 = 0.985$** (Mean Absolute Error $\approx 0.45\text{ GHz}$)

---

## 👥 Authors & Acknowledgments

Developed as part of research into machine learning surrogate models for radio frequency (RF) and antenna engineering.
