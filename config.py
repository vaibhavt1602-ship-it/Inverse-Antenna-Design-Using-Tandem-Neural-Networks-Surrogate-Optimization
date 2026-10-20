"""
Configuration file for Inverse Antenna Design using Tandem Network.
All hyperparameters and paths are defined here for easy experimentation.
"""

import os

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Look for data in current folder first, then parent project folder
if os.path.exists(os.path.join(BASE_DIR, "Antenna_Parameters.xlsx")):
    DATA_PATH = os.path.join(BASE_DIR, "Antenna_Parameters.xlsx")
else:
    DATA_PATH = os.path.join(PROJECT_DIR, "Antenna_Parameters.xlsx")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
# Geometry (input) parameters — what the inverse model will PREDICT
GEOMETRY_COLS = [
    "Sub_W",      # Substrate width (mm)
    "Sub_L",      # Substrate length (mm)
    "Sub_H",      # Substrate height (mm)
    "Patch_W",    # Patch width (mm)
    "Patch_L",    # Patch length (mm)
    "Feed_W",     # Feed line width (mm)
    "Slot1_W",    # Slot 1 width (mm)
    "Slot1_L",    # Slot 1 length (mm)
    "Slot2_W",    # Slot 2 width (mm)
    "Slot2_L",    # Slot 2 length (mm)
]

# Performance (output) parameters — what the user will SPECIFY as targets
PERFORMANCE_COLS = [
    "Freq_GHz",      # Operating frequency (GHz)
    "Gain",          # Antenna gain (dBi)
    "Directivity",   # Directivity (dBi)
    "S11(dB)",       # Return loss (dB)
    "VSWR",          # Voltage Standing Wave Ratio
]

# Frequency column (for filtering / multi-frequency handling)
FREQ_COL = "Freq_GHz"

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
TEST_SPLIT = 0.15           # Fraction for test set
VAL_SPLIT = 0.15            # Fraction for validation set (from remaining train)
RANDOM_SEED = 42
FILTER_S11_THRESHOLD = -5   # Keep samples with S11 < -5 dB (near-resonant + resonant)

# ============================================================================
# FORWARD MODEL HYPERPARAMETERS
# ============================================================================
FORWARD_HIDDEN_DIMS = [128, 256, 256, 128]  # Hidden layer sizes
FORWARD_DROPOUT = 0.15
FORWARD_ACTIVATION = "LeakyReLU"
FORWARD_LR = 1e-3
FORWARD_WEIGHT_DECAY = 1e-5
FORWARD_EPOCHS = 500
FORWARD_BATCH_SIZE = 64
FORWARD_PATIENCE = 40       # Early stopping patience
FORWARD_LR_SCHEDULER = "ReduceLROnPlateau"
FORWARD_LR_FACTOR = 0.5
FORWARD_LR_PATIENCE = 20

# ============================================================================
# INVERSE MODEL HYPERPARAMETERS
# ============================================================================
INVERSE_HIDDEN_DIMS = [128, 256, 256, 128]  # Hidden layer sizes
INVERSE_DROPOUT = 0.15
INVERSE_ACTIVATION = "LeakyReLU"
INVERSE_LR = 5e-4
INVERSE_WEIGHT_DECAY = 1e-5
INVERSE_EPOCHS = 600
INVERSE_BATCH_SIZE = 64
INVERSE_PATIENCE = 50

# ============================================================================
# TANDEM NETWORK HYPERPARAMETERS
# ============================================================================
# In tandem training, we connect: Inverse → Frozen Forward
# Loss = α * geometry_loss + β * performance_loss
TANDEM_ALPHA = 0.3           # Weight for geometry reconstruction loss
TANDEM_BETA = 0.7            # Weight for performance prediction loss
TANDEM_LR = 1e-4
TANDEM_WEIGHT_DECAY = 1e-5
TANDEM_EPOCHS = 400
TANDEM_BATCH_SIZE = 64
TANDEM_PATIENCE = 50

# ============================================================================
# OPTIMIZER
# ============================================================================
# AdamW is recommended — it decouples weight decay from gradient updates,
# providing better generalization than vanilla Adam.
OPTIMIZER = "AdamW"

# ============================================================================
# DEVICE
# ============================================================================
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
