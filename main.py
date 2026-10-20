"""
End-to-End Inverse Antenna Design & Optimization CLI.

Combines:
  1. Tandem Neural Network for Instant Initial Guess (<5 ms)
  2. Differentiable EM Surrogate Optimizer for Precision Tuning (<3 sec)
  3. Physical Antenna 2D Layout Visualization
  4. CST Microwave Studio & Ansys HFSS Automated Macro Export

Usage:
  python main.py --freq 28.0 --s11 -20.0 --gain 6.5 --opt gradient
  python main.py --interactive
"""

import argparse
import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from predict import load_trained_models
from optimizer import HybridAntennaSynthesizer
from utils.export_cad import export_cst_macro, export_hfss_script, export_specification_sheet
from config import RESULTS_DIR


def plot_antenna_layout(geom, filename="antenna_layout.png"):
    """
    Plot 2D top-view geometry layout of the synthesized microstrip patch antenna.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    sub_w = geom['Sub_W']
    sub_l = geom['Sub_L']
    patch_w = geom['Patch_W']
    patch_l = geom['Patch_L']
    feed_w = geom['Feed_W']
    slot1_w = geom['Slot1_W']
    slot1_l = geom['Slot1_L']
    slot2_w = geom['Slot2_W']
    slot2_l = geom['Slot2_L']

    # 1. Substrate (gray)
    sub = patches.Rectangle((-sub_w/2, -sub_l/2), sub_w, sub_l,
                            linewidth=1.5, edgecolor='black', facecolor='#e0e0e0', label='Substrate (FR4)')
    ax.add_patch(sub)

    # 2. Patch (copper/orange)
    patch = patches.Rectangle((-patch_w/2, -patch_l/2), patch_w, patch_l,
                              linewidth=1.2, edgecolor='#b85d19', facecolor='#f4a261', label='Patch (Copper)')
    ax.add_patch(patch)

    # 3. Feed Line
    feed_len = (sub_l - patch_l) / 2
    feed = patches.Rectangle((-feed_w/2, -sub_l/2), feed_w, feed_len,
                             linewidth=1, edgecolor='#b85d19', facecolor='#e76f51', label='Microstrip Feed')
    ax.add_patch(feed)

    # 4. Slots (white cuts)
    slot1 = patches.Rectangle((-slot1_w/2, 0), slot1_w, slot1_l,
                              linewidth=1, edgecolor='blue', facecolor='white', label='Slots (Etched)')
    slot2 = patches.Rectangle((-slot2_w/2, -slot2_l), slot2_w, slot2_l,
                              linewidth=1, edgecolor='blue', facecolor='white')
    ax.add_patch(slot1)
    ax.add_patch(slot2)

    # Dimensions annotations
    ax.text(0, sub_l/2 + 0.5, f"Sub_W = {sub_w:.2f} mm", ha='center', fontsize=10, weight='bold')
    ax.text(sub_w/2 + 0.5, 0, f"Sub_L = {sub_l:.2f} mm", va='center', rotation=-90, fontsize=10, weight='bold')
    ax.text(0, patch_l/4, f"Patch: {patch_w:.2f} × {patch_l:.2f} mm", ha='center', fontsize=9, color='#8a3b00')

    pad = max(sub_w, sub_l) * 0.2
    ax.set_xlim(-sub_w/2 - pad, sub_w/2 + pad)
    ax.set_ylim(-sub_l/2 - pad, sub_l/2 + pad)
    ax.set_aspect('equal')
    ax.set_title("Synthesized 5G Microstrip Patch Antenna Layout", fontsize=13, pad=15)
    ax.set_xlabel("Width X (mm)")
    ax.set_ylabel("Length Y (mm)")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.savefig(filename, dpi=180)
    plt.close()
    return filename


def plot_convergence(opt_history, filename="optimization_convergence.png"):
    """Plot optimizer loss curve."""
    if not opt_history or not opt_history.get("loss"):
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(opt_history["loss"], color='#2a9d8f', linewidth=2.2, label='Objective Cost')
    ax.set_title("Differentiable Surrogate Optimization Convergence", fontsize=12)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename


def run_synthesis(specs, optimizer_type="gradient"):
    print("=" * 70)
    print("🛰️  INVERSE ANTENNA DESIGN & SURROGATE OPTIMIZATION SYSTEM")
    print("=" * 70)

    print("\n📋 Desired RF Specifications:")
    print("-" * 50)
    for k, v in specs.items():
        unit = "GHz" if "Freq" in k else "dB" if "dB" in k or "Gain" in k or "Directivity" in k else ""
        print(f"  {k:<15}: {v:>10.4f} {unit}")

    # Load system
    tandem_model, scaler_geom, scaler_perf = load_trained_models()
    synthesizer = HybridAntennaSynthesizer(tandem_model, scaler_geom, scaler_perf)

    print(f"\n🚀 Running Synthesis Pipeline [Optimizer: {optimizer_type.upper()}]...")
    res = synthesizer.synthesize(specs, optimizer_type=optimizer_type, verbose=False)

    print("\n" + "=" * 70)
    print("✅ SYNTHESIS COMPLETE")
    print("=" * 70)
    print(f"  Stage 1 (Inverse Neural Net)  : {res['nn_time_ms']:.2f} ms")
    print(f"  Stage 2 (Surrogate Optimizer) : {res['opt_time_ms']:.2f} ms")
    print(f"  Total Synthesis Duration      : {res['total_time_ms']:.2f} ms ({res['total_time_ms']/1000.0:.3f} s)")
    print(f"  ⚡ Computational Acceleration : {res['speedup_vs_cst']:,.0f}x faster than traditional EM simulation!")

    print("\n📐 Synthesized Antenna Physical Dimensions (mm):")
    print("-" * 50)
    for k, v in res['refined_geometry'].items():
        print(f"  {k:<15}: {v:>10.4f} mm")

    print("\n📡 Simulated Performance (Surrogate Validation):")
    print("-" * 50)
    for k, v in res['final_performance'].items():
        target = specs.get(k, 0.0)
        err = abs((v - target) / target) * 100 if abs(target) > 1e-5 else 0
        status = "✅" if err < 15 else "⚠️" if err < 30 else "❌"
        print(f"  {k:<15}: {v:>10.4f}  (target: {target:>8.4f}, err: {err:>5.1f}%) {status}")

    # Generate visualizations & exports
    os.makedirs(RESULTS_DIR, exist_ok=True)
    layout_path = os.path.join(RESULTS_DIR, "synthesized_antenna_layout.png")
    plot_antenna_layout(res['refined_geometry'], layout_path)
    print(f"\n🎨 Saved 2D Antenna Layout  : {layout_path}")

    conv_path = os.path.join(RESULTS_DIR, "optimizer_convergence.png")
    plot_convergence(res['opt_history'], conv_path)
    print(f"📈 Saved Optimizer Convergence: {conv_path}")

    # Export CST and HFSS scripts
    cst_file = os.path.join(RESULTS_DIR, "antenna_cst_macro.vba")
    export_cst_macro(res['refined_geometry'], cst_file)
    print(f"💾 Exported CST Studio VBA Macro: {cst_file}")

    hfss_file = os.path.join(RESULTS_DIR, "antenna_hfss_script.py")
    export_hfss_script(res['refined_geometry'], hfss_file)
    print(f"💾 Exported Ansys HFSS Script   : {hfss_file}")

    json_p, csv_p = export_specification_sheet(
        res['refined_geometry'], res['final_performance'],
        os.path.join(RESULTS_DIR, "antenna_specs")
    )
    print(f"📄 Saved Spec Sheets            : {json_p}, {csv_p}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Inverse Antenna Design Synthesis Tool")
    parser.add_argument("--freq", type=float, default=28.0, help="Target Frequency (GHz)")
    parser.add_argument("--s11", type=float, default=-20.0, help="Target Return Loss S11 (dB)")
    parser.add_argument("--gain", type=float, default=6.5, help="Target Gain (dBi)")
    parser.add_argument("--directivity", type=float, default=7.5, help="Target Directivity (dBi)")
    parser.add_argument("--vswr", type=float, default=1.2, help="Target VSWR")
    parser.add_argument("--opt", type=str, default="gradient", choices=["gradient", "pso", "none"],
                        help="Optimizer type: 'gradient', 'pso', or 'none'")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive mode")

    args = parser.parse_args()

    if args.interactive:
        print("\n🎮 Interactive Antenna Specification Input:")
        try:
            f = float(input("  Operating Frequency in GHz [default: 28.0]: ") or 28.0)
            s = float(input("  Target S11 in dB [default: -20.0]: ") or -20.0)
            g = float(input("  Target Gain in dBi [default: 6.5]: ") or 6.5)
            d = float(input("  Target Directivity in dBi [default: 7.5]: ") or 7.5)
            v = float(input("  Target VSWR [default: 1.2]: ") or 1.2)
            opt = input("  Optimizer ('gradient'/'pso'/'none') [default: gradient]: ") or "gradient"
        except (ValueError, KeyboardInterrupt):
            print("\nExiting.")
            return
        specs = {"Freq_GHz": f, "Gain": g, "Directivity": d, "S11(dB)": s, "VSWR": v}
        run_synthesis(specs, optimizer_type=opt)
    else:
        specs = {
            "Freq_GHz": args.freq,
            "Gain": args.gain,
            "Directivity": args.directivity,
            "S11(dB)": args.s11,
            "VSWR": args.vswr
        }
        run_synthesis(specs, optimizer_type=args.opt)


if __name__ == "__main__":
    main()
