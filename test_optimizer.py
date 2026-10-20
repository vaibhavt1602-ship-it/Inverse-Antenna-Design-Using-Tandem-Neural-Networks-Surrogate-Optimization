import sys
sys.stdout.reconfigure(encoding='utf-8')
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from predict import load_trained_models
from optimizer import HybridAntennaSynthesizer

print("Loading trained models...")
tandem_model, scaler_geom, scaler_perf = load_trained_models()
synthesizer = HybridAntennaSynthesizer(tandem_model, scaler_geom, scaler_perf)

specs = {
    'Freq_GHz': 28.0,
    'Gain': 6.5,
    'Directivity': 7.5,
    'S11(dB)': -20.0,
    'VSWR': 1.2
}

print(f"\nTarget Specs: {specs}")
print("Synthesizing antenna geometry with Differentiable Gradient Optimizer...")
res = synthesizer.synthesize(specs, optimizer_type='gradient', verbose=True)

print("\n" + "="*60)
print("🎯 HYBRID SYNTHESIS SUMMARY")
print("="*60)
print(f"Neural Initial Guess: {res['nn_time_ms']:.2f} ms")
print(f"Gradient Optimization: {res['opt_time_ms']:.2f} ms")
print(f"Total Computation Time: {res['total_time_ms']:.2f} ms ({res['total_time_ms']/1000:.3f} seconds)")
print(f"⚡ Speedup vs Traditional EM (CST/HFSS): {res['speedup_vs_cst']:,.0f}x faster!")

print("\n📐 Optimized Geometry (mm):")
for k, v in res['refined_geometry'].items():
    print(f"  {k:<12}: {v:8.4f} mm")

print("\n📡 Simulated Performance (Surrogate Validation):")
for k, v in res['final_performance'].items():
    target = specs[k]
    err = abs((v - target) / target) * 100 if abs(target) > 1e-5 else 0
    print(f"  {k:<12}: {v:8.4f}  (target: {target:8.4f}, err: {err:5.1f}%)")
