"""
Benchmark U-Net CPU inference at different resolutions and configurations.
Measures wall-clock time for forward passes to determine viable CPU setups.
"""
import time
import torch
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from model.model import UNet

def benchmark(resolution, base_features, num_warmup=3, num_runs=20):
    model = UNet(in_channels=2, out_channels=1, base_features=base_features, dropout=0.0)
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    
    dummy = torch.randn(1, 2, resolution, resolution)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)
    
    mean_ms = np.mean(times)
    std_ms = np.std(times)
    fps = 1000.0 / mean_ms
    
    return num_params, mean_ms, std_ms, fps


print("=" * 80)
print("U-Net CPU Inference Benchmark")
print("=" * 80)
print(f"{'Config':<30} {'Params':>10} {'Mean (ms)':>12} {'Std (ms)':>10} {'FPS':>8}")
print("-" * 80)

configs = [
    # (resolution, base_features, label)
    (256, 64, "256x256, bf=64 (current)"),
    (256, 32, "256x256, bf=32"),
    (256, 16, "256x256, bf=16"),
    (128, 64, "128x128, bf=64"),
    (128, 32, "128x128, bf=32"),
    (128, 16, "128x128, bf=16"),
    (64, 64,  "64x64, bf=64"),
    (64, 32,  "64x32, bf=32"),
    (64, 16,  "64x64, bf=16"),
]

for res, bf, label in configs:
    params, mean, std, fps = benchmark(res, bf)
    print(f"  {label:<28} {params:>8.2f}M {mean:>10.1f}ms {std:>8.1f}ms {fps:>7.1f}")

print("-" * 80)

# Also try ONNX Runtime if available
try:
    import onnxruntime as ort
    print("\nONNX Runtime detected! Benchmarking ONNX export...")
    
    model = UNet(in_channels=2, out_channels=1, base_features=64, dropout=0.0)
    model.eval()
    dummy = torch.randn(1, 2, 128, 128)
    
    onnx_path = "benchmark_unet.onnx"
    torch.onnx.export(model, dummy, onnx_path, opset_version=11,
                      input_names=["input"], output_names=["output"])
    
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    input_np = dummy.numpy()
    # Warmup
    for _ in range(3):
        sess.run(None, {"input": input_np})
    
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        sess.run(None, {"input": input_np})
        times.append((time.perf_counter() - t0) * 1000)
    
    mean_ms = np.mean(times)
    fps = 1000.0 / mean_ms
    print(f"  ONNX RT 128x128, bf=64:       31.04M   {mean_ms:>10.1f}ms            {fps:>7.1f}")
    
    os.remove(onnx_path)
except ImportError:
    print("\n[info] onnxruntime not installed. Install with: pip install onnxruntime")

print("\n" + "=" * 80)
print("Target: >= 15 FPS for usable interactive simulation")
print("=" * 80)
