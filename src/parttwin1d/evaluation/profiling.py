"""Render latency profiling that separates warm-up (JIT/allocator) from steady state."""
from __future__ import annotations

import statistics
import time


def profile_render(renderer, camera, n_warmup: int = 5, n_frames: int = 50, segmentation: bool = False) -> dict:
    import torch

    dev = renderer.device
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    renderer.render(camera, segmentation=segmentation)
    first_ms = (time.perf_counter() - t0) * 1e3
    for _ in range(max(0, n_warmup - 1)):
        renderer.render(camera, segmentation=segmentation)
    times = []
    for _ in range(n_frames):
        torch.cuda.synchronize(dev)
        t = time.perf_counter()
        renderer.render(camera, segmentation=segmentation)
        times.append((time.perf_counter() - t) * 1e3)
    med = statistics.median(times)
    return {
        "resolution": f"{camera.width}x{camera.height}", "gaussians": renderer.gaussian_count,
        "segmentation_pass": segmentation, "first_frame_ms": first_ms,
        "steady_median_ms": med, "steady_p95_ms": sorted(times)[int(0.95 * (len(times) - 1))],
        "steady_fps": 1000.0 / med if med > 0 else float("inf"),
        "peak_gpu_mem_mb": torch.cuda.max_memory_allocated(dev) / 2**20,
        "frames": n_frames, "warmup": n_warmup,
    }
