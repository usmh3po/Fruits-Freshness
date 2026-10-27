"""
Efficiency benchmark for a trained run: FLOPs, latency (p50/p90), throughput, inference memory, and
the effect of fp16 (GPU) and int8 post-training quantization (CPU) on latency and test accuracy.

Reads runs/<run>/best.pt and runs/split_seed42.csv, writes runs/<run>/efficiency.json.

Usage:  python benchmark_inference.py --run s4x40_reg
"""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.flop_counter import FlopCounterMode

from eda import FRESHNESS_ORDER
from train import FreshnessNet, GPUNormalize, load_images, set_stage, train_mode


def rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def count_flops(model, x, train_stage=None):
    """FLOPs (multiply + add counted separately) of one forward, or forward + backward if train_stage.
    Runs on a copy, since a training step updates the BatchNorm running stats."""
    model = copy.deepcopy(model)
    with FlopCounterMode(display=False) as fc:
        if train_stage:
            set_stage(model, train_stage)
            train_mode(model, train_stage)
            out_f, out_r = model(x)
            (out_f.sum() + out_r.sum()).backward()
            model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                model(x)
    return fc.get_total_flops()


@torch.no_grad()
def time_model(model, x, reps, cuda, amp=False):
    """Per-call latencies in ms."""
    for _ in range(10):
        with torch.autocast("cuda", enabled=amp):
            model(x)
    times = []
    for _ in range(reps):
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast("cuda", enabled=amp):
            model(x)
        if cuda:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return np.array(times)


def latency_stats(times, bs):
    return {"batch_size": bs, "p50_ms": float(np.percentile(times, 50)), "p90_ms": float(np.percentile(times, 90)),
            "p50_ms_per_sample": float(np.percentile(times, 50)) / bs,
            "throughput_img_s": bs / float(np.percentile(times, 50) / 1000)}


@torch.no_grad()
def evaluate(model, X, y, device, amp=False, bs=64):
    norm = GPUNormalize(device)
    preds = []
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", enabled=amp):
            out_f, _ = model(norm(X[i:i + bs].to(device)))
        preds.append(out_f.float().argmax(1).cpu())
    p = torch.cat(preds).numpy()
    return {"accuracy": float(accuracy_score(y, p)), "f1_macro": float(f1_score(y, p, average="macro"))}


def quantize_int8(model, calib):
    """Static post-training int8 quantization (FX graph mode, oneDNN backend), calibrated on train images."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = "onednn"
    m = copy.deepcopy(model).cpu().eval()
    prepared = prepare_fx(m, get_default_qconfig_mapping("onednn"), example_inputs=(calib[:1],))
    with torch.no_grad():
        for i in range(0, len(calib), 32):
            prepared(calib[i:i + 32])
    return convert_fx(prepared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="s4x40_reg")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--data", default="Processed Data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--calib-images", type=int, default=256)
    args = ap.parse_args()
    run_dir = Path(args.runs_dir) / args.run
    cuda = torch.cuda.is_available()
    gpu = torch.device("cuda") if cuda else torch.device("cpu")
    cpu = torch.device("cpu")
    torch.manual_seed(args.seed)
    res = {"run": args.run, "gpu": torch.cuda.get_device_name() if cuda else None,
           "cpu_threads": torch.get_num_threads()}

    model = FreshnessNet(dropout=0.5)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu"))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    res["model"] = {"params": n_params, "fp32_size_mb": n_params * 4 / 2**20,
                    "checkpoint_mb": (run_dir / "best.pt").stat().st_size / 2**20}

    # FLOPs per image (224x224): inference forward, and one training step in each stage
    x1 = torch.randn(1, 3, 224, 224)
    res["flops"] = {"forward_gflops_per_image": count_flops(model, x1) / 1e9,
                    "train_step_stage1_gflops_per_image": count_flops(model, x1, 1) / 1e9,
                    "train_step_stage2_gflops_per_image": count_flops(model, x1, 2) / 1e9}
    hist = pd.read_csv(run_dir / "history.csv")
    n_train = json.loads((run_dir / "selection_precision.json").read_text())["n_selected"]
    split =pd.read_csv(Path(args.runs_dir) / f"split_seed{args.seed}.csv")
    n_val = int((split["split"] == "val").sum())
    f = res["flops"]
    total = sum(n_train * (f["train_step_stage1_gflops_per_image"] if s == 1 else f["train_step_stage2_gflops_per_image"])
                + n_val * f["forward_gflops_per_image"] for s in hist["stage"])
    res["flops"]["total_training_pflops"] = total / 1e6

    # Latency / throughput / memory
    lat = {}
    if cuda:
        m = model.to(gpu)
        for amp in (False, True):
            tag = "fp16" if amp else "fp32"
            for bs, reps in ((1, 300), (32, 100), (64, 50)):
                torch.cuda.reset_peak_memory_stats()
                t = time_model(m, torch.randn(bs, 3, 224, 224, device=gpu), reps, True, amp)
                lat[f"gpu_{tag}_bs{bs}"] = {**latency_stats(t, bs),
                                            "peak_vram_mb": torch.cuda.max_memory_allocated() / 2**20}
        model.to(cpu)
        torch.cuda.empty_cache()
    rss0 = rss_mb()
    for bs, reps in ((1, 100), (32, 10)):
        t = time_model(model, torch.randn(bs, 3, 224, 224), reps, False)
        lat[f"cpu_fp32_bs{bs}"] = {**latency_stats(t, bs), "process_rss_mb": rss_mb()}
    res["latency"] = lat
    res["process_rss_before_cpu_inference_mb"] = rss0

    # Accuracy of fp32 / fp16 / int8 on the test set
    data_dir = Path(args.data)
    test = split[split["split"] == "test"]
    fi = {s: i for i, s in enumerate(FRESHNESS_ORDER)}
    Xte = load_images(data_dir, test["path"], args.workers)
    yte = test["freshness"].map(fi).to_numpy()
    acc = {}
    if cuda:
        model.to(gpu)
        acc["gpu_fp32"] = evaluate(model, Xte, yte, gpu)
        acc["gpu_fp16"] = evaluate(model, Xte, yte, gpu, amp=True)
        model.to(cpu)
    try:
        train_paths = split[split["split"] == "train"]["path"].sample(args.calib_images, random_state=args.seed)
        norm = GPUNormalize(cpu)
        calib = norm(load_images(data_dir, train_paths, args.workers))
        q = quantize_int8(model, calib)
        tmp = run_dir / "int8.pt"
        torch.save(q.state_dict(), tmp)
        res["model"]["int8_size_mb"] = tmp.stat().st_size / 2**20
        tmp.unlink()
        for bs, reps in ((1, 100), (32, 10)):
            t = time_model(q, torch.randn(bs, 3, 224, 224), reps, False)
            lat[f"cpu_int8_bs{bs}"] = {**latency_stats(t, bs), "process_rss_mb": rss_mb()}
        acc["cpu_fp32"] = evaluate(model, Xte, yte, cpu)
        acc["cpu_int8"] = evaluate(q, Xte, yte, cpu)
    except Exception as e:  # quantization APIs vary between torch versions
        res["int8_error"] = repr(e)
    res["test_accuracy"] = acc

    (run_dir / "efficiency.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
