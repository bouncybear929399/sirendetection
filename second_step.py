#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频推理 - 第二步

功能
----
1. 扫描目标文件夹下所有 `*.wav`（均为第一步产出的 16k/单通道/2s 格式）；
2. 多进程 + 批处理调用 TorchScript 模型推理；
3. 从文件名 `{keyword}_{序号}.wav` 中取出下划线前的 keyword；
4. 将 keyword 与模型预测结果（["BKN","POL","FIR","AMB","ENG"] 的 argmax）比对；
5. 不一致的文件名写入当前目录下的 txt 列表。

用法示例
--------
# 默认 GPU + 4 进程
python step2_infer.py ./dataset_out

# 指定模型 / 进程数 / 输出文件 / 附带概率
python step2_infer.py ./dataset_out \
    --model ./postprocess_android/913/siren_detection_model(0913)_triple_threshold_recommended.pt \
    --device cuda --workers 4 --batch-size 16 \
    --out mismatch_list.txt --include-prob

依赖
----
    pip install numpy torch soundfile scipy
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from math import gcd
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch


CLASS_NAMES = ["BKN", "POL", "FIR", "AMB", "ENG"]
DEFAULT_MODEL = (
    r"D:\siren_lite_gru.pt"
)


# --------------------------------------------------------------------------- #
# Worker 侧
# --------------------------------------------------------------------------- #

_WORKER: dict = {}


def _init_worker(model_path: str, device: str, sample_rate: int, seconds: float) -> None:
    """每个子进程启动时调用一次：加载模型、缓存常量。"""
    try:
        torch.set_num_threads(1)          # 避免 N 进程 × N 线程超订 CPU
    except Exception:
        pass
    print(model_path)
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    _WORKER["model"] = model
    _WORKER["device"] = torch.device(device)
    _WORKER["sample_rate"] = int(sample_rate)
    _WORKER["seconds"] = float(seconds)
    _WORKER["target_len"] = int(round(sample_rate * seconds))


def _load_wav(path: str, sample_rate: int, target_len: int) -> np.ndarray:
    """
    读取 wav 并返回长度恰为 target_len 的 float32 单通道波形。
    第一步产出已是 16k/单通道/2s，所以正常情况下只是直接读取；
    对于目录里混入的不规范文件做 resample / pad / trim 兜底。
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    mono = np.ascontiguousarray(mono, dtype=np.float32)

    if sr != sample_rate:
        from scipy.signal import resample_poly
        g = gcd(int(sr), int(sample_rate))
        mono = resample_poly(mono, sample_rate // g, sr // g).astype(np.float32)

    n = mono.size
    if n < target_len:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[:n] = mono
        mono = padded
    elif n > target_len:
        mono = mono[:target_len]
    return mono


def _infer_batch(paths: List[str]) -> List[Tuple[str, int, List[float]]]:
    """
    在 worker 内处理一批文件。
    返回 [(path, pred_idx, probs_list), ...]；读失败的 pred_idx = -1。
    """
    model = _WORKER["model"]
    device = _WORKER["device"]
    sr = _WORKER["sample_rate"]
    tl = _WORKER["target_len"]

    valid_paths: List[str] = []
    valid_waves: List[np.ndarray] = []
    failed: List[str] = []

    for p in paths:
        try:
            valid_waves.append(_load_wav(p, sr, tl))
            valid_paths.append(p)
        except Exception as e:
            failed.append(p)
            print(f"\n[warn] 读取失败 {p}: {e}", file=sys.stderr)

    results: List[Tuple[str, int, List[float]]] = []

    if valid_waves:
        batch = np.stack(valid_waves, axis=0)                       # [B, N]
        tensor = torch.from_numpy(batch).unsqueeze(1).to(device)    # [B, 1, N]
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        for p, pred, pr in zip(valid_paths, preds.tolist(), probs):
            results.append((p, int(pred), pr.tolist()))

    for p in failed:
        results.append((p, -1, [0.0] * len(CLASS_NAMES)))

    return results


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="多进程推理并与文件名关键词比对",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("folder", help="第一步输出目录（包含 {keyword}_{idx}.wav）")
    p.add_argument("--model", default=DEFAULT_MODEL, help="TorchScript .pt 模型路径")
    p.add_argument("--sample-rate", type=int, default=16000, help="模型输入采样率")
    p.add_argument("--seconds", type=float, default=2.0, help="每段时长（秒）")
    p.add_argument("--device", default="cpu", help="cpu / cuda / cuda:0 ...")
    p.add_argument("--workers", type=int, default=4, help="进程数")
    p.add_argument("--batch-size", type=int, default=16, help="每个 worker 每批处理的文件数")
    p.add_argument("--out", default="mismatch_list.txt",
                   help="输出 txt 文件名（相对路径 -> 写到当前目录）")
    p.add_argument("--include-prob", action="store_true",
                   help="txt 中附带预测类别和五类概率")
    p.add_argument("--suffix", default=".wav", help="扫描的文件后缀")
    p.add_argument("--recursive", action="store_true", help="递归扫描子目录")
    return p.parse_args()


def _scan(folder: Path, suffix: str, recursive: bool) -> List[Path]:
    it = folder.rglob(f"*{suffix}") if recursive else folder.glob(f"*{suffix}")
    return sorted(p for p in it if p.is_file() and not p.name.startswith("."))


def main() -> int:
    args = parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"[error] 目录不存在: {folder}", file=sys.stderr)
        return 2

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        print(f"[error] 模型文件不存在: {model_path}", file=sys.stderr)
        return 2

    # 设备兜底
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，自动回退到 CPU。")
        args.device = "cpu"

    files = _scan(folder, args.suffix, args.recursive)
    if not files:
        print(f"[warn] 目录下没有 {args.suffix} 文件: {folder}")
        return 0

    workers = max(1, args.workers)
    bs = max(1, args.batch_size)

    print(f"[info] 目录     : {folder}")
    print(f"[info] 模型     : {model_path}")
    print(f"[info] 设备     : {args.device}")
    print(f"[info] 进程数   : {workers}")
    print(f"[info] batch    : {bs}")
    print(f"[info] 文件总数 : {len(files)}")
    print(f"[info] 输入规格 : {args.sample_rate} Hz / {args.seconds}s / mono")
    print("-" * 64)

    batches = [ [str(p) for p in files[i:i + bs]] for i in range(0, len(files), bs) ]

    ctx = mp.get_context("spawn")
    all_results: List[Tuple[str, int, List[float]]] = []
    done = 0
    total = len(files)

    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(model_path, args.device, args.sample_rate, args.seconds),
    ) as pool:
        for result in pool.imap_unordered(_infer_batch, batches):
            all_results.extend(result)
            done += len(result)
            print(f"\r  进度 {done}/{total}", end="", flush=True)
    print()

    # ---------- 比对 ---------- #
    mismatches: List[Tuple[str, str, str, List[float]]] = []
    parse_fail: List[str] = []
    read_fail: List[str] = []

    for path_str, pred_idx, probs in all_results:
        fname = os.path.basename(path_str)
        stem = os.path.splitext(fname)[0]

        if "_" not in stem:
            parse_fail.append(fname)
            continue
        keyword = stem.rsplit("_", 1)[0]

        if pred_idx < 0:
            read_fail.append(fname)
            mismatches.append((fname, keyword, "READ_ERROR", probs))
            continue

        pred_name = CLASS_NAMES[pred_idx]
        if keyword != pred_name:
            mismatches.append((fname, keyword, pred_name, probs))

    mismatches.sort(key=lambda x: x[0])

    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for fname, kw, pred_name, probs in mismatches:
            if args.include_prob and probs:
                score = " ".join(f"{n}={probs[i]:.4f}" for i, n in enumerate(CLASS_NAMES))
                f.write(f"{fname}\tkeyword={kw}\tpredict={pred_name}\t{score}\n")
            else:
                f.write(f"{fname}\n")

    # ---------- 汇总 ---------- #
    print("-" * 64)
    print(f"总文件数         : {total}")
    print(f"推理成功         : {sum(1 for r in all_results if r[1] >= 0)}")
    print(f"读取失败         : {len(read_fail)}")
    print(f"文件名无法解析   : {len(parse_fail)}")
    print(f"关键词不一致     : {len(mismatches)}")
    print(f"结果写入         : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())