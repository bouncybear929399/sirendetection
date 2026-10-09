#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频推理 - 第二步（长音频滑窗 + 窗口级/文件级双层评测）

核心行为：
    - 每个音频按 2s 窗 / 0.5s hop 滑窗，每个窗口独立推理
    - 每个窗口输出 (预测类别, 置信度)
    - 文件级聚合（mean/vote/max/first）用于文件级评测
    - 输出三份 CSV：
        window_predictions.csv   每个窗口一行（含平滑前后）
        window_mismatches.csv    所有错误的窗口
        switches.csv             类别切换点
    - 终端同时打印：文件级准确率 + 窗口级准确率（平滑前/后）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import multiprocessing as mp
import os
import re
import sys
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch


CLASS_NAMES = ["BKN", "POL", "FIR", "AMB", "ENG"]
BKN_INDEX = CLASS_NAMES.index("BKN")

FOLDER_ALIASES = {
    "BKN": "BKN", "BACKGROUND": "BKN", "NOISE": "BKN", "背景": "BKN", "噪声": "BKN",
    "POL": "POL", "POLICE": "POL", "警车": "POL",
    "FIR": "FIR", "FIRE": "FIR", "消防": "FIR", "消防车": "FIR",
    "AMB": "AMB", "AMBULANCE": "AMB", "救护": "AMB", "救护车": "AMB", "急救": "AMB",
    "ENG": "ENG", "ENGINEERING": "ENG", "工程": "ENG", "工程车": "ENG",
}

DEFAULT_MODEL = r"D:\siren_lite_gru.pt"


# --------------------------------------------------------------------------- #
# Worker 侧
# --------------------------------------------------------------------------- #

_WORKER: dict = {}


def _init_worker(model_path: str, device: str, sample_rate: int,
                 seconds: float, hop_seconds: float,
                 gpu_batch: int) -> None:
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    _WORKER["model"] = model
    _WORKER["device"] = torch.device(device)
    _WORKER["sample_rate"] = int(sample_rate)
    _WORKER["seconds"] = float(seconds)
    _WORKER["hop_seconds"] = float(hop_seconds)
    _WORKER["target_len"] = int(round(sample_rate * seconds))
    _WORKER["hop_len"] = int(round(sample_rate * hop_seconds))
    _WORKER["gpu_batch"] = int(gpu_batch)


def _load_wav_windows(path: str, sample_rate: int, target_len: int,
                      hop_len: int) -> np.ndarray:
    """
    读取 wav，返回 [num_windows, target_len] 的 float32。
      - 长度 <= target_len: 补零到 target_len, 返回 1 个窗口
      - 长度 >  target_len: 按 hop_len 滑窗
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    mono = np.ascontiguousarray(mono, dtype=np.float32)

    if sr != sample_rate:
        from scipy.signal import resample_poly
        g = gcd(int(sr), int(sample_rate))
        mono = resample_poly(mono, sample_rate // g, sr // g).astype(np.float32)

    n = mono.size
    if n <= target_len:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[:n] = mono
        return padded[None, :]

    starts = list(range(0, n - target_len + 1, hop_len))
    last_end = starts[-1] + target_len
    if n - last_end >= hop_len // 2:
        starts.append(n - target_len)

    windows = np.stack([mono[s:s + target_len] for s in starts], axis=0)
    return windows.astype(np.float32)


def _infer_batch(paths: List[str]) -> List[Tuple[str, np.ndarray]]:
    """
    返回 [(path, probs_matrix), ...], probs_matrix 形状 [num_windows, 5]
    内部按 gpu_batch 分块送 GPU，避免长音频一次性吃满显存。
    """
    model = _WORKER["model"]
    device = _WORKER["device"]
    sr = _WORKER["sample_rate"]
    tl = _WORKER["target_len"]
    hl = _WORKER["hop_len"]
    gb = _WORKER["gpu_batch"]

    results: List[Tuple[str, np.ndarray]] = []

    for p in paths:
        try:
            wins = _load_wav_windows(p, sr, tl, hl)
        except Exception as e:
            print(f"\n[warn] 读取失败 {p}: {e}", file=sys.stderr)
            results.append((p, np.zeros((1, len(CLASS_NAMES)), dtype=np.float32)))
            continue

        probs_chunks = []
        with torch.no_grad():
            for i in range(0, wins.shape[0], gb):
                chunk = wins[i:i + gb]
                tensor = torch.from_numpy(chunk).unsqueeze(1).to(device)
                logits = model(tensor)
                probs_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.concatenate(probs_chunks, axis=0)
        results.append((p, probs))

    return results


# --------------------------------------------------------------------------- #
# 真实类别推断
# --------------------------------------------------------------------------- #

def _get_ground_truth(path_str: str) -> Optional[str]:
    # 中文注释：优先从父目录解析类别。
    p = Path(path_str)
    parent = p.parent.name.strip()

    gt = FOLDER_ALIASES.get(parent)
    if gt is None:
        gt = FOLDER_ALIASES.get(parent.upper())

    if gt is not None:
        return gt

    # 中文注释：文件名中只要包含类别关键词，就认为该文件属于对应类别。
    stem_upper = p.stem.upper()

    for alias, class_name in FOLDER_ALIASES.items():
        if alias.upper() in stem_upper:
            return class_name

    return None

# --------------------------------------------------------------------------- #
# 时序平滑
# --------------------------------------------------------------------------- #

def smooth_sequence(
    seq: List[Tuple[int, float]],
    high_thresh: float,
    window_size: int,
    vote_needed: int,
    min_conf: float,
) -> List[Optional[int]]:
    out: List[Optional[int]] = []
    # 中文注释：初始尚未确认类别时输出 BKN，之后低置信度保持已确认状态。
    current: Optional[int] = BKN_INDEX
    window: List[Optional[int]] = []

    for pred_idx, conf in seq:
        if conf < min_conf:
            out.append(current)
            window.append(None)
            if len(window) > window_size:
                window.pop(0)
            continue

        if conf >= high_thresh and pred_idx != current:
            current = pred_idx
            window = [pred_idx]
            out.append(current)
            continue

        window.append(pred_idx)
        if len(window) > window_size:
            window.pop(0)
        if len(window) == window_size:
            valid = [x for x in window if x is not None]
            if valid:
                top, cnt = Counter(valid).most_common(1)[0]
                if top != current and cnt >= vote_needed:
                    current = top
        out.append(current)

    return out


# --------------------------------------------------------------------------- #
# 打印 / 写文件辅助
# --------------------------------------------------------------------------- #


def plot_audio_switches(path_str, base_preds, smooth_preds, sample_rate,
                        seconds, hop_seconds, output_dir):
    """绘制单条音频的类别时间线，并保存平滑前后跳变的精确时间清单。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 中文注释：复用切窗规则恢复实际起点，末尾追加窗口不能按固定步长估算。
    info = sf.info(path_str)
    num_samples = int(np.ceil(info.frames * sample_rate / info.samplerate))
    target_len = int(round(sample_rate * seconds))
    hop_len = int(round(sample_rate * hop_seconds))
    starts = list(range(0, num_samples - target_len + 1, hop_len)) if num_samples > target_len else [0]
    if num_samples > target_len and num_samples - starts[-1] - target_len >= hop_len // 2:
        starts.append(num_samples - target_len)
    if len(starts) != len(base_preds):
        raise ValueError("窗口数量与时间轴不一致")
    times = np.asarray(starts) / sample_rate
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 中文注释：路径摘要避免不同目录中的同名音频覆盖图片。
    name = _sanitize_name(Path(path_str).stem) + "_" + hashlib.sha256(path_str.encode("utf-8")).hexdigest()[:10]
    rows = []
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    try:
        for ax, predictions, stage in zip(axes, (base_preds, smooth_preds), ("Raw", "Smoothed")):
            # 中文注释：兼容旧结果，将未确认类别映射为 BKN 后绘制和统计跳变。
            predictions = np.asarray([BKN_INDEX if p is None or p < 0 else int(p) for p in predictions])
            changes = np.flatnonzero(np.diff(predictions) != 0) + 1
            ax.step(times, predictions, where="post", linewidth=1, color="#168579")
            ax.scatter(times[changes], predictions[changes], s=18, color="#d94747", zorder=3)
            for idx in changes:
                ax.axvline(times[idx], color="#d94747", alpha=0.25, linewidth=0.7)
                before, after = int(predictions[idx - 1]), int(predictions[idx])
                rows.append([stage, int(idx), f"{times[idx]:.6f}",
                             f"{times[idx] + seconds:.6f}",
                             CLASS_NAMES[before],
                             CLASS_NAMES[after]])
            ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
            ax.set_ylim(-0.4, len(CLASS_NAMES) - 0.6)
            ax.set_title(f"{stage}: {len(changes)} transitions")
            ax.grid(axis="x", alpha=0.2)
        axes[-1].set_xlabel("Window start time (seconds)")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}.png", dpi=160)
    finally:
        plt.close(fig)
    # 中文注释：窗口结束时间才是在线系统最早可获得预测的时刻，不含计算耗时。
    with open(output_dir / f"{name}_switches.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["stage", "window_idx", "window_start_sec", "available_at_sec", "from_class", "to_class"])
        writer.writerows(rows)


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\-.]", "_", name).strip("._")
    return cleaned or "model"


def _print_confusion(confusion: Dict[str, Dict[str, int]]) -> None:
    cols = CLASS_NAMES + ["ERR"]
    header = "真实\\预测".ljust(10) + "".join(c.rjust(8) for c in cols)
    print(header)
    print("-" * len(header))
    for true_cls in CLASS_NAMES:
        row = true_cls.ljust(10)
        for col in cols:
            row += str(confusion[true_cls].get(col, 0)).rjust(8)
        print(row)


def _print_model_summary(stats: Dict) -> None:
    print("=" * 64)
    print(f"模型            : {stats['model']}")
    print(f"参与评估文件数   : {stats['total_eval']}")
    print(f"读取失败         : {stats['read_fail']}")
    print(f"预测正确         : {stats['total_correct']}")
    pred_wrong = stats["total_eval"] - stats["total_correct"] - stats["read_fail"]
    print(f"预测错误         : {pred_wrong}")
    print(f"文件级准确率     : {stats['overall_acc']:.2f}%")
    print(f"总窗口数         : {stats['total_windows']}")
    print(f"平均窗口/文件    : {stats['avg_windows']:.2f}")
    print("=" * 64)

    print("各类别准确率（文件级）:")
    for cls in CLASS_NAMES:
        tot = stats["per_class_total"][cls]
        cor = stats["per_class_correct"][cls]
        if tot == 0:
            print(f"  {cls:4s}  : 无样本")
        else:
            print(f"  {cls:4s}  : {cor} / {tot}  = {100.0 * cor / tot:.2f}%")

    print("\n混淆矩阵 (行=真实, 列=预测, ERR=读取失败):")
    _print_confusion(stats["confusion"])

    # ---------- 窗口级准确率 ---------- #
    if stats.get("win_total", 0) > 0:
        print("\n窗口级准确率（每个 2s 窗口独立评估）:")
        print(f"  窗口总数          : {stats['win_total']}")
        print(f"  平滑前 正确/准确率: {stats['win_correct_base']} / "
              f"{stats['win_total']}  = {stats['win_acc_base']:.2f}%")
        if stats.get("time_smooth"):
            print(f"  平滑后 有效窗口数 : {stats['win_smooth_valid']}  "
                  f"(未决 {stats['win_total'] - stats['win_smooth_valid']})")
            print(f"  平滑后 正确/准确率: {stats['win_correct_smooth']} / "
                  f"{stats['win_smooth_valid']}  = {stats['win_acc_smooth']:.2f}%")
            delta = stats['win_acc_smooth'] - stats['win_acc_base']
            sign = "+" if delta >= 0 else ""
            print(f"  窗口级准确率变化  : {sign}{delta:.2f}%")

    # ---------- 时序平滑跳变统计 ---------- #
    if stats.get("time_smooth"):
        print("\n时序平滑统计（窗口级跳变）:")
        print(f"  原始跳变次数      : {stats['base_switch']}")
        print(f"  平滑后跳变次数    : {stats['smooth_switch']}")
        if stats["base_switch"] > 0:
            red = 100.0 * (stats["base_switch"] - stats["smooth_switch"]) / stats["base_switch"]
            print(f"  跳变下降          : {red:.1f}%")
        print(f"  平滑参数          : high={stats['high_thresh']} "
              f"W={stats['window_size']} V={stats['vote_needed']} "
              f"min_conf={stats['min_conf']}")

    print(f"\n错误清单: {stats['out_txt']}")
    if stats.get("window_csv"):
        print(f"窗口级明细      : {stats['window_csv']}")
    if stats.get("window_mismatch_csv"):
        print(f"窗口级错误      : {stats['window_mismatch_csv']}")
    if stats.get("switch_csv"):
        print(f"切换点清单      : {stats['switch_csv']}")


def _print_comparison(all_stats: List[Dict]) -> None:
    print("\n" + "=" * 120)
    print("模型对比汇总")
    print("=" * 120)
    header = "模型".ljust(34) + "文件级".rjust(10)
    for cls in CLASS_NAMES:
        header += cls.rjust(9)
    header += "窗口base".rjust(11) + "窗口smooth".rjust(12)
    print(header)
    print("-" * len(header))
    for st in all_stats:
        name = st["model"]
        if len(name) > 32:
            name = name[:29] + "..."
        row = name.ljust(34) + f"{st['overall_acc']:.2f}%".rjust(10)
        for cls in CLASS_NAMES:
            tot = st["per_class_total"][cls]
            cor = st["per_class_correct"][cls]
            row += ("    --".rjust(9) if tot == 0
                    else f"{100.0 * cor / tot:.2f}%".rjust(9))
        row += f"{st.get('win_acc_base', 0.0):.2f}%".rjust(11)
        if st.get("time_smooth"):
            row += f"{st.get('win_acc_smooth', 0.0):.2f}%".rjust(12)
        else:
            row += "    --".rjust(12)
        print(row)
    print("=" * 120)


def _write_summary_txt(all_stats: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        header = ["model", "model_path",
                  "file_acc", "total_eval", "total_correct", "read_fail",
                  "total_windows", "win_acc_base"]
        header += [f"{c}_acc" for c in CLASS_NAMES]
        if all_stats and all_stats[0].get("time_smooth"):
            header += ["win_acc_smooth", "base_switch", "smooth_switch"]
        f.write("\t".join(header) + "\n")

        for st in all_stats:
            cols = [
                st["model"], st["model_path"],
                f"{st['overall_acc']:.4f}",
                str(st["total_eval"]),
                str(st["total_correct"]),
                str(st["read_fail"]),
                str(st["total_windows"]),
                f"{st.get('win_acc_base', 0.0):.4f}",
            ]
            for cls in CLASS_NAMES:
                tot = st["per_class_total"][cls]
                cor = st["per_class_correct"][cls]
                acc = 100.0 * cor / tot if tot > 0 else 0.0
                cols.append(f"{acc:.4f}")
            if st.get("time_smooth"):
                cols += [
                    f"{st.get('win_acc_smooth', 0.0):.4f}",
                    str(st.get("base_switch", 0)),
                    str(st.get("smooth_switch", 0)),
                ]
            f.write("\t".join(cols) + "\n")


# --------------------------------------------------------------------------- #
# 单个模型的完整评测
# --------------------------------------------------------------------------- #

def _run_one_model(
    model_path: Path,
    file_records: List[Tuple[str, Optional[str]]],
    device: str,
    sample_rate: int,
    seconds: float,
    hop_seconds: float,
    workers: int,
    file_batch: int,
    gpu_batch: int,
    aggregate: str,
    include_prob: bool,
    out_txt: Path,
    time_smooth: bool,
    high_thresh: float,
    window_size: int,
    vote_needed: int,
    min_conf: float,
    window_csv: Optional[Path],
    probs_csv: Optional[Path],
    switch_csv: Optional[Path],
    plot_dir: Optional[Path] = None,
) -> Dict:
    paths = [rec[0] for rec in file_records]
    tasks = [paths[i:i + file_batch] for i in range(0, len(paths), file_batch)]

    total = len(paths)
    print(f"\n[model] {model_path.name}  ({total} files)")

    ctx = mp.get_context("spawn")
    all_results: List[Tuple[str, np.ndarray]] = []
    done = 0

    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(str(model_path), device, sample_rate, seconds,
                  hop_seconds, gpu_batch),
    ) as pool:
        for result in pool.imap_unordered(_infer_batch, tasks):
            all_results.extend(result)
            done += len(result)
            print(f"\r  进度 {done}/{total} files", end="", flush=True)
    print()

    # ---------- 统计容器 ---------- #
    gt_map = {rec[0]: rec[1] for rec in file_records}

    # 文件级
    per_class_total = {c: 0 for c in CLASS_NAMES}
    per_class_correct = {c: 0 for c in CLASS_NAMES}
    confusion = {c: {p: 0 for p in CLASS_NAMES + ["ERR"]} for c in CLASS_NAMES}
    mismatches: List[Tuple[str, str, str, List[float]]] = []
    read_fail_count = 0
    total_windows = 0

    # 窗口级
    win_total = 0
    win_correct_base = 0
    win_correct_smooth = 0
    win_smooth_valid = 0

    # 跳变
    base_switch = 0
    smooth_switch = 0

    # 平滑后文件级
    smooth_file_total = 0
    smooth_file_correct = 0

    # 窗口级输出
    window_rows: List[List] = []
    switch_rows: List[List] = []
    win_mismatch_rows: List[List] = []

    all_results.sort(key=lambda x: x[0])

    for path_str, probs_mat in all_results:
        fname = os.path.basename(path_str)
        gt = gt_map.get(path_str)

        if probs_mat.shape[0] == 0:
            read_fail_count += 1
            if gt is not None:
                per_class_total[gt] += 1
                confusion[gt]["ERR"] += 1
                mismatches.append((fname, gt, "READ_ERROR", []))
            continue

        W = probs_mat.shape[0]
        total_windows += W

        win_preds = probs_mat.argmax(axis=1)
        win_confs = probs_mat.max(axis=1).astype(np.float64)

        # ---------- 平滑（如果有） ---------- #
        if time_smooth:
            seq = [(int(win_preds[wi]), float(win_confs[wi])) for wi in range(W)]
            smoothed = smooth_sequence(seq, high_thresh, window_size,
                                       vote_needed, min_conf)
            final_win_preds = np.array(
                [(-1 if s is None else int(s)) for s in smoothed]
            )
        else:
            final_win_preds = win_preds.copy()

        # 中文注释：每条音频独立绘图，不改变原来的预测和平滑结果。
        if plot_dir is not None:
            plot_audio_switches(path_str, win_preds, final_win_preds,
                                sample_rate, seconds, hop_seconds, plot_dir)

        # ---------- 窗口级统计 ---------- #
        if gt is not None:
            for wi in range(W):
                win_total += 1
                base_pred_name = CLASS_NAMES[int(win_preds[wi])]
                if base_pred_name == gt:
                    win_correct_base += 1

                if time_smooth and final_win_preds[wi] >= 0:
                    smooth_pred_name = CLASS_NAMES[int(final_win_preds[wi])]
                    win_smooth_valid += 1
                    if smooth_pred_name == gt:
                        win_correct_smooth += 1

                # 记录错误窗口
                smooth_pred_name = (
                    CLASS_NAMES[int(final_win_preds[wi])]
                    if final_win_preds[wi] >= 0 else "ABSTAIN"
                )
                is_wrong_base = base_pred_name != gt
                is_wrong_smooth = (time_smooth
                                   and smooth_pred_name not in (gt, "ABSTAIN"))
                if is_wrong_base or is_wrong_smooth:
                    win_mismatch_rows.append([
                        fname, wi,
                        f"{wi * hop_seconds:.3f}",
                        gt, base_pred_name,
                        f"{win_confs[wi]:.4f}",
                        smooth_pred_name,
                    ])

        # ---------- 窗口级 CSV ---------- #
        if window_csv is not None:
            for wi in range(W):
                start_sec = wi * hop_seconds
                end_sec = start_sec + seconds
                smooth_name = (
                    CLASS_NAMES[int(final_win_preds[wi])]
                    if final_win_preds[wi] >= 0 else "ABSTAIN"
                )
                window_rows.append([
                    fname, wi,
                    f"{start_sec:.3f}", f"{end_sec:.3f}",
                    CLASS_NAMES[int(win_preds[wi])],
                    f"{win_confs[wi]:.4f}",
                    smooth_name,
                    f"{win_confs[wi]:.4f}",      # conf_smooth 沿用原窗口 conf
                    gt or "",
                    *[f"{v:.6f}" for v in probs_mat[wi]],
                ])

        # ---------- 切换点 CSV ---------- #
        if switch_csv is not None:
            for wi in range(1, W):
                if win_preds[wi] != win_preds[wi - 1]:
                    switch_rows.append([
                        fname, wi,
                        f"{wi * hop_seconds:.3f}",
                        CLASS_NAMES[int(win_preds[wi - 1])],
                        CLASS_NAMES[int(win_preds[wi])],
                        f"{win_confs[wi - 1]:.4f}",
                        f"{win_confs[wi]:.4f}",
                    ])

        # ---------- 跳变统计 ---------- #
        if time_smooth:
            for wi in range(1, W):
                if win_preds[wi] != win_preds[wi - 1]:
                    base_switch += 1
            for wi in range(1, W):
                if final_win_preds[wi] != final_win_preds[wi - 1]:
                    smooth_switch += 1

        # ---------- 文件级聚合 ---------- #
        if aggregate == "mean":
            file_prob = probs_mat.mean(axis=0)
            file_pred = int(file_prob.argmax())
        elif aggregate == "vote":
            # 平滑后如果有有效窗口，用平滑后的多数票；否则用原始窗口
            if time_smooth:
                valid = final_win_preds[final_win_preds >= 0]
            else:
                valid = win_preds
            if valid.size == 0:
                file_pred = int(win_preds[0])
            else:
                vals, cnts = np.unique(valid, return_counts=True)
                file_pred = int(vals[cnts.argmax()])
            file_prob = probs_mat.mean(axis=0)
        elif aggregate == "max":
            best_wi = int(win_confs.argmax())
            file_pred = int(win_preds[best_wi])
            file_prob = probs_mat[best_wi]
        elif aggregate == "first":
            file_pred = int(win_preds[0])
            file_prob = probs_mat[0]
        else:
            raise ValueError(f"未知聚合方式: {aggregate}")

        if gt is None:
            continue

        # ---------- 文件级统计（平滑后聚合） ---------- #
        per_class_total[gt] += 1
        pred_name = CLASS_NAMES[file_pred]
        confusion[gt][pred_name] += 1
        if gt == pred_name:
            per_class_correct[gt] += 1
        else:
            mismatches.append((fname, gt, pred_name, file_prob.tolist()))

        # 平滑后文件级
        if time_smooth:
            if aggregate == "vote":
                valid = final_win_preds[final_win_preds >= 0]
                if valid.size > 0:
                    vals, cnts = np.unique(valid, return_counts=True)
                    s_pred = int(vals[cnts.argmax()])
                    smooth_file_total += 1
                    if gt == CLASS_NAMES[s_pred]:
                        smooth_file_correct += 1
            elif aggregate == "mean":
                # mean 对平滑没意义，用 vote 代替
                valid = final_win_preds[final_win_preds >= 0]
                if valid.size > 0:
                    vals, cnts = np.unique(valid, return_counts=True)
                    s_pred = int(vals[cnts.argmax()])
                    smooth_file_total += 1
                    if gt == CLASS_NAMES[s_pred]:
                        smooth_file_correct += 1
            elif aggregate == "max":
                best_wi = int(win_confs.argmax())
                s_pred = int(final_win_preds[best_wi])
                if s_pred >= 0:
                    smooth_file_total += 1
                    if gt == CLASS_NAMES[s_pred]:
                        smooth_file_correct += 1
            else:  # first
                s_pred = int(final_win_preds[0])
                if s_pred >= 0:
                    smooth_file_total += 1
                    if gt == CLASS_NAMES[s_pred]:
                        smooth_file_correct += 1

    mismatches.sort(key=lambda x: x[0])

    # ---------- 写文件级错误清单 ---------- #
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for fname, gt, pred_name, probs in mismatches:
            if include_prob and probs:
                score = " ".join(f"{n}={probs[i]:.4f}"
                                 for i, n in enumerate(CLASS_NAMES))
                f.write(f"{fname}\tlabel={gt}\tpredict={pred_name}\t{score}\n")
            else:
                f.write(f"{fname}\n")

    # ---------- 写窗口级 CSV ---------- #
    if window_csv is not None:
        window_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(window_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "filename", "window_idx", "start_sec", "end_sec",
                "pred_base", "conf_base",
                "pred_smooth", "conf_smooth",
                "gt",
            ] + [f"p_{n}" for n in CLASS_NAMES])
            w.writerows(window_rows)

    # 单独保存每个窗口的类别概率，便于后续绘图、阈值分析和误差复盘。
    if probs_csv is not None:
        probs_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(probs_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "filename", "window_idx", "start_sec", "end_sec", "gt",
                "pred", "confidence",
            ] + [f"p_{name}" for name in CLASS_NAMES])
            for path_str, probs_mat in all_results:
                file_name = os.path.basename(path_str)
                gt_name = gt_map.get(path_str) or ""
                for window_idx, probabilities in enumerate(probs_mat):
                    prediction = int(probabilities.argmax())
                    confidence = float(probabilities[prediction])
                    start_sec = window_idx * hop_seconds
                    end_sec = start_sec + seconds
                    w.writerow([
                        file_name, window_idx,
                        f"{start_sec:.3f}", f"{end_sec:.3f}", gt_name,
                        CLASS_NAMES[prediction], f"{confidence:.6f}",
                        *[f"{value:.8f}" for value in probabilities],
                    ])

    # ---------- 写窗口级错误清单 ---------- #
    window_mismatch_path = None
    if window_csv is not None:
        window_mismatch_path = window_csv.parent / "window_mismatches.csv"
        with open(window_mismatch_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "window_idx", "start_sec", "gt",
                        "pred_base", "conf_base", "pred_smooth"])
            w.writerows(win_mismatch_rows)

    # ---------- 写切换点 CSV ---------- #
    if switch_csv is not None:
        switch_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(switch_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "window_idx", "at_sec",
                        "from_class", "to_class", "from_conf", "to_conf"])
            w.writerows(switch_rows)

    total_eval = sum(per_class_total.values())
    total_correct = sum(per_class_correct.values())
    overall_acc = 100.0 * total_correct / total_eval if total_eval > 0 else 0.0

    win_acc_base = (100.0 * win_correct_base / win_total
                    if win_total > 0 else 0.0)
    win_acc_smooth = (100.0 * win_correct_smooth / win_smooth_valid
                      if win_smooth_valid > 0 else 0.0)
    smooth_file_acc = (100.0 * smooth_file_correct / smooth_file_total
                       if smooth_file_total > 0 else 0.0)

    return {
        "model": model_path.name,
        "model_path": str(model_path),
        "per_class_total": per_class_total,
        "per_class_correct": per_class_correct,
        "confusion": confusion,
        "total_eval": total_eval,
        "total_correct": total_correct,
        "read_fail": read_fail_count,
        "overall_acc": overall_acc,
        "mismatch_count": len(mismatches),
        "total_windows": total_windows,
        "avg_windows": (total_windows / total_eval) if total_eval > 0 else 0.0,
        "out_txt": str(out_txt),
        "window_csv": str(window_csv) if window_csv else None,
        "probs_csv": str(probs_csv) if probs_csv else None,
        "window_mismatch_csv": str(window_mismatch_path) if window_mismatch_path else None,
        "switch_csv": str(switch_csv) if switch_csv else None,
        "time_smooth": time_smooth,
        "base_switch": base_switch,
        "smooth_switch": smooth_switch,
        "high_thresh": high_thresh,
        "window_size": window_size,
        "vote_needed": vote_needed,
        "min_conf": min_conf,
        # 窗口级
        "win_total": win_total,
        "win_correct_base": win_correct_base,
        "win_acc_base": win_acc_base,
        "win_smooth_valid": win_smooth_valid,
        "win_correct_smooth": win_correct_smooth,
        "win_acc_smooth": win_acc_smooth,
        # 平滑后文件级
        "smooth_file_total": smooth_file_total,
        "smooth_file_correct": smooth_file_correct,
        "smooth_file_acc": smooth_file_acc,
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="多进程推理 + 长音频滑窗 + 窗口级/文件级双层评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("folder", help="数据集根目录（下面有 BKN/POL/... 子文件夹）")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--model", default=None, help="单个 TorchScript .pt 模型")
    src.add_argument("--models-dir", default=None,
                     help="模型文件夹，遍历其中所有 *.pt 依次评测")

    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--seconds", type=float, default=2.0,
                   help="每个推理窗口时长（秒）")
    p.add_argument("--hop-seconds", type=float, default=0.5,
                   help="窗口滑动步长（秒）")
    p.add_argument("--device", default="cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16,
                   help="每批进 GPU 的窗口数")
    p.add_argument("--file-batch", type=int, default=8,
                   help="每个进程每次从队列取多少文件")

    p.add_argument("--aggregate", choices=["mean", "vote", "max", "first"],
                   default="vote",
                   help="长音频多窗口 -> 文件级预测的聚合方式")
    p.add_argument("--out", default="mismatch_list.txt",
                   help="单模型模式下错误清单的输出路径")
    p.add_argument("--out-dir", default="model_eval_results",
                   help="多模型模式下的结果输出目录")
    p.add_argument("--include-prob", action="store_true")
    p.add_argument("--suffix", default=".wav")
    p.add_argument("--no-recursive", action="store_true")

    # 窗口级 / 切换点
    p.add_argument("--save-windows", action="store_true",
                   help="输出窗口级明细 CSV")
    p.add_argument("--save-probs", default=None, metavar="PATH",
                   help="保存每个窗口五类概率的 CSV 文件")
    p.add_argument("--save-switches", action="store_true",
                   help="输出类别切换点 CSV")
    p.add_argument("--plot-switches", action="store_true",
                   help="为每条音频保存平滑前后跳变时间图和时间清单")

    # 时序平滑
    p.add_argument("--time-smooth", action="store_true",
                   help="在窗口级序列上做时序平滑")
    p.add_argument("--high-thresh", type=float, default=0.90)
    p.add_argument("--window-size", type=int, default=3)
    p.add_argument("--vote-needed", type=int, default=2)
    p.add_argument("--min-conf", type=float, default=0.5)
    return p.parse_args()


def _scan(folder: Path, suffix: str, recursive: bool) -> List[Path]:
    it = folder.rglob(f"*{suffix}") if recursive else folder.glob(f"*{suffix}")
    results = []
    for p in it:
        if not p.is_file():
            continue
        if p.name.startswith("."):
            continue
        if any(part.startswith(".") or part == "__pycache__" for part in p.parts):
            continue
        results.append(p)
    return sorted(results)


def main() -> int:
    args = parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"[error] 数据集目录不存在: {folder}", file=sys.stderr)
        return 2

    if args.models_dir:
        models_dir = Path(args.models_dir).expanduser().resolve()
        if not models_dir.is_dir():
            print(f"[error] 模型目录不存在: {models_dir}", file=sys.stderr)
            return 2
        model_paths = sorted(
            p for p in models_dir.glob("*.pt")
            if p.is_file() and not p.name.startswith(".")
        )
        if not model_paths:
            print(f"[error] 模型目录下没有 *.pt 文件: {models_dir}", file=sys.stderr)
            return 2
        multi_mode = True
    else:
        single = Path(args.model or DEFAULT_MODEL).expanduser()
        single = single if single.is_absolute() else (Path.cwd() / single)
        if not single.is_file():
            print(f"[error] 模型文件不存在: {single}", file=sys.stderr)
            return 2
        model_paths = [single]
        multi_mode = False

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，自动回退到 CPU。")
        args.device = "cpu"

    recursive = not args.no_recursive
    files = _scan(folder, args.suffix, recursive)
    if not files:
        print(f"[warn] 目录下没有 {args.suffix} 文件: {folder}")
        return 0

    file_records: List[Tuple[str, Optional[str]]] = [
        (str(p), _get_ground_truth(str(p))) for p in files
    ]
    parse_fail = sum(1 for _, gt in file_records if gt is None)

    workers = max(1, args.workers)
    gpu_batch = max(1, args.batch_size)
    file_batch = max(1, args.file_batch)

    print("=" * 64)
    print(f"[info] 数据集       : {folder}")
    print(f"[info] 音频总数     : {len(files)}   无法解析类别: {parse_fail}")
    print(f"[info] 设备         : {args.device}")
    print(f"[info] 进程/GPU batch: {workers} / {gpu_batch}")
    print(f"[info] file batch   : {file_batch}")
    print(f"[info] 输入规格     : {args.sample_rate} Hz / {args.seconds}s 窗 "
          f"/ {args.hop_seconds}s hop")
    print(f"[info] 聚合方式     : {args.aggregate}")
    print(f"[info] 模型数量     : {len(model_paths)}")
    if args.time_smooth:
        print(f"[info] 时序平滑     : 开启  (high={args.high_thresh} "
              f"W={args.window_size} V={args.vote_needed} "
              f"min_conf={args.min_conf})")
    print("=" * 64)

    if multi_mode:
        out_dir = Path(args.out_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    all_stats: List[Dict] = []
    for idx, mp_path in enumerate(model_paths, 1):
        print(f"\n({idx}/{len(model_paths)}) {mp_path}")

        if multi_mode:
            sub_dir = out_dir / _sanitize_name(mp_path.stem)
            out_txt = sub_dir / "mismatch_list.txt"
            window_csv = (sub_dir / "window_predictions.csv"
                          if args.save_windows else None)
            probs_csv = (sub_dir / Path(args.save_probs).name
                         if args.save_probs else None)
            switch_csv = (sub_dir / "switches.csv"
                          if args.save_switches else None)
        else:
            single_out = Path(args.out).expanduser()
            if not single_out.is_absolute():
                single_out = Path.cwd() / single_out
            out_txt = single_out
            base_dir = single_out.parent
            window_csv = (base_dir / "window_predictions.csv"
                          if args.save_windows else None)
            probs_csv = (Path(args.save_probs).expanduser()
                         if args.save_probs else None)
            if probs_csv is not None and not probs_csv.is_absolute():
                probs_csv = Path.cwd() / probs_csv
            switch_csv = (base_dir / "switches.csv"
                          if args.save_switches else None)

        stats = _run_one_model(
            model_path=mp_path,
            file_records=file_records,
            device=args.device,
            sample_rate=args.sample_rate,
            seconds=args.seconds,
            hop_seconds=args.hop_seconds,
            workers=workers,
            file_batch=file_batch,
            gpu_batch=gpu_batch,
            aggregate=args.aggregate,
            include_prob=args.include_prob,
            out_txt=out_txt,
            time_smooth=args.time_smooth,
            high_thresh=args.high_thresh,
            window_size=args.window_size,
            vote_needed=args.vote_needed,
            min_conf=args.min_conf,
            window_csv=window_csv,
            probs_csv=probs_csv,
            switch_csv=switch_csv,
            plot_dir=(out_txt.parent / "switch_plots" if args.plot_switches else None),
        )
        all_stats.append(stats)
        _print_model_summary(stats)

    if multi_mode:
        _print_comparison(all_stats)
        summary_path = out_dir / "summary.txt"
        _write_summary_txt(all_stats, summary_path)
        print(f"\n汇总写入     : {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
