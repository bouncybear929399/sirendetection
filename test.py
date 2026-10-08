#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频推理 - 第二步（支持类别文件夹 + 准确率统计 + 批量模型评测）

目录结构：
    <数据集根目录>/
        BKN/  xxx.wav  ...
        POL/  xxx.wav  ...
        FIR/  xxx.wav  ...
        AMB/  xxx.wav  ...
        ENG/  xxx.wav  ...

真实类别来源优先级：
    1. 音频父文件夹名（BKN / POL / FIR / AMB / ENG，兼容中文别名）
    2. 文件名下划线前的关键词

两种运行模式：
    A) 单模型：--model path/to/xxx.pt
        → 输出一份 mismatch_list.txt
    B) 多模型：--models-dir path/to/models/
        → 遍历目录下所有 *.pt，每个模型一份结果，写进 <out-dir>/<model>/mismatch_list.txt
        → 额外产出 <out-dir>/summary.txt 汇总所有模型的整体/各类准确率
"""


from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import re
import sys
from math import gcd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch


CLASS_NAMES = ["BKN", "POL", "FIR", "AMB", "ENG"]

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


def _init_worker(model_path: str, device: str, sample_rate: int, seconds: float) -> None:
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
    _WORKER["target_len"] = int(round(sample_rate * seconds))


def _load_wav(path: str, sample_rate: int, target_len: int) -> np.ndarray:
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
        batch = np.stack(valid_waves, axis=0)
        tensor = torch.from_numpy(batch).unsqueeze(1).to(device)
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
# 真实类别推断
# --------------------------------------------------------------------------- #

def _get_ground_truth(path_str: str) -> Optional[str]:
    p = Path(path_str)
    parent = p.parent.name.strip()
    gt = FOLDER_ALIASES.get(parent) or FOLDER_ALIASES.get(parent.upper())
    if gt is not None:
        return gt

    stem = p.stem
    if "_" in stem:
        kw = stem.rsplit("_", 1)[0].strip()
        gt = FOLDER_ALIASES.get(kw) or FOLDER_ALIASES.get(kw.upper())
        if gt is not None:
            return gt
    return None


# --------------------------------------------------------------------------- #
# 打印 / 写文件辅助
# --------------------------------------------------------------------------- #

def _sanitize_name(name: str) -> str:
    """把模型文件名清洗成安全的子目录名。"""
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
    print(f"参与评估         : {stats['total_eval']}")
    print(f"读取失败         : {stats['read_fail']}")
    print(f"预测正确         : {stats['total_correct']}")
    pred_wrong = stats["total_eval"] - stats["total_correct"] - stats["read_fail"]
    print(f"预测错误         : {pred_wrong}")
    print(f"总体准确率       : {stats['overall_acc']:.2f}%")
    print("=" * 64)

    print("各类别准确率:")
    for cls in CLASS_NAMES:
        tot = stats["per_class_total"][cls]
        cor = stats["per_class_correct"][cls]
        if tot == 0:
            print(f"  {cls:4s}  : 无样本")
        else:
            print(f"  {cls:4s}  : {cor} / {tot}  = {100.0 * cor / tot:.2f}%")

    print("\n混淆矩阵 (行=真实, 列=预测, ERR=读取失败):")
    _print_confusion(stats["confusion"])
    print(f"\n错误清单: {stats['out_txt']}")


def _print_comparison(all_stats: List[Dict]) -> None:
    print("\n" + "=" * 96)
    print("模型对比汇总")
    print("=" * 96)
    header = "模型".ljust(36) + "总体".rjust(10)
    for cls in CLASS_NAMES:
        header += cls.rjust(10)
    print(header)
    print("-" * len(header))
    for st in all_stats:
        name = st["model"]
        if len(name) > 34:
            name = name[:31] + "..."
        row = name.ljust(36) + f"{st['overall_acc']:.2f}%".rjust(10)
        for cls in CLASS_NAMES:
            tot = st["per_class_total"][cls]
            cor = st["per_class_correct"][cls]
            if tot == 0:
                row += "    --".rjust(10)
            else:
                row += f"{100.0 * cor / tot:.2f}%".rjust(10)
        print(row)
    print("=" * 96)


def _write_summary_txt(all_stats: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        header = ["model", "model_path", "overall_acc", "total_eval",
                  "total_correct", "read_fail"]
        header += [f"{c}_acc" for c in CLASS_NAMES]
        f.write("\t".join(header) + "\n")

        for st in all_stats:
            cols = [
                st["model"],
                st["model_path"],
                f"{st['overall_acc']:.4f}",
                str(st["total_eval"]),
                str(st["total_correct"]),
                str(st["read_fail"]),
            ]
            for cls in CLASS_NAMES:
                tot = st["per_class_total"][cls]
                cor = st["per_class_correct"][cls]
                acc = 100.0 * cor / tot if tot > 0 else 0.0
                cols.append(f"{acc:.4f}")
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
    workers: int,
    batch_size: int,
    include_prob: bool,
    out_txt: Path,
) -> Dict:
    """跑一个模型，返回统计信息 dict。"""
    paths = [rec[0] for rec in file_records]
    batches = [paths[i:i + batch_size] for i in range(0, len(paths), batch_size)]

    total = len(paths)
    print(f"\n[model] {model_path.name}  ({total} files)")

    ctx = mp.get_context("spawn")
    all_results: List[Tuple[str, int, List[float]]] = []
    done = 0

    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(str(model_path), device, sample_rate, seconds),
    ) as pool:
        for result in pool.imap_unordered(_infer_batch, batches):
            all_results.extend(result)
            done += len(result)
            print(f"\r  进度 {done}/{total}", end="", flush=True)
    print()

    # ---------- 统计 ---------- #
    per_class_total = {c: 0 for c in CLASS_NAMES}
    per_class_correct = {c: 0 for c in CLASS_NAMES}
    confusion = {c: {p: 0 for p in CLASS_NAMES + ["ERR"]} for c in CLASS_NAMES}
    mismatches: List[Tuple[str, str, str, List[float]]] = []
    read_fail_count = 0

    gt_map = {rec[0]: rec[1] for rec in file_records}

    for path_str, pred_idx, probs in all_results:
        fname = os.path.basename(path_str)
        gt = gt_map.get(path_str)
        if gt is None:
            continue

        per_class_total[gt] += 1

        if pred_idx < 0:
            read_fail_count += 1
            confusion[gt]["ERR"] += 1
            mismatches.append((fname, gt, "READ_ERROR", probs))
            continue

        pred_name = CLASS_NAMES[pred_idx]
        confusion[gt][pred_name] += 1

        if gt == pred_name:
            per_class_correct[gt] += 1
        else:
            mismatches.append((fname, gt, pred_name, probs))

    mismatches.sort(key=lambda x: x[0])

    # ---------- 写单模型错误清单 ---------- #
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, "w", encoding="utf-8") as f:
        for fname, gt, pred_name, probs in mismatches:
            if include_prob and probs:
                score = " ".join(f"{n}={probs[i]:.4f}" for i, n in enumerate(CLASS_NAMES))
                f.write(f"{fname}\tlabel={gt}\tpredict={pred_name}\t{score}\n")
            else:
                f.write(f"{fname}\n")

    total_eval = sum(per_class_total.values())
    total_correct = sum(per_class_correct.values())
    overall_acc = 100.0 * total_correct / total_eval if total_eval > 0 else 0.0

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
        "out_txt": str(out_txt),
    }


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="多进程推理 + 类别准确率统计 + 批量模型评测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("folder", help="数据集根目录（下面有 BKN/POL/... 子文件夹）")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--model", default=None,
                     help="单个 TorchScript .pt 模型（默认走 DEFAULT_MODEL）")
    src.add_argument("--models-dir", default=None,
                     help="模型文件夹，遍历其中所有 *.pt 依次评测")

    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--device", default="cpu", help="cpu / cuda / cuda:0 ...")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=16)

    p.add_argument("--out", default="mismatch_list.txt",
                   help="单模型模式下错误清单的输出路径")
    p.add_argument("--out-dir", default="model_eval_results",
                   help="多模型模式下的结果输出目录")
    p.add_argument("--include-prob", action="store_true",
                   help="错误清单中附带预测类别和五类概率")
    p.add_argument("--suffix", default=".wav")
    p.add_argument("--no-recursive", action="store_true",
                   help="只扫描根目录（默认递归扫描子文件夹）")
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

    # ---------- 决定模型列表 ---------- #
    if args.models_dir:
        models_dir = Path(args.models_dir).expanduser().resolve()
        if not models_dir.is_dir():
            print(f"[error] 模型目录不存在: {models_dir}", file=sys.stderr)
            return 2
        model_paths = sorted(
            p for p in models_dir.glob("*.pt") if p.is_file() and not p.name.startswith(".")
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

    # ---------- 设备检查 ---------- #
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA 不可用，自动回退到 CPU。")
        args.device = "cpu"

    # ---------- 扫描音频 + 预计算 gt ---------- #
    recursive = not args.no_recursive
    files = _scan(folder, args.suffix, recursive)
    if not files:
        print(f"[warn] 目录下没有 {args.suffix} 文件: {folder}")
        return 0

    file_records: List[Tuple[str, Optional[str]]] = []
    for p in files:
        file_records.append((str(p), _get_ground_truth(str(p))))

    parse_fail = sum(1 for _, gt in file_records if gt is None)

    workers = max(1, args.workers)
    bs = max(1, args.batch_size)

    # ---------- 打印运行头 ---------- #
    print("=" * 64)
    print(f"[info] 数据集     : {folder}")
    print(f"[info] 音频总数   : {len(files)}   无法解析类别: {parse_fail}")
    print(f"[info] 设备       : {args.device}")
    print(f"[info] 进程/batch : {workers} / {bs}")
    print(f"[info] 递归扫描   : {recursive}")
    print(f"[info] 输入规格   : {args.sample_rate} Hz / {args.seconds}s / mono")
    print(f"[info] 模型数量   : {len(model_paths)}")
    print("=" * 64)

    # ---------- 决定输出路径 ---------- #
    if multi_mode:
        out_dir = Path(args.out_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        single_out = Path(args.out).expanduser()
        if not single_out.is_absolute():
            single_out = Path.cwd() / single_out

    # ---------- 逐个模型评测 ---------- #
    all_stats: List[Dict] = []
    for idx, mp_path in enumerate(model_paths, 1):
        print(f"\n({idx}/{len(model_paths)}) {mp_path}")

        if multi_mode:
            sub_dir = out_dir / _sanitize_name(mp_path.stem)
            out_txt = sub_dir / "mismatch_list.txt"
        else:
            out_txt = single_out

        stats = _run_one_model(
            model_path=mp_path,
            file_records=file_records,
            device=args.device,
            sample_rate=args.sample_rate,
            seconds=args.seconds,
            workers=workers,
            batch_size=bs,
            include_prob=args.include_prob,
            out_txt=out_txt,
        )
        all_stats.append(stats)
        _print_model_summary(stats)

    # ---------- 汇总 ---------- #
    if multi_mode:
        _print_comparison(all_stats)
        summary_path = out_dir / "summary.txt"
        _write_summary_txt(all_stats, summary_path)
        print(f"\n汇总写入     : {summary_path}")
        print(f"单模型结果   : {out_dir}\\<模型名>\\mismatch_list.txt")
    else:
        print(f"\n错误清单写入 : {all_stats[0]['out_txt']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())