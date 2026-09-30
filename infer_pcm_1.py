# -*- coding: utf-8 -*-
import argparse
import glob
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch


CLASS_NAMES = ["BKN", "POL", "FIR", "AMB", "ENG"]
DEFAULT_MODEL = ("./postprocess_android/914/checkpoint_best(0914)_single_threshold_0.8.ptcheckpoint_best(0914)_single_threshold_0.8.pt")
# 每个推理窗口长度为 2 秒，窗口之间默认移动 0.5 秒。
DEFAULT_SECONDS = 2.0
DEFAULT_HOP_SECONDS = 0.5

# 默认 PCM 路径写在代码里；不传 --pcm 时会按这里的列表批量推理。
# 可以填写单个 PCM 文件、目录或通配符，例如 r"E:\data\pcm\*.pcm"。
DEFAULT_PCM_PATHS = [
     "/mnt/e/BaiduNetdiskDownload/数据/ENG_左后20米.pcm",
      "/mnt/e/BaiduNetdiskDownload/数据/AMB_左后20米.pcm",
      "/mnt/e/BaiduNetdiskDownload/数据/FIR_左后20米.pcm",
      "/mnt/e/BaiduNetdiskDownload/数据/POL_100米位置右后.pcm",
      "/mnt/e/BaiduNetdiskDownload/数据/AMB_100米位置右后.pcm",
      "/mnt/e/BaiduNetdiskDownload/数据/FIR_100米位置右后.pcm",
      "/mnt/e/BaiduNetdiskDownload/数据/FIR_前.pcm",
      "/mnt/e/BaiduNetdiskDownload/数据/BKN.pcm"
     # "/mnt/e/datasets/direction_dataset/ENG/静止_30m_正后_S4_001.pcm"

]

LABEL_ALIASES = {
    "BKN": 0,
    "NOISE": 0,
    "BACKGROUND": 0,
    "噪声": 0,
    "背景": 0,
    "POL": 1,
    "警车": 1,
    "FIR": 2,
    "消防车": 2,
    "消防": 2,
    "AMB": 3,
    "救护车": 3,
    "救护": 3,
    "ENG": 4,
    "工程车": 4,
    "工程": 4,
}


def collect_pcm_files(pcm_inputs):
    """支持一次传入多个 PCM 文件、目录或通配符；目录会递归收集 .pcm 文件。"""
    pcm_files = []
    for pcm_input in pcm_inputs:
        matched_paths = glob.glob(pcm_input)
        candidates = matched_paths if matched_paths else [pcm_input]

        for candidate in candidates:
            if os.path.isfile(candidate):
                if Path(candidate).suffix.lower() == ".pcm":
                    pcm_files.append(os.path.abspath(candidate))
            elif os.path.isdir(candidate):
                for path in Path(candidate).rglob("*.pcm"):
                    pcm_files.append(str(path.resolve()))
            else:
                raise FileNotFoundError(f"PCM path not found: {candidate}")

    pcm_files = sorted(set(pcm_files))
    if not pcm_files:
        raise RuntimeError(f"No PCM files found from inputs: {pcm_inputs}")
    return pcm_files


def load_pcm_chunks(pcm_path, sample_rate, seconds, channels, channel, dtype, hop_seconds=None):
    """直接从原始 PCM 文件读取数据，并按滑动窗口切分后逐段推理。"""
    pcm_data = np.fromfile(pcm_path, dtype=dtype)
    if pcm_data.size == 0:
        raise ValueError(f"PCM file is empty: {pcm_path}")

    if channels <= 0:
        raise ValueError(f"channels must be positive, but got {channels}")

    if pcm_data.size % channels != 0:
        usable = pcm_data.size // channels * channels
        pcm_data = pcm_data[:usable]

    if channels > 1:
        # 多通道 PCM 先 reshape 成 [帧数, 通道数]，再取指定通道。
        frame_count = pcm_data.size // channels
        pcm_data = pcm_data.reshape(frame_count, channels)
        if channel < 0 or channel >= channels:
            raise ValueError(f"channel index {channel} out of range for {channels} channels")
        pcm_data = pcm_data[:, channel]

    chunk_samples = int(sample_rate * seconds)
    # hop_seconds=0.5 时，窗口依次为 0-2s、0.5-2.5s、1.0-3.0s。
    hop_samples = int(sample_rate * hop_seconds) if hop_seconds is not None else chunk_samples
    hop_samples = max(1, hop_samples)
    if chunk_samples <= 0:
        raise ValueError(f"seconds must be positive, but got {seconds}")

    chunks = []
    starts = []
    if len(pcm_data) < chunk_samples:
        # 比一帧还短时补零，避免短音频无法推理。
        padded = np.zeros(chunk_samples, dtype=pcm_data.dtype)
        padded[: len(pcm_data)] = pcm_data
        return [padded], [0]

    for start in range(0, len(pcm_data) - chunk_samples + 1, hop_samples):
        chunks.append(pcm_data[start:start + chunk_samples])
        starts.append(start)

    return chunks, starts


def pcm_to_float_tensor(chunk, dtype, device):
    """整型 PCM 先归一化到 [-1, 1]，与训练/导出模型的输入假设保持一致。"""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        scale = max(abs(info.min), abs(info.max))
        audio = chunk.astype(np.float32) / float(scale)
    else:
        audio = chunk.astype(np.float32)

    audio = np.clip(audio, -1.0, 1.0)
    return torch.from_numpy(audio).view(1, 1, -1).to(device)


def infer_chunk(model, chunk, dtype, device):
    """对单个 PCM chunk 推理，直接使用 5 类 softmax 的 argmax 作为结果。"""
    x = pcm_to_float_tensor(chunk, dtype, device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_prob = float(probs[pred_idx].item())
    return pred_idx, pred_prob, probs.cpu()


def parse_label(label):
    """支持数字标签和字符串标签，方便命令行快速对照真值。"""
    if label is None:
        return None

    label_text = str(label).strip()
    if not label_text:
        return None

    if label_text.isdigit():
        label_idx = int(label_text)
        if 0 <= label_idx < len(CLASS_NAMES):
            return label_idx
        raise ValueError(f"label index out of range: {label}")

    label_upper = label_text.upper()
    if label_upper in LABEL_ALIASES:
        return LABEL_ALIASES[label_upper]

    raise ValueError(f"unknown label: {label}. Use one of {CLASS_NAMES} or 0-4.")


def infer_label_from_path(pcm_path):
    """如果路径里自带类别名，就自动推断真值标签。"""
    abs_path = os.path.abspath(pcm_path)
    path_upper = abs_path.upper()
    for idx, class_name in enumerate(CLASS_NAMES):
        if class_name in path_upper:
            return idx

    keyword_pairs = [
        (0, ["噪声", "背景", "noise", "background", "bkn"]),
        (1, ["警车", "pol"]),
        (2, ["消防车", "消防", "fir"]),
        (3, ["救护车", "救护", "amb"]),
        (4, ["工程车", "工程", "eng"]),
    ]
    lower_path = abs_path.lower()
    for label_idx, keywords in keyword_pairs:
        for keyword in keywords:
            if keyword.isascii():
                if keyword in lower_path:
                    return label_idx
            else:
                if keyword in abs_path:
                    return label_idx
    return None


def infer_one_pcm(model, pcm_path, args, dtype, device):
    """推理单个 PCM 文件，返回本文件的窗口级统计。"""
    expected_label = parse_label(args.label)
    if expected_label is None:
        expected_label = infer_label_from_path(pcm_path)

    chunks, starts = load_pcm_chunks(
        pcm_path=pcm_path,
        sample_rate=args.sample_rate,
        seconds=args.seconds,
        channels=args.channels,
        channel=args.channel,
        dtype=dtype,
        hop_seconds=args.hop_seconds,
    )

    cls_counter = Counter()
    correct_count = 0

    if not args.quiet:
        print("=" * 100)
        print(f"PCM   : {os.path.abspath(pcm_path)}")
        print(f"Format: {args.dtype}, {args.channels}ch, channel={args.channel}, {args.sample_rate} Hz")
        print(f"Chunks: {len(chunks)} x {args.seconds:.2f}s")
        if args.hop_seconds is not None:
            print(f"Hop   : {args.hop_seconds:.2f}s")
        if expected_label is not None:
            print(f"Label : {CLASS_NAMES[expected_label]} ({expected_label})")

    for idx, (chunk, start_sample) in enumerate(zip(chunks, starts), start=1):
        pred_idx, pred_prob, probs = infer_chunk(model, chunk, dtype, device)
        cls_counter[CLASS_NAMES[pred_idx]] += 1
        if expected_label is not None and pred_idx == expected_label:
            correct_count += 1

        if not args.quiet:
            start_sec = start_sample / float(args.sample_rate)
            score_text = ", ".join(f"{name}:{probs[i]:.3f}" for i, name in enumerate(CLASS_NAMES))
            print(
                f"[{idx:03d}] t={start_sec:8.3f}s predict={CLASS_NAMES[pred_idx]}({pred_idx}) "
                f"pred_prob={pred_prob:.4f} | {score_text}"
            )

    total = len(chunks)
    accuracy = correct_count / total if expected_label is not None and total else None
    return {
        "path": pcm_path,
        "file": os.path.basename(pcm_path),
        "label": CLASS_NAMES[expected_label] if expected_label is not None else "",
        "total": total,
        "counter": cls_counter,
        "correct": correct_count,
        "accuracy": accuracy,
    }


def print_file_accuracy_table(rows):
    """最后按文件输出准确率表，形式接近 evaluate_lite_gru_field_dataset.py。"""
    print("\nPCM 文件推理准确率：")
    print("| 文件 | 标签 | Noise | POL | FIR | AMB | ENG | 总窗口 | 准确率 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        counter = row["counter"]
        acc_text = "N/A" if row["accuracy"] is None else f"{row['accuracy'] * 100:.2f}%"
        print(
            f"| {row['file']} | {row['label'] or '-'} | {counter['BKN']} | {counter['POL']} | "
            f"{counter['FIR']} | {counter['AMB']} | {counter['ENG']} | {row['total']} | {acc_text} |"
        )


def print_overall_summary(rows):
    """输出所有 PCM 合并后的整体统计。"""
    total_counter = Counter()
    total_windows = 0
    total_correct = 0
    labeled_windows = 0

    for row in rows:
        total_counter.update(row["counter"])
        total_windows += row["total"]
        if row["accuracy"] is not None:
            total_correct += row["correct"]
            labeled_windows += row["total"]

    print("\n整体统计：")
    print("class\tname\tcount\tratio")
    for class_name in CLASS_NAMES:
        count = total_counter[class_name]
        ratio = count / max(1, total_windows) * 100
        class_idx = CLASS_NAMES.index(class_name)
        print(f"{class_idx}\t{class_name}\t{count}\t{ratio:.2f}%")

    siren_count = total_windows - total_counter["BKN"]
    siren_ratio = siren_count / max(1, total_windows) * 100
    print(f"\nSiren detected: {siren_count}/{total_windows} ({siren_ratio:.2f}%)")

    if labeled_windows > 0:
        print(
            f"Overall class accuracy: {total_correct}/{labeled_windows} "
            f"({total_correct / labeled_windows * 100:.2f}%)"
        )
    else:
        print("Overall class accuracy: N/A (no labels inferred)")


def main(args):
    device = torch.device(args.device)
    dtype = np.dtype(args.dtype)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"PT model not found: {args.model}")

    pcm_files = collect_pcm_files(args.pcm)

    # 只加载一次模型，多个 PCM 复用同一个 TorchScript 模型实例。
    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    print(f"Model : {os.path.abspath(args.model)}")
    print(f"PCM files: {len(pcm_files)}")
    print("Postprocess: disabled, using direct argmax over 5-class softmax.")

    rows = []
    for pcm_path in pcm_files:
        rows.append(infer_one_pcm(model, pcm_path, args, dtype, device))
    print(f"Model : {os.path.abspath(args.model)}")
    print_file_accuracy_table(rows)
    print_overall_summary(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Run siren inference on one or more raw PCM files.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--pcm",
        nargs="+",
        default=DEFAULT_PCM_PATHS,
        help="One or more PCM files, directories, or wildcard patterns. Directories are searched recursively.",
    )
    parser.add_argument("--sample_rate", type=int, default=48000)
    parser.add_argument("--seconds", type=float, default=DEFAULT_SECONDS, help="每个推理窗口的时长，默认 2 秒。")
    parser.add_argument(
        "--hop_seconds",
        type=float,
        default=DEFAULT_HOP_SECONDS,
        help="滑动窗口移动步长，默认 0.5 秒；例如窗口为 0-2s、0.5-2.5s。",
    )
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--dtype", default="int16", choices=["int16", "int32", "float32"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--label", default=None, help="True class for accuracy, e.g. AMB, POL, FIR, ENG, BKN, or 0-4.")
    parser.add_argument(
        "--infer_label_from_path",
        action="store_true",
        help="Deprecated: labels are now inferred from file path by default when --label is not provided.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print final per-file accuracy table and overall summary.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
