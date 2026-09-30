# -*- coding: utf-8 -*-
"""对 WAV 音频按固定时间窗口进行警笛类型推理。"""

import argparse
import os
from math import gcd

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly


CLASS_NAMES = ["BKN", "POL", "FIR", "AMB", "ENG"]
DEFAULT_MODEL = ("./postprocess_android/913/siren_detection_model(0913)_triple_threshold_recommended.pt")
DEFAULT_WAV = "/mnt/d/test_real/动态FIR/消防_动态_左车道100m超车_75km.wav"


def load_wav_as_float(wav_path, target_sr, channel):
    """读取 WAV、选择一个通道并归一化为模型需要的 float32 波形。"""
    source_sr, audio = wavfile.read(wav_path)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    if channel < 0 or channel >= audio.shape[1]:
        raise ValueError(f"通道索引 {channel} 超出范围，音频共有 {audio.shape[1]} 个通道。")

    # 端到端模型训练时使用 [-1, 1] 的浮点波形；必须按原始整数位深进行归一化。
    selected = audio[:, channel]
    if np.issubdtype(selected.dtype, np.integer):
        info = np.iinfo(selected.dtype)
        selected = selected.astype(np.float32) / float(max(abs(info.min), abs(info.max)))
    else:
        selected = selected.astype(np.float32)
    selected = np.clip(selected, -1.0, 1.0)

    if source_sr != target_sr:
        # 用 SciPy 重采样，避免依赖 torchaudio 的 TorchCodec 组件。
        divisor = gcd(source_sr, target_sr)
        selected = resample_poly(selected, target_sr // divisor, source_sr // divisor).astype(np.float32)

    return selected, source_sr, audio.shape[1]


def run_inference(args):
    """按固定窗口和滑动步长逐段推理，并打印每段的五类概率。"""
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")
    if not os.path.isfile(args.wav):
        raise FileNotFoundError(f"WAV 文件不存在: {args.wav}")

    waveform, source_sr, channels = load_wav_as_float(args.wav, args.sample_rate, args.channel)
    window_samples = int(args.sample_rate * args.seconds)
    hop_samples = int(args.sample_rate * args.hop_seconds)
    if window_samples <= 0:
        raise ValueError("seconds 必须大于 0。")
    if hop_samples <= 0:
        raise ValueError("hop_seconds 必须大于 0。")

    # 每个窗口仍固定为 2 秒，但起点每次向后移动 0.5 秒，例如：
    # 0-2 秒、0.5-2.5 秒、1-3 秒。仅保留长度完整的窗口，不对末尾补零。
    if len(waveform) < window_samples:
        raise RuntimeError("音频不足一个推理窗口。")
    window_starts = list(range(0, len(waveform) - window_samples + 1, hop_samples))
    window_count = len(window_starts)

    device = torch.device(args.device)
    model = torch.jit.load(args.model, map_location=device)
    model.eval()

    print(f"模型: {os.path.abspath(args.model)}")
    print(f"音频: {os.path.abspath(args.wav)}")
    print(f"音频格式: {source_sr} Hz, {channels} 通道，使用通道 {args.channel}")
    print(
        f"模型输入: {args.sample_rate} Hz, 每段 {args.seconds:.2f}s, "
        f"滑动步长 {args.hop_seconds:.2f}s, 共 {window_count} 个完整窗口"
    )
    print("-" * 100)

    with torch.no_grad():
        for index, start_sample in enumerate(window_starts):
            chunk = waveform[start_sample:start_sample + window_samples]
            # TorchScript 端到端模型输入形状为 [B, 1, N]。
            input_tensor = torch.from_numpy(chunk.copy()).view(1, 1, -1).to(device)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu()
            pred_idx = int(torch.argmax(probs).item())
            score_text = ", ".join(f"{name}:{probs[i]:.3f}" for i, name in enumerate(CLASS_NAMES))
            start_sec = start_sample / args.sample_rate
            end_sec = (start_sample + window_samples) / args.sample_rate
            print(
                f"[{index + 1:03d}] t={start_sec:6.1f}-{end_sec:6.1f}s "
                f"predict={CLASS_NAMES[pred_idx]}({pred_idx}) prob={probs[pred_idx]:.4f} | {score_text}"
            )

    # 计算最后一个完整窗口之后剩余的音频长度。
    last_window_end = window_starts[-1] + window_samples
    remaining_seconds = (len(waveform) - last_window_end) / args.sample_rate
    if remaining_seconds > 0:
        print(f"末尾 {remaining_seconds:.3f}s 不足一个窗口，未参与推理。")


def parse_args():
    parser = argparse.ArgumentParser(description="按固定时长窗口推理 WAV 警笛音频。")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="TorchScript PT 模型路径。")
    parser.add_argument("--wav", default=DEFAULT_WAV, help="待测试的 WAV 文件路径。")
    parser.add_argument("--sample_rate", type=int, default=48000, help="模型输入采样率。")
    parser.add_argument("--seconds", type=float, default=2.0, help="每个推理窗口的时长。")
    parser.add_argument(
        "--hop_seconds",
        type=float,
        default=0.5,
        help="相邻窗口起点的时间间隔，默认每次向后滑动 0.5 秒。",
    )
    parser.add_argument("--channel", type=int, default=0, help="多通道 WAV 使用的通道索引。")
    parser.add_argument("--device", default="cpu", help="推理设备，例如 cpu 或 cuda。")
    return parser.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
