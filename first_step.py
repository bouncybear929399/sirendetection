#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频数据集统一预处理工具 —— 第一步

功能
----
1. 读取输入文件夹下所有音频：
     - 常见编码格式：wav / flac / ogg / mp3 / opus / aiff ...（soundfile 优先，失败回退 ffmpeg）
     - 裸 PCM：需手动指定采样率 / 声道 / 位宽 / 字节序 / 是否为浮点
2. 统一转为 单声道 + 目标采样率（默认 16 kHz）的 float32 内部表示；
3. 按 2 秒切分：
     - 完整 2 秒片段            → 直接保留；
     - 末尾剩余 >= 1s 且 < 2s   → 补静音到 2 秒后保留；
     - 末尾剩余 < 1s            → 丢弃；
4. 输出 16-bit PCM WAV，命名 `{keyword}_{序号:05d}.wav`，写入指定输出文件夹。

用法示例
--------
# 普通音频文件
python step1_preprocess.py ./raw_audio ./dataset_out -k wakeword

# 裸 PCM（16k / 单声道 / 16bit / 小端）
python step1_preprocess.py ./raw_pcm ./dataset_out -k wakeword \
    --pcm --pcm-sr 16000 --pcm-channels 1 --pcm-width 2

# 裸 PCM（48k / 双声道 / 24bit / 大端）
python step1_preprocess.py ./raw_pcm ./dataset_out -k wakeword \
    --pcm --pcm-sr 48000 --pcm-channels 2 --pcm-width 3 --pcm-big-endian

依赖
----
    pip install numpy soundfile scipy
（mp3/m4a 等如 soundfile 不支持，可额外安装 ffmpeg 并加入 PATH）
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

try:
    from scipy.signal import resample_poly
except ImportError:  # pragma: no cover
    resample_poly = None


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #

AUDIO_EXTS = {
    ".wav", ".flac", ".ogg", ".oga", ".opus", ".mp3", ".m4a", ".aac",
    ".aiff", ".aif", ".aifc", ".au", ".w64", ".caf", ".wma",
}
PCM_EXTS = {".pcm", ".raw", ".bin", ".dat"}


@dataclass
class PcmConfig:
    """裸 PCM 解析参数"""
    sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2          # 每样本字节数：1 / 2 / 3 / 4
    is_float: bool = False         # 4 字节时是否为 float32
    is_signed: bool = True         # 1 字节时是否为 int8
    byte_order: str = "<"          # '<' 小端 / '>' 大端


@dataclass
class TargetConfig:
    sample_rate: int = 16000
    channels: int = 1
    segment_seconds: float = 2.0
    min_keep_seconds: float = 1.0


# --------------------------------------------------------------------------- #
# 读取
# --------------------------------------------------------------------------- #

def _load_pcm(path: Path, cfg: PcmConfig) -> Tuple[np.ndarray, int]:
    """读取裸 PCM，返回 (单声道 float32 [-1,1], 采样率)。"""
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        raise ValueError("空文件")

    n = int(cfg.sample_width)
    bo = cfg.byte_order

    if n == 1:
        if cfg.is_float:
            raise ValueError("不支持 1 字节浮点 PCM")
        dtype = np.dtype("i1") if cfg.is_signed else np.dtype("u1")
        audio = raw.view(dtype).astype(np.float32) / 128.0
    elif n == 2:
        if cfg.is_float:
            raise ValueError("不支持 2 字节浮点 PCM")
        usable = (raw.size // 2) * 2
        audio = raw[:usable].view(np.dtype(bo + "i2")).astype(np.float32) / 32768.0
    elif n == 3:
        usable = (raw.size // 3) * 3
        b = raw[:usable].reshape(-1, 3).astype(np.int32)
        if bo == ">":
            b = b[:, ::-1]
        val = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        val = np.where(val & 0x800000, val - 0x1000000, val)
        audio = val.astype(np.float32) / float(1 << 23)
    elif n == 4:
        usable = (raw.size // 4) * 4
        if cfg.is_float:
            audio = raw[:usable].view(np.dtype(bo + "f4")).astype(np.float32)
        else:
            audio = raw[:usable].view(np.dtype(bo + "i4")).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"不支持的 sample_width={n}（只支持 1/2/3/4）")

    # 声道处理
    ch = int(cfg.channels)
    if ch < 1:
        raise ValueError("channels 必须 >= 1")
    usable = (audio.size // ch) * ch
    audio = audio[:usable].reshape(-1, ch)
    mono = audio.mean(axis=1) if ch > 1 else audio[:, 0]

    return np.ascontiguousarray(mono, dtype=np.float32), int(cfg.sample_rate)


def _load_encoded(path: Path) -> Tuple[np.ndarray, int]:
    """用 soundfile 读取编码音频。"""
    if sf is None:
        raise RuntimeError("缺少 soundfile，请 pip install soundfile")
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(mono, dtype=np.float32), int(sr)


def _load_encoded_ffmpeg(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """soundfile 失败时的回退：用 ffmpeg 直接解码成单声道 f32le。"""
    import subprocess

    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(target_sr), "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg 未安装或不在 PATH 中") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg 解码失败: {e.stderr.decode(errors='ignore')}") from e

    audio = np.frombuffer(out, dtype="<f4").astype(np.float32)
    return np.ascontiguousarray(audio), int(target_sr)


def load_audio(
    path: Path,
    pcm_cfg: Optional[PcmConfig],
    target_sr: int,
) -> Tuple[np.ndarray, int]:
    """统一入口：返回 (单声道 float32, 原始采样率)。"""
    if pcm_cfg is not None:
        return _load_pcm(path, pcm_cfg)
    try:
        return _load_encoded(path)
    except Exception as e:
        print(f"  [warn] soundfile 读取失败({e})，改用 ffmpeg ...")
        return _load_encoded_ffmpeg(path, target_sr)


# --------------------------------------------------------------------------- #
# 重采样 / 切分
# --------------------------------------------------------------------------- #

def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if audio.size == 0 or orig_sr == target_sr:
        return audio
    if resample_poly is None:
        raise RuntimeError("需要 scipy，请 pip install scipy")
    g = math.gcd(int(orig_sr), int(target_sr))
    up, down = int(target_sr) // g, int(orig_sr) // g
    return resample_poly(audio, up, down).astype(np.float32)


def split_segments(
    audio: np.ndarray,
    sr: int,
    segment_seconds: float,
    min_keep_seconds: float,
) -> List[np.ndarray]:
    """
    按固定长度切分：
      - 完整片段直接保留；
      - 末尾剩余 >= min_keep 的补静音到整段长度；
      - 末尾剩余 <  min_keep 的丢弃。
    """
    seg_len = int(round(sr * segment_seconds))
    min_len = int(round(sr * min_keep_seconds))
    if seg_len <= 0:
        raise ValueError("segment_seconds 过小")

    segments: List[np.ndarray] = []
    n = audio.size
    pos = 0

    while pos + seg_len <= n:
        segments.append(np.ascontiguousarray(audio[pos:pos + seg_len]))
        pos += seg_len

    rest = audio[pos:]
    if rest.size >= min_len:
        buf = np.zeros(seg_len, dtype=np.float32)
        buf[: rest.size] = rest
        segments.append(buf)

    return segments


# --------------------------------------------------------------------------- #
# 写盘
# --------------------------------------------------------------------------- #

_IDX_RE_CACHE: dict = {}


def _next_index(out_dir: Path, keyword: str) -> int:
    """扫描输出目录中已有的 {keyword}_NNNNN.wav，返回下一个可用序号。"""
    pat = _IDX_RE_CACHE.get(keyword)
    if pat is None:
        pat = re.compile(rf"^{re.escape(keyword)}_(\d+)\.wav$", re.IGNORECASE)
        _IDX_RE_CACHE[keyword] = pat

    max_idx = -1
    for f in out_dir.iterdir():
        if f.is_file():
            m = pat.match(f.name)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def write_segments(
    segments: List[np.ndarray],
    out_dir: Path,
    keyword: str,
    start_index: int,
    sr: int,
) -> List[Path]:
    if sf is None:
        raise RuntimeError("缺少 soundfile，请 pip install soundfile")
    written: List[Path] = []
    for i, seg in enumerate(segments):
        idx = start_index + i
        out_path = out_dir / f"{keyword}_{idx:05d}.wav"
        sf.write(str(out_path), seg, sr, subtype="PCM_16")
        written.append(out_path)
    return written


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def iter_input_files(input_dir: Path, recursive: bool, pcm_mode: bool) -> Iterator[Path]:
    it = input_dir.rglob("*") if recursive else input_dir.glob("*")
    for p in sorted(it):
        if not p.is_file() or p.name.startswith("."):
            continue
        if pcm_mode:
            yield p                      # PCM 模式下把所有文件都当作候选
        elif p.suffix.lower() in AUDIO_EXTS:
            yield p


def run(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"[error] 输入目录不存在: {input_dir}", file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    pcm_cfg = None
    if args.pcm:
        pcm_cfg = PcmConfig(
            sample_rate=args.pcm_sr,
            channels=args.pcm_channels,
            sample_width=args.pcm_width,
            is_float=args.pcm_float,
            is_signed=not args.pcm_unsigned,
            byte_order=">" if args.pcm_big_endian else "<",
        )

    target = TargetConfig(
        sample_rate=args.target_sr,
        channels=1,
        segment_seconds=args.segment_seconds,
        min_keep_seconds=args.min_keep_seconds,
    )

    files = list(iter_input_files(input_dir, args.recursive, args.pcm))
    if not files:
        print("[warn] 输入目录下没有找到可处理的文件")
        return 0

    print(f"[info] 输入目录 : {input_dir}")
    print(f"[info] 输出目录 : {output_dir}")
    print(f"[info] 关键词   : {args.keyword}")
    print(f"[info] 目标格式 : 单声道 / {target.sample_rate} Hz / "
          f"{target.segment_seconds}s 片段 / PCM_16 WAV")
    if pcm_cfg:
        print(f"[info] PCM 参数 : sr={pcm_cfg.sample_rate} ch={pcm_cfg.channels} "
              f"width={pcm_cfg.sample_width}B float={pcm_cfg.is_float} "
              f"endian={pcm_cfg.byte_order}")
    print(f"[info] 待处理文件数: {len(files)}\n")

    next_index = _next_index(output_dir, args.keyword)

    ok_files = 0
    fail_files = 0
    total_segments = 0
    total_dropped = 0

    for i, path in enumerate(files, 1):
        rel = path.relative_to(input_dir)
        try:
            audio, sr = load_audio(path, pcm_cfg, target.sample_rate)
            if audio.size == 0:
                raise ValueError("音频为空")

            audio = resample(audio, sr, target.sample_rate)

            # 统计丢弃的尾部（仅用于日志展示）
            seg_len = int(round(target.sample_rate * target.segment_seconds))
            min_len = int(round(target.sample_rate * target.min_keep_seconds))
            tail = audio.size % seg_len
            dropped = 1 if (0 < tail < min_len) else 0

            segments = split_segments(
                audio,
                target.sample_rate,
                target.segment_seconds,
                target.min_keep_seconds,
            )

            if not segments:
                print(f"[{i}/{len(files)}] {rel}  "
                      f"时长 {audio.size / target.sample_rate:.3f}s → 无有效片段，跳过")
                ok_files += 1
                total_dropped += 1
                continue

            written = write_segments(
                segments, output_dir, args.keyword, next_index, target.sample_rate
            )
            next_index += len(written)
            total_segments += len(written)
            total_dropped += dropped
            ok_files += 1

            print(f"[{i}/{len(files)}] {rel}  "
                  f"{sr}Hz {audio.size / target.sample_rate:.3f}s → "
                  f"{len(written)} 段"
                  + (f"（丢弃尾部 {dropped} 段）" if dropped else ""))

        except Exception as e:
            fail_files += 1
            print(f"[{i}/{len(files)}] {rel}  [error] {e}", file=sys.stderr)

    print("\n" + "=" * 60)
    print(f"完成：成功 {ok_files} 个文件，失败 {fail_files} 个")
    print(f"输出片段总数 : {total_segments}")
    print(f"丢弃尾部片段 : {total_dropped}")
    print(f"输出目录     : {output_dir}")
    print("=" * 60)
    return 0 if fail_files == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="音频数据集统一预处理（第一步）：单声道 / 16k / 2s 切片",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input_dir", help="输入音频文件夹")
    ap.add_argument("output_dir", help="输出文件夹")
    ap.add_argument("-k", "--keyword", required=True, help="输出文件名关键词")

    ap.add_argument("-r", "--recursive", action="store_true", help="递归子目录")

    ap.add_argument("--target-sr", type=int, default=16000, help="目标采样率")
    ap.add_argument("--segment-seconds", type=float, default=2.0, help="切片长度（秒）")
    ap.add_argument("--min-keep-seconds", type=float, default=1.0,
                    help="末尾保留阈值（秒），小于该值的尾部片段丢弃；"
                         "大于等于该值且不足切片长度的补静音")

    # PCM 相关
    ap.add_argument("--pcm", action="store_true", help="输入为裸 PCM")
    ap.add_argument("--pcm-sr", type=int, default=16000, help="PCM 采样率")
    ap.add_argument("--pcm-channels", type=int, default=1, help="PCM 声道数")
    ap.add_argument("--pcm-width", type=int, default=2,
                    choices=[1, 2, 3, 4], help="PCM 每样本字节数")
    ap.add_argument("--pcm-float", action="store_true",
                    help="PCM 为 float32（仅 width=4 时有效）")
    ap.add_argument("--pcm-unsigned", action="store_true",
                    help="PCM 为无符号（仅 width=1 时有效）")
    ap.add_argument("--pcm-big-endian", action="store_true", help="PCM 为大端序")

    return ap


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("\n[abort] 用户中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())