import math
import os
import random

import librosa
import numpy as np

import torch
import torch.nn.functional as F
import torchaudio


CLASS_NAMES = ["BKN", "POL", "FIR", "AMB", "ENG"]
# 数据集里允许扫描到的音频后缀。这里保留大小写两套写法，避免 Windows / Linux
# 下文件命名风格不同导致漏读样本。
AUDIO_EXTENSIONS = (".wav", ".WAV", ".flac", ".FLAC", ".mp3", ".MP3", ".ogg", ".OGG", ".m4a", ".M4A")
# 统一把英文类别名、中文类别名、常见别名都映射成模型内部使用的 5 个标准类别。
CLASS_NAME_ALIASES = {
    "BKN": "BKN",
    "BACKGROUND": "BKN",
    "NOISE": "BKN",
    "\u80cc\u666f": "BKN",
    "\u566a\u58f0": "BKN",
    "POL": "POL",
    "POLICE": "POL",
    "\u8b66\u8f66": "POL",
    "FIR": "FIR",
    "FIRE": "FIR",
    "\u6d88\u9632": "FIR",
    "\u6d88\u9632\u8f66": "FIR",
    "AMB": "AMB",
    "AMBULANCE": "AMB",
    "\u6551\u62a4": "AMB",
    "\u6551\u62a4\u8f66": "AMB",
    "\u6025\u6551": "AMB",
    "ENG": "ENG",
    "ENGINEERING": "ENG",
    "\u5de5\u7a0b": "ENG",
    "\u5de5\u7a0b\u8f66": "ENG",
}


def detect_audio_format(path):
    # 中文注释：有些数据文件虽然扩展名写成 .wav，但文件头其实是 MP3。
    # 这里先根据文件头做一次轻量判断，必要时显式告诉 torchaudio 按 MP3 解码。
    try:
        with open(path, "rb") as audio_file:
            header = audio_file.read(16)
    except OSError:
        return None

    if header.startswith(b"ID3"):
        return "mp3"

    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return "mp3"

    return None


def get_required_attr(config, name):
    # 中文注释：训练和增强的关键参数统一要求从配置文件提供，避免 dataset.py 自己再维护一套默认值。
    if not hasattr(config, name):
        raise AttributeError(f"Missing required config field: {name}")
    return getattr(config, name)


class AudioFeatureExtractor(object):
    def __init__(
        self,
        sample_rate,
        feature_type,
        n_fft,
        n_mels,
        n_mfcc,
        win_length,
        hop_length,
        f_min,
        f_max,
        power,
    ):
        super().__init__()
        # feature_type 用来统一控制当前实验到底抽取哪一种声学特征：
        # 1. logmel：更直接保留频谱能量分布
        # 2. mfcc：更强调包络信息，维度也更紧凑
        self.feature_type = str(feature_type).lower()
        self.output_dim = int(n_mfcc if self.feature_type == "mfcc" else n_mels)

        if self.feature_type == "mfcc":
            # MFCC 路径：先做 Mel 频谱，再取对数，最后做倒谱变换。
            self.transform = torchaudio.transforms.MFCC(
                sample_rate=sample_rate,
                n_mfcc=int(n_mfcc),
                log_mels=True,
                melkwargs={
                    "n_fft": int(n_fft),
                    "win_length": int(win_length),
                    "hop_length": int(hop_length),
                    "n_mels": int(n_mels),
                    "f_min": int(f_min),
                    "f_max": int(f_max),
                    "power": float(power),
                },
            )
        else:
            # log-Mel 路径：MelSpectrogram + 对数幅度。
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=int(n_fft),
                win_length=int(win_length),
                hop_length=int(hop_length),
                n_mels=int(n_mels),
                f_min=int(f_min),
                f_max=int(f_max),
                power=float(power),
            )
            amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")
            self.transform = torch.nn.Sequential(mel_transform, amplitude_to_db)

    def __call__(self, x):
        # 输入 x 是单通道 waveform，输出是 [F, T] 的时频特征。
        return self.transform(x)


class SirenDataset(torch.utils.data.Dataset):
    def __init__(self, h, mode="train", shuffle=False):
        super().__init__()
        self.h = h
        # mode 决定当前数据集对象服务于训练还是评估，不同模式下：
        # 1. 读取目录不同
        # 2. 是否开启增强不同
        # 3. 超长音频的裁剪策略不同
        self.mode = mode
        self.sr = int(h.sr)
        self.secs = float(h.secs)
        # 每条样本最终都被整理成固定长度，方便 DataLoader 拼 batch。
        self.target_num_samples = int(self.sr * self.secs)
        self.class2idx = h.siren_type

        # 通过配置切换使用 log-Mel 还是 MFCC。
        self.feature_type = str(get_required_attr(h, "feature_type")).lower()
        self.n_mfcc = int(get_required_attr(h, "n_mfcc"))
        self.feature_dim = self.n_mfcc if self.feature_type == "mfcc" else int(h.n_mels)
        self.transform = AudioFeatureExtractor(
            sample_rate=self.sr,
            feature_type=self.feature_type,
            n_fft=int(h.n_fft),
            n_mels=int(h.n_mels),
            n_mfcc=self.n_mfcc,
            win_length=int(h.win_len),
            hop_length=int(h.hop_len),
            f_min=int(get_required_attr(h, "f_min")),
            f_max=int(get_required_attr(h, "f_max")),
            power=float(h.power),
        )

        # 下面这些增强参数全部只在训练集里使用。
        # 验证集 / 测试集不能做随机增强，否则评估结果会漂移，不可复现。
        self.enable_augment = bool(get_required_attr(h, "train_augment"))
        self.road_noise_prob = float(get_required_attr(h, "road_noise_prob"))
        self.background_mix_prob = float(get_required_attr(h, "background_mix_prob"))
        self.road_noise_mode = str(get_required_attr(h, "road_noise_mode")).lower()
        self.real_noise_dir = str(get_required_attr(h, "real_noise_dir"))
        self.road_noise_snr_ranges = [
            (
                float(get_required_attr(h, "road_noise_near_snr_min")),
                float(get_required_attr(h, "road_noise_near_snr_max")),
                float(get_required_attr(h, "road_noise_near_weight")),
            ),
            (
                float(get_required_attr(h, "road_noise_mid_snr_min")),
                float(get_required_attr(h, "road_noise_mid_snr_max")),
                float(get_required_attr(h, "road_noise_mid_weight")),
            ),
            (
                float(get_required_attr(h, "road_noise_far_snr_min")),
                float(get_required_attr(h, "road_noise_far_snr_max")),
                float(get_required_attr(h, "road_noise_far_weight")),
            ),
        ]
        self.lowpass_filter_prob = float(get_required_attr(h, "lowpass_filter_prob"))
        self.lowpass_cutoff_min = float(get_required_attr(h, "lowpass_cutoff_min"))
        self.lowpass_cutoff_max = float(get_required_attr(h, "lowpass_cutoff_max"))
        self.lowpass_q = float(get_required_attr(h, "lowpass_q"))
        self.smooth_eq_prob = float(get_required_attr(h, "smooth_eq_prob"))
        self.smooth_eq_gain_db_min = float(get_required_attr(h, "smooth_eq_gain_db_min"))
        self.smooth_eq_gain_db_max = float(get_required_attr(h, "smooth_eq_gain_db_max"))
        self.smooth_eq_q_min = float(get_required_attr(h, "smooth_eq_q_min"))
        self.smooth_eq_q_max = float(get_required_attr(h, "smooth_eq_q_max"))
        self.smooth_eq_freq_ranges = [
            (
                float(get_required_attr(h, "smooth_eq_low_freq_min")),
                float(get_required_attr(h, "smooth_eq_low_freq_max")),
            ),
            (
                float(get_required_attr(h, "smooth_eq_mid_freq_min")),
                float(get_required_attr(h, "smooth_eq_mid_freq_max")),
            ),
            (
                float(get_required_attr(h, "smooth_eq_high_freq_min")),
                float(get_required_attr(h, "smooth_eq_high_freq_max")),
            ),
        ]
        self.doppler_prob = float(get_required_attr(h, "doppler_prob"))
        self.doppler_factor_min = float(get_required_attr(h, "doppler_factor_min"))
        self.doppler_factor_max = float(get_required_attr(h, "doppler_factor_max"))
        self.doppler_gain_min = float(get_required_attr(h, "doppler_gain_min"))
        self.doppler_gain_max = float(get_required_attr(h, "doppler_gain_max"))
        self.time_shift_prob = float(get_required_attr(h, "time_shift_prob"))
        self.time_shift_sec_max = float(get_required_attr(h, "time_shift_sec_max"))
        self.occlusion_prob = float(get_required_attr(h, "occlusion_prob"))
        self.occlusion_sec_min = float(get_required_attr(h, "occlusion_sec_min"))
        self.occlusion_sec_max = float(get_required_attr(h, "occlusion_sec_max"))
        self.occlusion_gain_min = float(get_required_attr(h, "occlusion_gain_min"))
        self.occlusion_gain_max = float(get_required_attr(h, "occlusion_gain_max"))
        self.reverb_prob = float(get_required_attr(h, "reverb_prob"))
        self.reverb_taps_min = int(get_required_attr(h, "reverb_taps_min"))
        self.reverb_taps_max = int(get_required_attr(h, "reverb_taps_max"))
        self.reverb_delay_ms_min = float(get_required_attr(h, "reverb_delay_ms_min"))
        self.reverb_delay_ms_max = float(get_required_attr(h, "reverb_delay_ms_max"))
        self.reverb_reflection_gain_min = float(get_required_attr(h, "reverb_reflection_gain_min"))
        self.reverb_reflection_gain_max = float(get_required_attr(h, "reverb_reflection_gain_max"))
        self.reverb_wet_min = float(get_required_attr(h, "reverb_wet_min"))
        self.reverb_wet_max = float(get_required_attr(h, "reverb_wet_max"))
        self.recording_chain_prob = float(get_required_attr(h, "recording_chain_prob"))
        self.recording_gain_min = float(get_required_attr(h, "recording_gain_min"))
        self.recording_gain_max = float(get_required_attr(h, "recording_gain_max"))
        self.recording_bandpass_prob = float(get_required_attr(h, "recording_bandpass_prob"))
        self.recording_highpass_min = float(get_required_attr(h, "recording_highpass_min"))
        self.recording_highpass_max = float(get_required_attr(h, "recording_highpass_max"))
        self.recording_lowpass_min = float(get_required_attr(h, "recording_lowpass_min"))
        self.recording_lowpass_max = float(get_required_attr(h, "recording_lowpass_max"))
        self.recording_quantize_prob = float(get_required_attr(h, "recording_quantize_prob"))
        self.recording_soft_clip_prob = float(get_required_attr(h, "recording_soft_clip_prob"))
        self.recording_soft_clip_drive_min = float(get_required_attr(h, "recording_soft_clip_drive_min"))
        self.recording_soft_clip_drive_max = float(get_required_attr(h, "recording_soft_clip_drive_max"))
        self.impulse_noise_prob = float(get_required_attr(h, "impulse_noise_prob"))
        self.quant_bits_min = int(get_required_attr(h, "quant_bits_min"))
        self.quant_bits_max = int(get_required_attr(h, "quant_bits_max"))
        self.impulse_count_min = int(get_required_attr(h, "impulse_count_min"))
        self.impulse_count_max = int(get_required_attr(h, "impulse_count_max"))
        self.impulse_amp_min = float(get_required_attr(h, "impulse_amp_min"))
        self.impulse_amp_max = float(get_required_attr(h, "impulse_amp_max"))
        self.impulse_width_ms_max = float(get_required_attr(h, "impulse_width_ms_max"))
        self.freq_mask_prob = float(get_required_attr(h, "freq_mask_prob"))
        self.freq_mask_width_max = int(get_required_attr(h, "freq_mask_width_max"))
        self._validate_augment_config()
        self.real_noise_files = self._collect_noise_files(self.real_noise_dir)

        # 主数据目录通常是标准类别子目录结构；
        # 额外新增数据可能是类别子目录，也可能是文件平铺目录，所以这里都做兼容。
        self.train_data_dir = str(get_required_attr(h, "train_data_dir"))
        self.test_data_dir = str(get_required_attr(h, "test_data_dir"))
        self.extra_data_dir = str(get_required_attr(h, "extra_data_dir"))
        # audio_indexes 里保存的是每条样本的元信息，不在初始化阶段直接把音频全部读进内存。
        self.audio_indexes = self._build_index()
        if shuffle:
            random.shuffle(self.audio_indexes)

    def _resolve_base_dir(self):
        # 根据当前模式决定“主目录”从哪里取：
        # 训练优先用 train_data_dir，评估优先用 test_data_dir。
        if self.mode == "train":
            candidates = [
                self.train_data_dir,
                str(get_required_attr(self.h, "data_dir")),
            ]
        else:
            candidates = [
                self.test_data_dir,
                os.path.join(str(get_required_attr(self.h, "data_dir")), "test_set"),
                str(get_required_attr(self.h, "data_dir")),
            ]

        for candidate in candidates:
            if candidate and os.path.isdir(candidate):
                return candidate

        searched = [candidate for candidate in candidates if candidate]
        raise FileNotFoundError(f"Cannot find a valid {self.mode} data directory. Checked: {searched}")

    def _make_record(self, file_name, class_name, file_path, source):
        # 每条样本统一抽象成一个 record，后面无论来自哪个目录，训练时都按同一种格式处理。
        return {
            "file_name": file_name,
            "class_name": class_name,
            "path": file_path,
            "source": source,
        }

    def _collect_class_dir_records(self, base_dir, source_name):
        # 读取形如：
        # base_dir/BKN/*.wav
        # base_dir/POL/*.wav
        # 这种“按类别分文件夹”的标准结构。
        records = []
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(base_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            # 目录名本身就是类别名时，直接用父目录名作为标签来源，最稳妥。
            for file_name in os.listdir(class_dir):
                if self._is_audio_file(file_name):
                    records.append(
                        self._make_record(
                            file_name=file_name,
                            class_name=class_name,
                            file_path=os.path.join(class_dir, file_name),
                            source=source_name,
                        )
                    )
        return records

    def _collect_flat_records(self, base_dir, source_name):
        # 读取平铺目录，例如：
        # base_dir/dynamic_xxx_AMB_001.wav
        # base_dir/动态_10km_正前_工程车_0175.wav
        # 这类目录没有类别子文件夹，所以必须靠文件名推断类别。
        records = []
        if not os.path.isdir(base_dir):
            return records

        # 平铺目录下只能依赖文件名规则来推断类别。
        for file_name in os.listdir(base_dir):
            file_path = os.path.join(base_dir, file_name)
            if os.path.isfile(file_path) and self._is_audio_file(file_name):
                records.append(
                    self._make_record(
                        file_name=file_name,
                        class_name=self._infer_class_name(file_name),
                        file_path=file_path,
                        source=source_name,
                    )
                )
        return records

    def _collect_mixed_records(self, base_dir, class_source_name, flat_source_name):
        # 兼容两种额外数据布局：
        # 1. 目录下直接按类别分文件夹存储
        # 2. 目录下平铺文件，类别通过文件名推断
        class_records = self._collect_class_dir_records(base_dir, class_source_name)
        flat_records = self._collect_flat_records(base_dir, flat_source_name)
        return self._merge_records(class_records, flat_records)

    @staticmethod
    def _merge_records(*record_groups):
        # 多个来源的样本合并时，按绝对路径去重，避免同一条音频被重复加入训练集。
        merged = []
        seen_paths = set()
        for records in record_groups:
            for record in records:
                norm_path = os.path.abspath(record["path"])
                if norm_path in seen_paths:
                    continue
                seen_paths.add(norm_path)
                merged.append(record)
        return merged

    def _build_index(self):
        # 这是整个 dataset 的核心之一：
        # 1. 先找到主数据目录
        # 2. 再把主目录、额外目录中的音频都扫描出来
        # 3. 最后拼成一个统一的样本索引列表
        base_dir = self._resolve_base_dir()

        if self.mode == "train":
            # 训练阶段合并主训练目录和额外训练目录。
            base_records = self._collect_class_dir_records(base_dir, "train_class_dir")
            extra_records = []
            for extra_dir_name in ("train", "train_add"):
                extra_train_dir = os.path.join(self.extra_data_dir, extra_dir_name)
                extra_records = self._merge_records(
                    extra_records,
                    self._collect_mixed_records(
                        extra_train_dir,
                        f"{extra_dir_name}_class_dir",
                        f"{extra_dir_name}_flat_dir",
                    ),
                )
            indexes = self._merge_records(base_records, extra_records)
        else:
            # 测试阶段优先支持 test/类别名/*.wav，也兼容 test/*.wav。
            class_dir_records = self._collect_class_dir_records(base_dir, "test_class_dir")
            flat_records = self._collect_flat_records(base_dir, "test_dir")
            extra_records = []
            for extra_dir_name in ("test", "test_add"):
                extra_test_dir = os.path.join(self.extra_data_dir, extra_dir_name)
                if os.path.isdir(extra_test_dir) and os.path.abspath(extra_test_dir) != os.path.abspath(base_dir):
                    extra_records = self._merge_records(
                        extra_records,
                        self._collect_mixed_records(
                            extra_test_dir,
                            f"{extra_dir_name}_class_dir",
                            f"{extra_dir_name}_flat_dir",
                        ),
                    )
            indexes = self._merge_records(class_dir_records, flat_records, extra_records)

        if not indexes:
            raise FileNotFoundError(f"No audio files found for mode={self.mode} under base_dir={base_dir}")
        return indexes

    @staticmethod
    def _is_audio_file(file_name):
        return file_name.endswith(AUDIO_EXTENSIONS)

    @staticmethod
    def _collect_noise_files(noise_dir):
        # 扫描真实背景噪声目录，收集可用于混入训练样本的噪声音频文件。
        if not noise_dir or not os.path.isdir(noise_dir):
            return []

        files = []
        for file_name in os.listdir(noise_dir):
            if file_name.endswith(AUDIO_EXTENSIONS):
                files.append(os.path.join(noise_dir, file_name))
        return sorted(files)

    @staticmethod
    def _infer_class_name(file_name):
        # 同时兼容英文命名和中文命名，例如：
        # dynamic_xxx_10m_AMB_xxx.wav
        # 动态_10km_正前_工程车_0175.wav
        stem = os.path.splitext(file_name)[0]
        normalized_stem = stem.upper()
        parts = [part for part in stem.replace("-", "_").replace(" ", "_").split("_") if part]

        # 先按分段精确匹配，避免把其他描述字段误识别成类别。
        for part in parts:
            alias = CLASS_NAME_ALIASES.get(part.upper())
            if alias is not None:
                return alias
            alias = CLASS_NAME_ALIASES.get(part)
            if alias is not None:
                return alias

        # 再对整个文件名做兜底扫描，兼容类别词嵌在更长片段中的情况。
        for alias_key, class_name in CLASS_NAME_ALIASES.items():
            if alias_key.isascii():
                if alias_key in normalized_stem:
                    return class_name
            else:
                if alias_key in stem:
                    return class_name

        raise ValueError(f"Cannot infer class name from file: {file_name}")

    def __len__(self):
        return len(self.audio_indexes)

    def wav2label(self, record):
        class_name = record["class_name"]
        class_label = self.class2idx[class_name]
        # BKN 视为无警笛，其余类别都视为有警笛。
        is_has_siren = 0 if class_label == 0 else 1
        return is_has_siren, class_label

    def _torchaudio_load_with_fallback(self, path):
        # 中文注释：优先走 torchaudio；如果当前环境解不了 mp3/m4a，再退回 librosa/audioread。
        detected_format = detect_audio_format(path)
        try:
            if detected_format is not None:
                return torchaudio.load(path, format=detected_format)
            return torchaudio.load(path)
        except Exception as exc:
            lower_path = str(path).lower()
            if detected_format != "mp3" and lower_path.endswith(".wav"):
                try:
                    return torchaudio.load(path, format="mp3")
                except Exception:
                    pass
            if lower_path.endswith((".mp3", ".m4a", ".aac", ".ogg")) or detected_format == "mp3":
                try:
                    waveform, sr = librosa.load(path, sr=None, mono=False)
                    waveform = np.asarray(waveform, dtype=np.float32)
                    if waveform.ndim == 1:
                        waveform = waveform[None, :]
                    return torch.from_numpy(waveform), sr
                except Exception:
                    pass
            raise RuntimeError(f"Failed to load audio: {path}\nError: {exc}") from exc

    def _load_waveform(self, path):
        # 真正取样本时才加载音频，这样初始化速度更快，也不会一次占用太多内存。
        waveform, sr = self._torchaudio_load_with_fallback(path)

        if waveform.size(0) > 1:
            # 当前模型只做单通道检测，多通道音频默认取第一个通道。
            waveform = waveform[:1]
        if sr != self.sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(waveform)
        waveform = waveform.squeeze(0)
        return waveform

    def _crop_or_pad(self, waveform):
        # 这个函数负责把任意长度的音频整理成固定时长：
        # 1. 空音频补零
        # 2. 太短就重复拼接
        # 3. 太长则裁成目标长度
        if waveform.numel() == 0:
            waveform = torch.zeros(self.target_num_samples, dtype=torch.float32)
        if waveform.numel() < self.target_num_samples:
            # 太短就重复拼接，保证每个样本长度统一到 secs 秒。
            repeat = self.target_num_samples // waveform.numel() + 1
            waveform = waveform.repeat(repeat)

        if self.mode == "train" and waveform.numel() > self.target_num_samples:
            # 训练时随机裁剪，等价于一种时序数据增强。
            max_audio_start = waveform.numel() - self.target_num_samples
            audio_start = random.randint(0, max_audio_start)
            waveform = waveform[audio_start:audio_start + self.target_num_samples]
        else:
            # 验证和测试时固定从头截断，保证评估结果稳定可复现。
            waveform = waveform[:self.target_num_samples]
        return waveform.contiguous()

    @staticmethod
    def _rms(x):
        # 计算均方根能量，后面做 SNR 控制时要用。
        return torch.sqrt(torch.mean(x * x) + 1e-8)

    def _prepare_noise_waveform(self, noise_waveform):
        # 把真实噪声音频裁剪或补齐到和训练样本一致的长度。
        if noise_waveform.numel() == 0:
            return torch.zeros(self.target_num_samples, dtype=torch.float32)

        if noise_waveform.numel() < self.target_num_samples:
            repeat = self.target_num_samples // noise_waveform.numel() + 1
            noise_waveform = noise_waveform.repeat(repeat)

        if noise_waveform.numel() > self.target_num_samples:
            max_start = noise_waveform.numel() - self.target_num_samples
            start = random.randint(0, max_start)
            noise_waveform = noise_waveform[start:start + self.target_num_samples]
        else:
            noise_waveform = noise_waveform[:self.target_num_samples]

        return noise_waveform.contiguous()

    def _load_noise_waveform(self, path):
        # 真实噪声也统一转成单通道，并重采样到训练采样率。
        waveform, sr = self._torchaudio_load_with_fallback(path)
        if waveform.size(0) > 1:
            waveform = waveform[:1]
        if sr != self.sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sr)(waveform)
        return waveform.squeeze(0)

    def _validate_augment_config(self):
        """集中检查增强概率和范围，避免错误配置运行到训练中途才暴露。"""
        probabilities = {
            "road_noise_prob": self.road_noise_prob,
            "background_mix_prob": self.background_mix_prob,
            "lowpass_filter_prob": self.lowpass_filter_prob,
            "smooth_eq_prob": self.smooth_eq_prob,
            "doppler_prob": self.doppler_prob,
            "time_shift_prob": self.time_shift_prob,
            "occlusion_prob": self.occlusion_prob,
            "reverb_prob": self.reverb_prob,
            "recording_chain_prob": self.recording_chain_prob,
            "recording_bandpass_prob": self.recording_bandpass_prob,
            "recording_quantize_prob": self.recording_quantize_prob,
            "recording_soft_clip_prob": self.recording_soft_clip_prob,
            "impulse_noise_prob": self.impulse_noise_prob,
            "freq_mask_prob": self.freq_mask_prob,
        }
        for name, probability in probabilities.items():
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {probability}")

        ranges = {
            "doppler_factor": (self.doppler_factor_min, self.doppler_factor_max),
            "doppler_gain": (self.doppler_gain_min, self.doppler_gain_max),
            "lowpass_cutoff": (self.lowpass_cutoff_min, self.lowpass_cutoff_max),
            "smooth_eq_gain_db": (self.smooth_eq_gain_db_min, self.smooth_eq_gain_db_max),
            "smooth_eq_q": (self.smooth_eq_q_min, self.smooth_eq_q_max),
            "occlusion_sec": (self.occlusion_sec_min, self.occlusion_sec_max),
            "occlusion_gain": (self.occlusion_gain_min, self.occlusion_gain_max),
            "reverb_delay_ms": (self.reverb_delay_ms_min, self.reverb_delay_ms_max),
            "reverb_reflection_gain": (
                self.reverb_reflection_gain_min,
                self.reverb_reflection_gain_max,
            ),
            "reverb_wet": (self.reverb_wet_min, self.reverb_wet_max),
            "recording_gain": (self.recording_gain_min, self.recording_gain_max),
            "recording_highpass": (self.recording_highpass_min, self.recording_highpass_max),
            "recording_lowpass": (self.recording_lowpass_min, self.recording_lowpass_max),
            "recording_soft_clip_drive": (
                self.recording_soft_clip_drive_min,
                self.recording_soft_clip_drive_max,
            ),
        }
        for name, (minimum, maximum) in ranges.items():
            if minimum > maximum:
                raise ValueError(f"{name} minimum must not exceed maximum: {minimum} > {maximum}")

        total_snr_weight = 0.0
        for snr_min, snr_max, weight in self.road_noise_snr_ranges:
            if snr_min > snr_max or weight < 0.0:
                raise ValueError(f"Invalid road-noise SNR range: {(snr_min, snr_max, weight)}")
            total_snr_weight += weight
        if total_snr_weight <= 0.0:
            raise ValueError("At least one road-noise SNR weight must be positive")

        if self.recording_lowpass_max >= self.sr / 2:
            raise ValueError("recording_lowpass_max must be lower than the Nyquist frequency")
        if self.lowpass_cutoff_max >= self.sr / 2:
            raise ValueError("lowpass_cutoff_max must be lower than the Nyquist frequency")
        if self.lowpass_cutoff_min <= 0.0 or self.lowpass_q <= 0.0:
            raise ValueError("Low-pass cutoff frequency and Q must be positive")
        if self.smooth_eq_q_min <= 0.0:
            raise ValueError("Smooth-EQ Q values must be positive")
        for freq_min, freq_max in self.smooth_eq_freq_ranges:
            if freq_min <= 0.0 or freq_min > freq_max:
                raise ValueError(f"Invalid Smooth-EQ frequency range: {(freq_min, freq_max)}")
            if freq_max >= self.sr / 2:
                raise ValueError("Smooth-EQ frequencies must be lower than the Nyquist frequency")
        if self.time_shift_sec_max < 0.0 or self.impulse_width_ms_max <= 0.0:
            raise ValueError("Time-shift and impulse-width limits must be positive")
        if not 0.0 <= self.occlusion_gain_min <= self.occlusion_gain_max <= 1.0:
            raise ValueError("Occlusion gains must stay in [0, 1]")
        if not 0.0 <= self.reverb_wet_min <= self.reverb_wet_max <= 1.0:
            raise ValueError("Reverb wet ratios must stay in [0, 1]")
        if self.reverb_taps_min < 1 or self.reverb_taps_min > self.reverb_taps_max:
            raise ValueError("Invalid reverb tap range")
        if self.freq_mask_width_max < 0 or self.freq_mask_width_max >= self.feature_dim:
            raise ValueError("freq_mask_width_max must be smaller than the feature dimension")

    def _sample_road_noise_snr(self):
        """按近、中、远三档权重选取道路场景，再在档位内均匀采样 SNR。"""
        selected_range = random.choices(
            self.road_noise_snr_ranges,
            weights=[item[2] for item in self.road_noise_snr_ranges],
            k=1,
        )[0]
        return random.uniform(selected_range[0], selected_range[1])

    def _add_synthetic_road_noise(self, waveform):
        # 合成道路噪声版本：低频隆隆声 + 宽带噪声 + 发动机谐波。
        num_samples = waveform.numel()
        t = torch.arange(num_samples, dtype=waveform.dtype) / float(self.sr)

        white = torch.randn(num_samples, dtype=waveform.dtype)
        rumble = torch.cumsum(torch.randn(num_samples, dtype=waveform.dtype), dim=0)
        rumble = rumble / (rumble.abs().max() + 1e-6)

        engine_base = random.uniform(35.0, 120.0)
        phase = random.uniform(0.0, 2.0 * math.pi)
        engine = (
            torch.sin(2.0 * math.pi * engine_base * t + phase)
            + 0.5 * torch.sin(2.0 * math.pi * engine_base * 2.0 * t + phase)
            + 0.25 * torch.sin(2.0 * math.pi * engine_base * 3.0 * t + phase)
        )

        envelope = 0.75 + 0.25 * torch.sin(2.0 * math.pi * random.uniform(0.2, 1.0) * t + phase)
        noise = (0.55 * white + 0.9 * rumble + 0.45 * engine) * envelope
        noise = noise / (self._rms(noise) + 1e-6)

        snr_db = self._sample_road_noise_snr()
        target_noise_rms = self._rms(waveform) / (10.0 ** (snr_db / 20.0))
        return waveform + noise * target_noise_rms

    def _add_real_road_noise(self, waveform):
        # 真实背景噪声版本：从真实噪声库中随机抽一段，再按随机 SNR 混入训练样本。
        if not self.real_noise_files:
            return self._add_synthetic_road_noise(waveform)

        noise_path = random.choice(self.real_noise_files)
        try:
            noise_waveform = self._load_noise_waveform(noise_path)
        except Exception:
            return self._add_synthetic_road_noise(waveform)

        noise_waveform = self._prepare_noise_waveform(noise_waveform).to(waveform.dtype)
        noise_rms = self._rms(noise_waveform)
        if noise_rms <= 1e-8:
            return waveform

        snr_db = self._sample_road_noise_snr()
        target_noise_rms = self._rms(waveform) / (10.0 ** (snr_db / 20.0))
        scaled_noise = noise_waveform / noise_rms * target_noise_rms
        return waveform + scaled_noise

    def _add_road_noise(self, waveform):
        # 通过配置切换道路噪声来源，方便做消融实验：
        # real: 真实背景噪声
        # synthetic: 合成道路噪声
        # off: 不加道路噪声
        if self.road_noise_mode == "off":
            return waveform
        if self.road_noise_mode == "synthetic":
            return self._add_synthetic_road_noise(waveform)
        return self._add_real_road_noise(waveform)

    def _apply_lowpass_filter(self, waveform):
        """随机衰减高频，保留警笛主要频段并模拟真实传播链路的高频损失。"""
        cutoff_freq = random.uniform(self.lowpass_cutoff_min, self.lowpass_cutoff_max)
        return torchaudio.functional.lowpass_biquad(
            waveform,
            sample_rate=self.sr,
            cutoff_freq=cutoff_freq,
            Q=self.lowpass_q,
        )

    def _apply_smooth_random_eq(self, waveform):
        """随机调整三个宽频段的增益，模拟传播路径和录音设备的平滑频响差异。"""
        augmented = waveform
        for freq_min, freq_max in self.smooth_eq_freq_ranges:
            center_freq = random.uniform(freq_min, freq_max)
            gain_db = random.uniform(self.smooth_eq_gain_db_min, self.smooth_eq_gain_db_max)
            q_value = random.uniform(self.smooth_eq_q_min, self.smooth_eq_q_max)
            augmented = torchaudio.functional.equalizer_biquad(
                augmented,
                sample_rate=self.sr,
                center_freq=center_freq,
                gain=gain_db,
                Q=q_value,
            )
        return augmented

    def _simulate_recording_chain(self, waveform):
        # 用平滑带通模拟车载麦克风和车窗频响，再叠加量化与 AGC 软削顶。
        augmented = waveform * random.uniform(self.recording_gain_min, self.recording_gain_max)

        if random.random() < self.recording_bandpass_prob:
            highpass_cutoff = random.uniform(self.recording_highpass_min, self.recording_highpass_max)
            lowpass_cutoff = random.uniform(self.recording_lowpass_min, self.recording_lowpass_max)
            augmented = torchaudio.functional.highpass_biquad(augmented, self.sr, highpass_cutoff)
            augmented = torchaudio.functional.lowpass_biquad(augmented, self.sr, lowpass_cutoff)

        if random.random() < self.recording_quantize_prob:
            quant_bits = random.randint(self.quant_bits_min, self.quant_bits_max)
            quant_scale = float(2 ** (quant_bits - 1))
            augmented = torch.round(augmented * quant_scale) / quant_scale

        if random.random() < self.recording_soft_clip_prob:
            drive = random.uniform(self.recording_soft_clip_drive_min, self.recording_soft_clip_drive_max)
            augmented = torch.tanh(augmented * drive) / math.tanh(drive)

        return augmented

    def _simulate_doppler_and_dynamics(self, waveform):
        # 整段连续伸缩比逐段拼接更平滑，用于模拟相对速度导致的轻微频率和周期变化。
        speed_factor = random.uniform(self.doppler_factor_min, self.doppler_factor_max)
        new_len = max(8, int(round(waveform.numel() / speed_factor)))
        augmented = F.interpolate(
            waveform.view(1, 1, -1),
            size=new_len,
            mode="linear",
            align_corners=False,
        ).view(-1)

        if augmented.numel() >= self.target_num_samples:
            max_start = augmented.numel() - self.target_num_samples
            start = random.randint(0, max_start)
            augmented = augmented[start:start + self.target_num_samples]
        else:
            # 伸缩后不足 2 秒时随机放置，后续道路噪声会自然填充弱警笛之外的区域。
            total_padding = self.target_num_samples - augmented.numel()
            left_padding = random.randint(0, total_padding)
            right_padding = total_padding - left_padding
            augmented = F.pad(augmented, (left_padding, right_padding))

        # 缓慢增益变化近似车辆接近或远离，避免使用突变包络破坏警笛周期。
        time_axis = torch.linspace(0.0, 1.0, steps=augmented.numel(), dtype=augmented.dtype)
        start_gain = random.uniform(self.doppler_gain_min, 1.0)
        end_gain = random.uniform(1.0, self.doppler_gain_max)
        if random.random() < 0.5:
            start_gain, end_gain = end_gain, start_gain
        gain_envelope = torch.lerp(
            torch.tensor(start_gain, dtype=augmented.dtype),
            torch.tensor(end_gain, dtype=augmented.dtype),
            time_axis,
        )
        return augmented * gain_envelope

    def _random_time_shift(self, waveform):
        """把警笛向前或向后平移，模拟目标进入或离开固定 2 秒窗口。"""
        max_shift = min(waveform.numel() - 1, int(round(self.time_shift_sec_max * self.sr)))
        if max_shift <= 0:
            return waveform

        shift = random.randint(-max_shift, max_shift)
        if shift == 0:
            return waveform

        shifted = torch.zeros_like(waveform)
        if shift > 0:
            shifted[shift:] = waveform[:-shift]
        else:
            shifted[:shift] = waveform[-shift:]
        return shifted

    def _simulate_occlusion(self, waveform):
        """对短时间区域做平滑衰减，模拟大车、护栏或车身造成的瞬时遮挡。"""
        duration = random.uniform(self.occlusion_sec_min, self.occlusion_sec_max)
        length = min(waveform.numel(), max(2, int(round(duration * self.sr))))
        start = random.randint(0, max(0, waveform.numel() - length))
        gain = random.uniform(self.occlusion_gain_min, self.occlusion_gain_max)

        # 正弦平方窗在两端为 0、中间为 1，可避免直接切断产生人造高频边缘。
        phase = torch.linspace(0.0, math.pi, steps=length, dtype=waveform.dtype)
        attenuation = 1.0 - (1.0 - gain) * torch.sin(phase).pow(2)
        augmented = waveform.clone()
        augmented[start:start + length] *= attenuation
        return augmented

    def _simulate_short_reverb(self, waveform):
        """使用稀疏多抽头脉冲响应模拟车厢和道路的短反射。"""
        tap_count = random.randint(self.reverb_taps_min, self.reverb_taps_max)
        min_delay = max(1, int(round(self.reverb_delay_ms_min * self.sr / 1000.0)))
        max_delay = max(min_delay, int(round(self.reverb_delay_ms_max * self.sr / 1000.0)))
        delays = sorted(random.randint(min_delay, max_delay) for _ in range(tap_count))

        impulse_response = torch.zeros(max_delay + 1, dtype=waveform.dtype)
        impulse_response[0] = 1.0
        for tap_index, delay in enumerate(delays):
            base_gain = random.uniform(
                self.reverb_reflection_gain_min,
                self.reverb_reflection_gain_max,
            )
            decay = 1.0 - tap_index / max(1, tap_count)
            impulse_response[delay] += base_gain * decay * random.choice([-1.0, 1.0])

        # conv1d 实现的是互相关，因此先翻转脉冲响应以得到因果卷积结果。
        reverberant = F.conv1d(
            waveform.view(1, 1, -1),
            impulse_response.flip(0).view(1, 1, -1),
            padding=max_delay,
        ).view(-1)[:waveform.numel()]
        wet = random.uniform(self.reverb_wet_min, self.reverb_wet_max)
        return waveform * (1.0 - wet) + reverberant * wet

    def _add_impulse_noise(self, waveform):
        # 加少量尖峰/短脉冲，模拟敲击、电流毛刺、接触不良带来的瞬态冲击噪声。
        augmented = waveform.clone()
        num_samples = augmented.numel()
        if num_samples <= 0:
            return augmented

        impulse_count = random.randint(self.impulse_count_min, self.impulse_count_max)
        for _ in range(impulse_count):
            center = random.randint(0, num_samples - 1)
            max_width = max(1, int(round(self.impulse_width_ms_max * self.sr / 1000.0)))
            width = random.randint(1, max_width)
            amplitude = random.uniform(self.impulse_amp_min, self.impulse_amp_max)
            amplitude *= random.choice([-1.0, 1.0])

            start = max(0, center - width // 2)
            end = min(num_samples, start + width)
            pulse_len = end - start
            if pulse_len <= 0:
                continue

            if random.random() < 0.5:
                pulse = torch.full((pulse_len,), amplitude, dtype=augmented.dtype)
            else:
                pulse = torch.linspace(amplitude, 0.0, steps=pulse_len, dtype=augmented.dtype)
            augmented[start:end] += pulse

        return augmented

    def _apply_frequency_mask(self, feature):
        """遮挡极少量 Mel 频带，防止模型只依赖某一条固定窄带谱线。"""
        if self.freq_mask_width_max <= 0 or feature.size(0) <= 1:
            return feature
        width = random.randint(1, min(self.freq_mask_width_max, feature.size(0) - 1))
        start = random.randint(0, feature.size(0) - width)
        masked = feature.clone()
        # 使用每个时间帧的频带均值填充，比直接写 0 更适合未经标准化的 log-Mel。
        neutral_value = feature.mean(dim=0, keepdim=True)
        masked[start:start + width] = neutral_value
        return masked

    def _apply_training_augment(self, waveform, has_siren):
        # 增强顺序对应真实声学链路：声源运动/遮挡 -> 道路背景 -> 传播反射 -> 录音设备。
        # 运动、时移和遮挡只用于警笛样本，避免对 BKN 做没有物理意义的变换。
        augmented = waveform.clone()

        if has_siren and random.random() < self.doppler_prob:
            augmented = self._simulate_doppler_and_dynamics(augmented)

        if has_siren and random.random() < self.time_shift_prob:
            augmented = self._random_time_shift(augmented)

        if has_siren and random.random() < self.occlusion_prob:
            augmented = self._simulate_occlusion(augmented)

        noise_probability = self.road_noise_prob if has_siren else self.background_mix_prob
        if random.random() < noise_probability:
            augmented = self._add_road_noise(augmented)

        # 道路混噪后独立执行低通，使模型适应远距离和隔窗录音中的高频衰减。
        if random.random() < self.lowpass_filter_prob:
            augmented = self._apply_lowpass_filter(augmented)

        # EQ 对所有类别使用，模拟整条录音链路的宽带频响差异，避免引入类别特有伪特征。
        if random.random() < self.smooth_eq_prob:
            augmented = self._apply_smooth_random_eq(augmented)

        if random.random() < self.reverb_prob:
            augmented = self._simulate_short_reverb(augmented)

        if random.random() < self.recording_chain_prob:
            augmented = self._simulate_recording_chain(augmented)

        if random.random() < self.impulse_noise_prob:
            augmented = self._add_impulse_noise(augmented)

        # 增强后重新裁剪/补齐，并把幅值约束回 [-1, 1] 附近。
        augmented = self._crop_or_pad(augmented)
        peak = augmented.abs().max()
        if peak > 1.0:
            augmented = augmented / peak
        return augmented.contiguous()

    def __getitem__(self, item):
        # DataLoader 每次取样时，完整流程如下：
        # 1. 根据索引找到这条样本的 record
        # 2. 读取原始 waveform
        # 3. 裁剪或补齐到固定时长
        # 4. 如果是训练模式，就做随机增强
        # 5. 生成标签
        # 6. 提取时频特征
        # 7. 返回 waveform、feature 和标签
        record = self.audio_indexes[item]
        file_path = record["path"]
        waveform = self._load_waveform(file_path)
        waveform = self._crop_or_pad(waveform)
        is_has_siren, class_label = self.wav2label(record)

        # 只对训练数据做增强，验证集和测试集必须保持原始分布。
        if self.mode == "train" and self.enable_augment:
            waveform = self._apply_training_augment(waveform, bool(is_has_siren))

        # 输出 shape: [1, F, T]
        # 这里额外保留一个通道维，是为了和原先模型的输入接口保持一致，
        # 后面的 model.py 会再把它整理成 Conv1d / GRU 所需的张量格式。
        feature = self.transform(waveform)
        if self.mode == "train" and self.enable_augment and random.random() < self.freq_mask_prob:
            feature = self._apply_frequency_mask(feature)
        feature = feature.unsqueeze(0)
        return waveform.float(), feature.float(), is_has_siren, class_label

    def get_class_counts(self):
        # 返回每个类别的样本数，通常在训练开始前用于构造类别权重或做数据分布检查。
        # 训练前统计每类样本量，供损失函数计算类别权重。
        counts = [0 for _ in range(len(CLASS_NAMES))]
        for record in self.audio_indexes:
            _, class_label = self.wav2label(record)
            counts[class_label] += 1
        return counts

    def get_source_summary(self):
        # 按数据来源和类别统计样本量，方便确认新增数据是否正确并入。
        summary = {}
        for record in self.audio_indexes:
            source = record["source"]
            class_name = record["class_name"]
            if source not in summary:
                summary[source] = {name: 0 for name in CLASS_NAMES}
                summary[source]["total"] = 0
            summary[source][class_name] += 1
            summary[source]["total"] += 1
        return summary


if __name__ == "__main__":
    import yaml

    from utils import AttrDict

    with open("config.yaml", "rb") as config_file:
        params = yaml.safe_load(config_file)
    # 直接运行 dataset.py 时，做一个最小自检：
    # 1. 能否成功构建索引
    # 2. 能否正常取出一条样本
    # 3. 返回张量形状是否符合预期
    dataset = SirenDataset(AttrDict(params), shuffle=True)
    print(f"dataset size: {len(dataset)}")
    for preview_idx, record in enumerate(dataset.audio_indexes[:3]):
        print(f"[{preview_idx}] {record}")
    waveform, feature, is_has_siren, class_label = dataset[0]
    print(waveform.shape, feature.shape, is_has_siren, class_label)
