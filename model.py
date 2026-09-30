import torch
from torch import nn
import torchaudio


class EndToEndSirenLiteModel(nn.Module):
    def __init__(self, base_model, h, input_sr=48000):
        super().__init__()
        # 直接复用外部传入的训练好主干网络，不在这里重新实例化。
        self.base_model = base_model
        # 导出后的端到端模型直接接收原始 PCM，因此内部需要自带重采样和特征提取。
        self.resample = torchaudio.transforms.Resample(orig_freq=input_sr, new_freq=int(h.sr))
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(h.sr),
            n_fft=int(h.n_fft),
            win_length=int(h.win_len),
            hop_length=int(h.hop_len),
            n_mels=int(h.n_mels),
            f_min=int(h.f_min),
            f_max=int(h.f_max),
            power=float(h.power),
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def forward(self, x_pcm_float):
        # 支持 [B, 1, N] 和 [B, N] 两种输入形式，方便不同推理端接入。
        if x_pcm_float.dim() == 3:
            x_pcm_float = x_pcm_float.squeeze(1)

        # 导出模型内部自己完成：48k PCM -> 16k waveform -> log-Mel -> 分类 logits。
        wav_16k = self.resample(x_pcm_float)
        log_mel = self.amplitude_to_db(self.mel_transform(wav_16k)).unsqueeze(1)
        logits, _, _, _ = self.base_model(wav_16k, log_mel)
        return logits


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        # 深度可分离卷积块：先做逐通道时序卷积，再用 1x1 卷积融合通道信息。
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # 残差连接有助于稳定训练，也能减少浅层信息丢失。
        return x + self.block(x)


class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        # 对每个时间帧学习一个权重，再做加权求和，避免简单平均稀释关键警笛片段。
        attn_score = torch.softmax(self.attn(x).squeeze(-1), dim=1)
        pooled = torch.sum(x * attn_score.unsqueeze(-1), dim=1)
        return pooled, attn_score


class SirenLiteGRUNetwork(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        input_dim = int(h.n_mels)
        channels = int(getattr(h, "conv_channels", 32))
        hidden_size = int(getattr(h, "gru_hidden", 48))
        gru_layers = int(getattr(h, "gru_layers", 2))
        dropout = float(getattr(h, "dropout", 0.1))
        num_blocks = int(getattr(h, "num_blocks", 2))

        # 先沿 Mel 维度做归一化，减小不同频带数值范围的差异。
        self.feature_norm = nn.LayerNorm(input_dim)
        self.stem = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels),
            nn.SiLU(inplace=True),
        )
        # 多层轻量残差块继续提取局部时序模式，如警笛的周期性起伏。
        self.blocks = nn.Sequential(
            *[
                DepthwiseResidualBlock(
                    channels=channels,
                    kernel_size=5,
                    dilation=2 ** block_idx,
                    dropout=dropout,
                )
                for block_idx in range(num_blocks)
            ]
        )
        # GRU 负责建模更长的时间依赖，比纯卷积更适合警笛的连续调制模式。
        self.gru = nn.GRU(
            input_size=channels,
            hidden_size=hidden_size,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.pool = TemporalAttentionPooling(hidden_size)
        self.frame_head = nn.Linear(hidden_size, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, int(h.num_class)),
        )

    def forward(self, x_wav, x_mel):
        # 当前版本保留 waveform 输入只是为了兼容旧训练接口，实际只使用 log-Mel 特征。
        del x_wav
        if x_mel.dim() == 4:
            # [B, 1, Mel, T] -> [B, Mel, T]
            x_mel = x_mel.squeeze(1)
        # 先转成 [B, T, Mel]，让 LayerNorm 在最后一维 Mel 上工作。
        x_mel = x_mel.transpose(1, 2)
        x_mel = self.feature_norm(x_mel)
        # Conv1d 需要 [B, C, L]，这里把 Mel 频带视作通道，只沿时间轴卷积。
        x = x_mel.transpose(1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        # GRU 需要 [B, T, C]。
        x = x.transpose(1, 2)

        sequence, _ = self.gru(x)
        pooled, frame_attention = self.pool(sequence)
        logits = self.classifier(pooled)
        # 额外保留帧级输出，后续如果要做定位或多任务训练会比较方便。
        frame_logits = self.frame_head(sequence).squeeze(-1)
        return logits, pooled, frame_logits, frame_attention


if __name__ == "__main__":
    import yaml

    from utils import AttrDict

    with open("config.yaml", "rb") as config_file:
        params = yaml.safe_load(config_file)
    h = AttrDict(params)

    model = SirenLiteGRUNetwork(h)
    e2e = EndToEndSirenLiteModel(model, h)
    dummy_wav = torch.randn(1, int(h.sr * h.secs))
    total_samples = int(h.sr * h.secs)
    num_frames = 1 + total_samples // int(h.hop_len)
    dummy_mel = torch.randn(1, 1, int(h.n_mels), num_frames)
    output = model(dummy_wav, dummy_mel)
    dummy_pcm = torch.randn(1, 1, 96000)
    logits = e2e(dummy_pcm)
    print(output[0].shape, output[1].shape, output[2].shape, output[3].shape)
    print(logits.shape)
