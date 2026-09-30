"""导出警笛总概率与最高单类概率同时达标的双阈值 Android 模型。"""

import argparse
import os
from pathlib import Path

import torch


# 沿用现有导出脚本的默认路径，也可通过命令行指定其他模型和目录。
DEFAULT_INPUT = "./model/siren_detection_model(0913).pt"
DEFAULT_OUTPUT_DIR = "./postprocess_android/913/"


class DualThresholdPostprocessCore(torch.nn.Module):
    """同时检查警笛总概率与最高单类概率，任一未达标时输出背景。"""

    def __init__(
        self,
        siren_sum_threshold,
        top1_threshold,
        reject_bkn_logit=8.0,
        reject_other_logit=-8.0,
    ):
        super().__init__()
        self.siren_sum_threshold = float(siren_sum_threshold)
        self.top1_threshold = float(top1_threshold)
        self.reject_bkn_logit = float(reject_bkn_logit)
        self.reject_other_logit = float(reject_other_logit)

    def forward(self, logits):
        logits = logits.to(torch.float32)
        probs = torch.softmax(logits, dim=1)
        # 第 0 类为背景，第 1～4 类为警笛；直接使用全类别 softmax 概率。
        siren_probs = probs[:, 1:5]
        siren_sum = torch.sum(siren_probs, dim=1)
        top1 = torch.max(siren_probs, dim=1)[0]

        # 总概率判断整体是否像警笛，最高单类概率判断是否明确属于某一种警笛。
        # 两项必须同时达标，等于阈值也通过；不检查第一与第二类别的概率差。
        keep_siren = (
            (siren_sum >= self.siren_sum_threshold)
            & (top1 >= self.top1_threshold)
        )

        # 通过时保留原始分数；拒绝时提高背景分数，压低所有其他类别分数。
        rejected_logits = torch.full_like(logits, self.reject_other_logit)
        rejected_logits[:, 0] = self.reject_bkn_logit
        return torch.where(keep_siren.unsqueeze(1), logits, rejected_logits)


class AndroidDualThresholdPostprocessWrapper(torch.nn.Module):
    """兼容浮点音频和 int16 PCM 输入，并接入双阈值后处理。"""

    def __init__(self, base_model, postprocess):
        super().__init__()
        self.base_model = base_model
        self.postprocess = postprocess

    def forward(self, x):
        # int16 PCM 归一化到浮点范围；浮点输入仅转换数据类型。
        if x.dtype == torch.int16:
            x_model = x.to(torch.float32) / 32768.0
        else:
            x_model = x.to(torch.float32)
        logits = self.base_model(x_model)
        return self.postprocess(logits)


# 沿用当前三档的总概率与单类门槛；默认导出全部，可用命令行选择一档。
VARIANTS = {
    "loose": {
        "siren_sum_threshold": 0.75,
        "top1_threshold": 0.50,
    },
    "recommended": {
        "siren_sum_threshold": 0.80,
        "top1_threshold": 0.55,
    },
    "conservative": {
        "siren_sum_threshold": 0.85,
        "top1_threshold": 0.60,
    },
}


def export_variant(input_path, output_path, thresholds):
    """将原始 TorchScript 模型和双阈值后处理一起导出。"""
    device = torch.device("cpu")
    base_model = torch.jit.load(str(input_path), map_location=device)
    base_model.eval()

    postprocess = DualThresholdPostprocessCore(**thresholds)
    wrapper = AndroidDualThresholdPostprocessWrapper(base_model, postprocess).to(device)
    wrapper.eval()

    scripted = torch.jit.script(wrapper)
    scripted.save(str(output_path))
    return output_path


def verify_model(path):
    """加载导出文件，检查两种输入类型的前向推理。"""
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    results = []
    for dtype in (torch.float32, torch.int16):
        x = torch.zeros(1, 1, 96000, dtype=dtype)
        with torch.no_grad():
            y = model(x)
        results.append((str(dtype), str(y.dtype), tuple(y.shape)))
    return results


def main(args):
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input PT not found: {input_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = VARIANTS.keys() if args.variant == "all" else [args.variant]
    for name in selected:
        thresholds = VARIANTS[name]
        # 文件名包含输入模型名称、双阈值标识和档位，便于区分各个版本。
        output_path = output_dir / f"{input_path.stem}_dual_threshold_{name}.pt"
        export_variant(input_path, output_path, thresholds)
        print(f"\nExported: {os.path.abspath(output_path)}")
        print("Thresholds:", thresholds)
        for input_dtype, output_dtype, output_shape in verify_model(output_path):
            print(f"  {input_dtype} -> {output_dtype}, shape={output_shape}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="导出警笛总概率与最高单类概率同时达标的双阈值 Android 兼容 PT 模型。"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="原始 TorchScript 模型路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="模型输出目录")
    parser.add_argument(
        "--variant",
        default="all",
        choices=["all", *VARIANTS],
        help="all 导出三档，也可指定 loose、recommended 或 conservative",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
