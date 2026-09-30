"""导出最高单类警笛概率阈值为 0.6、0.7、0.8 的 Android 模型。"""

import argparse
import os
from pathlib import Path

import torch


# 沿用现有导出脚本的默认路径，也可通过命令行指定其他模型和目录。
DEFAULT_INPUT = "./model/siren_detection_model(0913).pt"
DEFAULT_OUTPUT_DIR = "./postprocess_android/913/"


class SingleThresholdPostprocessCore(torch.nn.Module):
    """只检查最高警笛类别概率，未达标时将输出改为背景。"""

    def __init__(
        self,
        top1_threshold,
        reject_bkn_logit=8.0,
        reject_other_logit=-8.0,
    ):
        super().__init__()
        self.top1_threshold = float(top1_threshold)
        self.reject_bkn_logit = float(reject_bkn_logit)
        self.reject_other_logit = float(reject_other_logit)

    def forward(self, logits):
        logits = logits.to(torch.float32)
        probs = torch.softmax(logits, dim=1)
        # 第 0 类为背景，第 1～4 类为警笛；直接使用全类别 softmax 概率。
        siren_probs = probs[:, 1:5]
        top1 = torch.max(siren_probs, dim=1)[0]

        # 单类概率达到阈值即通过，等于阈值也通过；不检查总概率或类别差值。
        keep_siren = top1 >= self.top1_threshold

        # 通过时保留原始分数；拒绝时提高背景分数，压低所有其他类别分数。
        rejected_logits = torch.full_like(logits, self.reject_other_logit)
        rejected_logits[:, 0] = self.reject_bkn_logit
        return torch.where(keep_siren.unsqueeze(1), logits, rejected_logits)


class AndroidSingleThresholdPostprocessWrapper(torch.nn.Module):
    """兼容浮点音频和 int16 PCM 输入，并接入单阈值后处理。"""

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


# 默认导出三档；命令行可选择单独导出其中一档。
VARIANTS = {
    "0.6": {
        "top1_threshold": 0.60,
    },
    "0.7": {
        "top1_threshold": 0.70,
    },
    "0.8": {
        "top1_threshold": 0.80,
    },
}


def export_variant(input_path, output_path, thresholds):
    """将原始 TorchScript 模型和单阈值后处理一起导出。"""
    device = torch.device("cpu")
    base_model = torch.jit.load(str(input_path), map_location=device)
    base_model.eval()

    postprocess = SingleThresholdPostprocessCore(**thresholds)
    wrapper = AndroidSingleThresholdPostprocessWrapper(base_model, postprocess).to(device)
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
        # 文件名包含输入模型名称和阈值，便于区分三档及已有的多条件版本。
        output_path = output_dir / f"{input_path.stem}_single_threshold_{name}.pt"
        export_variant(input_path, output_path, thresholds)
        print(f"\nExported: {os.path.abspath(output_path)}")
        print("Thresholds:", thresholds)
        for input_dtype, output_dtype, output_shape in verify_model(output_path):
            print(f"  {input_dtype} -> {output_dtype}, shape={output_shape}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="导出最高警笛类别概率阈值为 0.6、0.7、0.8 的 Android 兼容 PT 模型。"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="原始 TorchScript 模型路径")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="模型输出目录")
    parser.add_argument(
        "--variant",
        default="all",
        choices=["all", *VARIANTS],
        help="all 导出三档，也可指定 0.6、0.7 或 0.8",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
