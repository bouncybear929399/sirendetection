import argparse
import os
from pathlib import Path

import torch


DEFAULT_INPUT = "./model/siren_detection_model(0913).pt"
DEFAULT_OUTPUT_DIR = "./postprocess_android/913/"


class ThresholdPostprocessCore(torch.nn.Module):
    # 三项条件同时达标才保留预测，任一不达标则输出背景。
    def __init__(
        self,
        siren_sum_threshold,
        top1_threshold,
        margin_threshold,
        reject_bkn_logit=8.0,
        reject_other_logit=-8.0,
    ):
        super().__init__()
        self.siren_sum_threshold = float(siren_sum_threshold)
        self.top1_threshold = float(top1_threshold)
        self.margin_threshold = float(margin_threshold)
        self.reject_bkn_logit = float(reject_bkn_logit)
        self.reject_other_logit = float(reject_other_logit)

    def forward(self, logits):
        logits = logits.to(torch.float32)
        probs = torch.softmax(logits, dim=1)
        # 第 0 类为背景，第 1～4 类为警笛；不对警笛概率重新归一化。
        siren_probs = probs[:, 1:5]
        siren_sum = torch.sum(siren_probs, dim=1)
        # 排名仅包含四种警笛，背景不参与第一、第二名的比较。
        top2_probs, _ = torch.topk(siren_probs, k=2, dim=1)
        top1 = top2_probs[:, 0]
        top2 = top2_probs[:, 1]
        margin = top1 - top2

        # 总概率衡量整体警笛置信度，单类概率衡量最可能警笛的置信度。
        # 概率差要求第一候选领先第二候选；三项均包含恰好等于阈值的情况。
        keep_siren = (
            (siren_sum >= self.siren_sum_threshold)
            & (top1 >= self.top1_threshold)
            & (margin >= self.margin_threshold)
        )

        # 通过时保留模型分数；拒绝时将背景分数置高，其余类别分数置低。
        rejected_logits = torch.full_like(logits, self.reject_other_logit)
        rejected_logits[:, 0] = self.reject_bkn_logit
        return torch.where(keep_siren.unsqueeze(1), logits, rejected_logits)


class AndroidThresholdPostprocessWrapper(torch.nn.Module):
    def __init__(self, base_model, postprocess):
        super().__init__()
        self.base_model = base_model
        self.postprocess = postprocess

    def forward(self, x):
        if x.dtype == torch.int16:
            x_model = x.to(torch.float32) / 32768.0
        else:
            x_model = x.to(torch.float32)
        logits = self.base_model(x_model)
        return self.postprocess(logits)


VARIANTS = {
    "loose": {
        "siren_sum_threshold": 0.75,
        "top1_threshold": 0.50,
        "margin_threshold": 0.25,
    },
    "recommended": {
        "siren_sum_threshold": 0.80,
        "top1_threshold": 0.55,
        "margin_threshold": 0.35,
    },
    "conservative": {
        "siren_sum_threshold": 0.85,
        "top1_threshold": 0.60,
        "margin_threshold": 0.45,
    },
}


def export_variant(input_path, output_path, thresholds):
    device = torch.device("cpu")
    base_model = torch.jit.load(str(input_path), map_location=device)
    base_model.eval()

    postprocess = ThresholdPostprocessCore(**thresholds)
    wrapper = AndroidThresholdPostprocessWrapper(base_model, postprocess).to(device)
    wrapper.eval()

    scripted = torch.jit.script(wrapper)
    scripted.save(str(output_path))
    return output_path


def verify_model(path):
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
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PT not found: {input_path}")

    selected = VARIANTS.keys() if args.variant == "all" else [args.variant]
    for name in selected:
        thresholds = VARIANTS[name]
        output_path = output_dir / f"{input_path.stem}_triple_threshold_{name}.pt"
        export_variant(input_path, output_path, thresholds)
        print(f"\nExported: {os.path.abspath(output_path)}")
        print("Thresholds:", thresholds)
        for input_dtype, output_dtype, output_shape in verify_model(output_path):
            print(f"  {input_dtype} -> {output_dtype}, shape={output_shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export threshold-postprocessed Android-compatible PT variants.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variant", default="all", choices=["all", "loose", "recommended", "conservative"])
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
