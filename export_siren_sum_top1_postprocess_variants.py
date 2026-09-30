"""导出先按警笛总概率过滤、通过后在四种警笛中取 top1 的 Android 模型。"""

import argparse
import os
from pathlib import Path

import torch


# 沿用现有导出脚本的默认路径，也可通过命令行指定其他模型和目录。
DEFAULT_INPUT = "./model/siren_detection_model(0913).pt"
DEFAULT_OUTPUT_DIR = "./postprocess_android/913/"


class SirenSumTop1PostprocessCore(torch.nn.Module):
    # 总概率达标后只在四种警笛中取 top1，不设单类概率或类别差值门槛。
    def __init__(
        self,
        siren_sum_threshold,
        reject_bkn_logit=8.0,
        reject_other_logit=-8.0,
    ):
        super().__init__()
        self.siren_sum_threshold = float(siren_sum_threshold)
        self.reject_bkn_logit = float(reject_bkn_logit)
        self.reject_other_logit = float(reject_other_logit)

    def forward(self, logits):
        logits = logits.to(torch.float32)
        probs = torch.softmax(logits, dim=1)
        # 先使用原始全类别概率计算总警笛概率，不能先屏蔽背景或重新归一化。
        siren_sum = torch.sum(probs[:, 1:5], dim=1)
        keep_siren = siren_sum >= self.siren_sum_threshold

        # 保留原来的 logits 输出格式；下游 argmax 读取最终类别。
        # 通过时只保留第 1～4 类分数，屏蔽背景，确保 top1 一定来自警笛类别。
        # 负无穷用于类别屏蔽；警笛分数及其相对顺序保持不变，并列时 argmax 取首项。
        accepted_logits = torch.full_like(logits, float('-inf'))
        accepted_logits[:, 1:5] = logits[:, 1:5]

        # 总概率不足时输出背景；通过分支再做 softmax 得到的是警笛内部归一化概率。
        rejected_logits = torch.full_like(logits, self.reject_other_logit)
        rejected_logits[:, 0] = self.reject_bkn_logit
        return torch.where(keep_siren.unsqueeze(1), accepted_logits, rejected_logits)


class AndroidSirenSumTop1PostprocessWrapper(torch.nn.Module):
    """兼容浮点音频和 int16 PCM 输入，并接入总概率过滤与警笛 top1 后处理。"""

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


# 仅按总警笛概率设置三档，达标后直接选警笛 top1；默认导出全部。
VARIANTS = {
    "loose": {
        "siren_sum_threshold": 0.75,
    },
    "recommended": {
        "siren_sum_threshold": 0.80,
    },
    "conservative": {
        "siren_sum_threshold": 0.85,
    },
}


def export_variant(input_path, output_path, thresholds):
    """将原始 TorchScript 模型和总概率过滤与警笛 top1 后处理一起导出。"""
    device = torch.device("cpu")
    base_model = torch.jit.load(str(input_path), map_location=device)
    base_model.eval()

    postprocess = SirenSumTop1PostprocessCore(**thresholds)
    wrapper = AndroidSirenSumTop1PostprocessWrapper(base_model, postprocess).to(device)
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
        # 文件名由输入模型名称、总概率过滤与 top1 标识、档位组成。
        output_path = output_dir / f"{input_path.stem}_siren_sum_top1_{name}.pt"
        export_variant(input_path, output_path, thresholds)
        print(f"\nExported: {os.path.abspath(output_path)}")
        print("Thresholds:", thresholds)
        for input_dtype, output_dtype, output_shape in verify_model(output_path):
            print(f"  {input_dtype} -> {output_dtype}, shape={output_shape}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="导出按警笛总概率过滤、达标后直接选择警笛 top1 的 Android 兼容 PT 模型。"
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
