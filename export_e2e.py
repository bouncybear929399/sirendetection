import argparse
import os

import torch
import yaml
from torch.utils.mobile_optimizer import optimize_for_mobile

from model import EndToEndSirenLiteModel, SirenLiteGRUNetwork
from utils import AttrDict


DEFAULT_OUTPUT = "./model/siren_detection_model(0928).pt"
DEFAULT_CHECKPOINT = "/root/shh/siren/siren-detection-lite-gru/check_point/SirenLiteGRU_20260928_134143/checkpoint_best.pth.tar"


def load_config(config_path):
    # 读取训练时的配置，并把后续会参与 shape 计算的字段显式转成数值类型。
    with open(config_path, "rb") as config_file:
        config = AttrDict(yaml.safe_load(config_file))
    config.sr = int(config.sr)
    config.secs = float(config.secs)
    config.n_mels = int(config.n_mels)
    config.n_mfcc = int(getattr(config, "n_mfcc", config.n_mels))
    return config


def load_checkpoint(model, checkpoint_path, device):
    # 兼容普通单卡权重和 DataParallel 保存出来的权重。
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("clf_state_dict", checkpoint.get("state_dict", checkpoint))
    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def resolve_checkpoint_path(checkpoint_arg):
    # 如果显式传了 checkpoint，就优先使用；否则自动寻找最新 best checkpoint。
    if checkpoint_arg:
        if not os.path.isfile(checkpoint_arg):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_arg}")
        return checkpoint_arg

    checkpoint_root = os.path.join(os.getcwd(), "check_point")
    candidates = []
    if os.path.isdir(checkpoint_root):
        for root, _, files in os.walk(checkpoint_root):
            if "checkpoint_best.pth.tar" in files:
                path = os.path.join(root, "checkpoint_best.pth.tar")
                candidates.append((os.path.getmtime(path), path))

    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found. Pass --checkpoint explicitly or place checkpoint_best.pth.tar under ./check_point/"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def export_model(args):
    device = torch.device("cpu")
    h = load_config(args.config)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint)

    # 先恢复训练时的 base model 权重，再包一层端到端前处理模块。
    base_model = SirenLiteGRUNetwork(h).to(device)
    base_model = load_checkpoint(base_model, checkpoint_path, device)
    base_model.eval()

    e2e_model = EndToEndSirenLiteModel(base_model, h, input_sr=args.input_sr).to(device)
    e2e_model.eval()

    input_samples = int(args.input_sr * h.secs)
    example_input = torch.randn(1, 1, input_samples, dtype=torch.float32, device=device)

    with torch.no_grad():
        # trace 后得到 TorchScript，可直接给移动端或部署端加载。
        traced_model = torch.jit.trace(e2e_model, example_input)
        if args.mobile:
            traced_model = optimize_for_mobile(traced_model)
        traced_model.save(args.output)

    print(f"Checkpoint: {os.path.abspath(checkpoint_path)}")
    print(f"Exported PT: {os.path.abspath(args.output)}")
    print(f"Feature   : {getattr(h, 'feature_type', 'logmel')}")
    print(f"Input shape: [1, 1, {input_samples}] float32, {args.input_sr} Hz PCM")
    print("Output: 5-class logits, class order = BKN, POL, FIR, AMB, ENG")


def parse_args():
    parser = argparse.ArgumentParser(description="Export SirenLite-GRU model to TorchScript.")
    parser.add_argument("--config", default="config_bkn_pair_V1.yaml")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint path. Defaults to the fixed latest best checkpoint in this project.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--input_sr", type=int, default=48000)
    parser.add_argument("--mobile", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    export_model(parse_args())

