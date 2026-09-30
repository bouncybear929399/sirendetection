import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# V1 仅增加训练 BKN 配对混合，保持数据集返回接口不变。
from dataset_V1 import SirenDataset
from model import SirenLiteGRUNetwork
from train import Trainer
from utils import AttrDict, get_logger, save_experiment_config, set_type


def setup_seed(seed):
    # 固定随机种子，尽量让多次训练结果可复现。
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(args, h):
    setup_seed(int(args.random_seed))

    cuda = bool(args.cuda)
    device_ids = args.device_ids
    # 优先使用配置里的 GPU；如果不可用则自动回退到 CPU。
    if not cuda or not device_ids or not torch.cuda.is_available():
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{device_ids[0]}")
        if len(device_ids) > 1:
            args.dp = True

    # 训练集启用随机打乱和随机裁剪；验证集保持固定顺序。
    train_dataset = SirenDataset(h, mode="train", shuffle=True)
    eval_dataset = SirenDataset(h, mode="test", shuffle=False)
    args.logger.info(
        "Dataset V1: training BKN pool=%s, conditional pair probability=%s",
        len(train_dataset.bkn_mix_paths),
        h.bkn_pair_mix_prob,
    )

    def log_dataset_summary(name, dataset):
        class_counts = dataset.get_class_counts()
        args.logger.info("%s class counts: %s", name, class_counts)
        source_summary = dataset.get_source_summary()
        for source_name, stats in source_summary.items():
            args.logger.info(
                "%s source=%s total=%s detail=%s",
                name,
                source_name,
                stats["total"],
                {cls_name: stats[cls_name] for cls_name in ["BKN", "POL", "FIR", "AMB", "ENG"]},
            )

    log_dataset_summary("train", train_dataset)
    log_dataset_summary("eval", eval_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.workers),
        pin_memory=cuda,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=cuda,
        drop_last=False,
    )

    # 构建轻量化 1DConv + GRU 网络。
    net = SirenLiteGRUNetwork(h)
    if args.dp:
        net = nn.DataParallel(net, device_ids=args.device_ids)
    net = net.to(args.device)

    # AdamW 负责参数更新；CosineAnnealingLR 负责后期平滑降学习率。
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(args.epochs),
        eta_min=float(args.lr) * 0.1,
    )
    # 统计每个类别的样本数，后续损失函数会据此做类别不平衡处理。
    class_counts = train_dataset.get_class_counts()
    trainer = Trainer(
        args=args,
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        class_counts=class_counts,
    )
    trainer.train(train_loader, eval_loader)


if __name__ == "__main__":
    # 先选择实验配置，再构造参数解析器，确保快照记录实际使用的文件。
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="config.yaml", help="Training YAML config path")
    config_args, _ = config_parser.parse_known_args()
    config_path = os.path.abspath(config_args.config)
    with open(config_path, "rb") as config_file:
        params = yaml.safe_load(config_file)

    # 把配置文件里的每个字段都映射成命令行参数，便于训练时临时覆盖。
    parser = argparse.ArgumentParser(
        description="Train SirenLite-GRU detection model.", parents=[config_parser]
    )
    for key, value in params.items():
        parser.add_argument(f"--{key}", default=value, type=set_type(value))
    args = parser.parse_args()

    h = AttrDict(params)
    # 命令行参数优先级高于配置文件，把覆盖后的值写回统一配置对象。
    for key, value in vars(args).items():
        setattr(h, key, value)

    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.version = f"SirenLiteGRU_{time_str}"
    log_dir = os.path.join("runs", args.version)
    os.makedirs(log_dir, exist_ok=True)
    # 把配置来源和解析后的最终参数挂到 args 上，供训练阶段继续复用。
    args.log_dir = log_dir
    args.source_config_path = config_path
    args.raw_config = dict(params)
    args.resolved_config = dict(h)
    args.logger = get_logger(os.path.join(log_dir, "running.log"))
    args.logger.info("Start training %s", args.version)
    save_experiment_config(
        save_dir=log_dir,
        source_config_path=args.source_config_path,
        raw_config=args.raw_config,
        resolved_config=args.resolved_config,
        args=args,
    )
    args.logger.info("Saved experiment config snapshot to %s", os.path.abspath(log_dir))

    run(args, h)
