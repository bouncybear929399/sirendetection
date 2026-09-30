import logging
import os
import shutil
from datetime import datetime

import yaml


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 让配置既可以 h["sr"] 访问，也可以 h.sr 访问。
        self.__dict__ = self


def set_type(value):
    # 根据配置默认值的类型，动态决定命令行参数该如何解析。
    if isinstance(value, bool):
        return lambda x: str(x).lower() in {"1", "true", "yes", "y"}
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    if isinstance(value, list):
        return lambda x: [int(item) for item in str(x).split(",") if item != ""]
    return str


def get_logger(filename):
    # 用文件名作为 logger 名称，避免不同训练任务之间的 handler 相互污染。
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    # 日志同时写文件和终端，方便训练中实时看，也方便训练后回溯。
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler = logging.FileHandler(filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _to_plain_data(value):
    # 递归把配置对象转换成基础 Python 类型，便于后续写入 YAML 和 JSON。
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_experiment_config(save_dir, source_config_path, raw_config, resolved_config, args=None):
    # 为每次实验保存一份完整的配置快照，方便后续复现实验和排查问题。
    os.makedirs(save_dir, exist_ok=True)

    raw_config_dict = _to_plain_data(dict(raw_config or {}))
    resolved_config_dict = _to_plain_data(dict(resolved_config or {}))
    cli_args_dict = _to_plain_data(vars(args)) if args is not None else {}

    overrides = {}
    for key, value in resolved_config_dict.items():
        if raw_config_dict.get(key) != value:
            overrides[key] = value

    record = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_config_path": os.path.abspath(source_config_path) if source_config_path else "",
        "raw_config": raw_config_dict,
        "resolved_config": resolved_config_dict,
        "cli_args": cli_args_dict,
        "overrides": overrides,
    }

    # 直接保存解析后的最终配置，后续重新训练时可以优先参考这份文件。
    with open(os.path.join(save_dir, "resolved_config.yaml"), "w", encoding="utf-8") as config_file:
        yaml.safe_dump(resolved_config_dict, config_file, allow_unicode=True, sort_keys=False)

    # 再保存一份带元信息的完整记录，便于追踪命令行覆盖项和原始配置来源。
    with open(os.path.join(save_dir, "experiment_record.yaml"), "w", encoding="utf-8") as record_file:
        yaml.safe_dump(record, record_file, allow_unicode=True, sort_keys=False)

    if source_config_path and os.path.isfile(source_config_path):
        # 额外备份训练启动时读取的原始配置文件，避免后续 config 被修改后无法回溯。
        shutil.copy2(source_config_path, os.path.join(save_dir, os.path.basename(source_config_path)))
