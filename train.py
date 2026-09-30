import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils import save_experiment_config


plt.switch_backend("Agg")

CLASS_NAMES = ["BKN", "POL", "FIR", "AMB", "ENG"]


def save_checkpoint(state, filename):
    torch.save(state, filename)


class DynamicFocalLoss(nn.Module):
    def __init__(self, class_counts, initial_gamma=1.5, max_weight=5.0):
        super().__init__()
        self.class_counts = class_counts
        self.initial_gamma = initial_gamma
        self.max_weight = max_weight
        self.gamma = initial_gamma
        self.weights = None
        self.update_weights(epoch=0)

    def update_weights(self, epoch):
        # 当前版本使用固定类别权重，只保留 focal gamma 的静态设置。
        del epoch
        self.gamma = self.initial_gamma
        total = sum(self.class_counts)
        weights = [total / count if count > 0 else 1.0 for count in self.class_counts]
        weights = [min(weight, self.max_weight) for weight in weights]
        weight_sum = sum(weights)
        self.weights = torch.tensor(
            [weight * len(weights) / weight_sum for weight in weights],
            dtype=torch.float32,
        )

    def forward(self, inputs, targets):
        # 先计算逐样本交叉熵，再叠加 focal 项和类别权重。
        weights = self.weights.to(inputs.device)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        return (weights[targets] * (1 - pt) ** self.gamma * ce_loss).mean()


class Trainer(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = kwargs["args"]
        self.classifier = kwargs["net"].to(self.args.device)
        self.optimizer = kwargs["optimizer"]
        self.scheduler = kwargs["scheduler"]
        # 是否保存每个 epoch 的模型由配置文件统一控制，默认关闭以兼容旧配置。
        self.save_each_epoch = bool(getattr(self.args, "save_each_epoch", False))
        class_counts = kwargs.get("class_counts", [1, 1, 1, 1, 1])
        loss_name = str(getattr(self.args, "loss_name", "dynamic_focal")).lower()

        # 支持标准交叉熵和 focal loss 两种损失函数。
        if loss_name == "ce":
            weights = torch.tensor(class_counts, dtype=torch.float32)
            weights = torch.clamp(weights, min=1.0)
            weights = weights.sum() / weights
            weights = weights / weights.sum() * len(class_counts)
            self.criterion = nn.CrossEntropyLoss(weight=weights.to(self.args.device))
        else:
            self.criterion = DynamicFocalLoss(
                class_counts=class_counts,
                initial_gamma=float(getattr(self.args, "focal_initial_gamma", 1.5)),
                max_weight=float(getattr(self.args, "focal_max_weight", 5.0)),
            ).to(self.args.device)

        # 辅助任务只判断“是否存在警笛”：BKN 为 0，其余四类为 1。
        # 它复用主分类器已有的 5 类 logits，不引入新的模型参数或部署分支。
        self.binary_aux_loss_weight = float(getattr(self.args, "binary_aux_loss_weight", 0.0))
        if self.binary_aux_loss_weight < 0.0:
            raise ValueError("binary_aux_loss_weight must be non-negative")

    @staticmethod
    def _binary_siren_logit(logits):
        """把 5 类 logits 转换为“警笛 vs BKN”的单个二分类 logit。"""
        # 取四类警笛中的最大 logit，与推理阶段 5 类 argmax 的判定规则保持一致：
        # 只有至少一个警笛类别得分高于 BKN，当前窗口才会被判为警笛。
        siren_logit = torch.max(logits[:, 1:], dim=1).values
        background_logit = logits[:, 0]
        return siren_logit - background_logit

    def compute_loss(self, logits, class_label):
        """计算五分类主损失和可选的警笛存在性辅助损失。"""
        primary_loss = self.criterion(logits, class_label)
        if self.binary_aux_loss_weight <= 0.0:
            return primary_loss

        binary_logit = self._binary_siren_logit(logits)
        binary_target = (class_label != 0).float()
        auxiliary_loss = F.binary_cross_entropy_with_logits(binary_logit, binary_target)
        return primary_loss + self.binary_aux_loss_weight * auxiliary_loss

    def train(self, train_loader, eval_loader):
        version_dir = os.path.join(self.args.model_dir, self.args.version)
        os.makedirs(version_dir, exist_ok=True)
        epoch_checkpoint_dir = os.path.join(version_dir, "epoch_checkpoints")
        if self.save_each_epoch:
            os.makedirs(epoch_checkpoint_dir, exist_ok=True)
        # 在模型权重目录里也保存一份实验配置，方便后续拿到 checkpoint 时直接查看对应参数。
        save_experiment_config(
            save_dir=version_dir,
            source_config_path=getattr(self.args, "source_config_path", ""),
            raw_config=getattr(self.args, "raw_config", {}),
            resolved_config=getattr(self.args, "resolved_config", {}),
            args=self.args,
        )

        best_acc = 0.0
        history = {
            "train_loss": [],
            "eval_loss": [],
            "eval_acc": [],
            "class_acc": {cls: [] for cls in CLASS_NAMES},
        }

        for epoch in range(self.args.epochs):
            self.classifier.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, total=len(train_loader), ncols=100)

            for waveform, log_mel, _, class_label in pbar:
                # 数据集同时返回原始波形和 log-Mel，训练时统一搬到目标设备上。
                waveform = waveform.float().to(self.args.device)
                log_mel = log_mel.float().to(self.args.device)
                class_label = class_label.long().view(-1).to(self.args.device)

                # 前向得到类别 logits，再计算损失。
                logits, _, _, _ = self.classifier(waveform, log_mel)
                # 主五分类损失之外，辅助损失专门强化 BKN/警笛边界。
                loss = self.compute_loss(logits, class_label)

                # 标准训练三步：清梯度 -> 反向传播 -> 参数更新。
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                pbar.set_description(f"Epoch {epoch + 1}/{self.args.epochs} loss={loss.item():.4f}")

            if self.scheduler is not None and epoch >= 20:
                self.scheduler.step()
            if isinstance(self.criterion, DynamicFocalLoss):
                self.criterion.update_weights(epoch)

            avg_train_loss = train_loss / max(1, len(train_loader))
            history["train_loss"].append(avg_train_loss)

            avg_eval_loss, eval_acc, class_acc, wrong_samples = self.evaluate(eval_loader, epoch)
            history["eval_loss"].append(avg_eval_loss)
            history["eval_acc"].append(eval_acc)
            for cls_name, cls_value in class_acc.items():
                history["class_acc"][cls_name].append(cls_value)

            # 每轮验证完成后保存一次完整状态。文件名使用从 1 开始的 epoch 编号，
            # 例如 checkpoint_epoch_001.pth.tar，方便按文件名直接排序和对比。
            if self.save_each_epoch:
                epoch_checkpoint_path = os.path.join(
                    epoch_checkpoint_dir,
                    f"checkpoint_epoch_{epoch + 1:03d}.pth.tar",
                )
                save_checkpoint(
                    {
                        # 保持与原 checkpoint 的约定一致，epoch 在文件内部仍使用从 0 开始的索引。
                        "epoch": epoch,
                        "clf_state_dict": self.classifier.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                        "best_acc": max(best_acc, eval_acc),
                        "train_loss": avg_train_loss,
                        "eval_loss": avg_eval_loss,
                        "eval_acc": eval_acc,
                        "class_acc": class_acc,
                    },
                    epoch_checkpoint_path,
                )
                if getattr(self.args, "logger", None) is not None:
                    self.args.logger.info(
                        "Saved epoch checkpoint: epoch=%s path=%s",
                        epoch + 1,
                        os.path.abspath(epoch_checkpoint_path),
                    )

            if eval_acc >= best_acc:
                best_acc = eval_acc
                # 只保存当前最佳模型，便于后续导出和部署。
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "clf_state_dict": self.classifier.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_acc": best_acc,
                    },
                    os.path.join(version_dir, "checkpoint_best.pth.tar"),
                )
                # 最佳模型刷新时，把当前验证集分错的文件名和路径也保存下来。
                self.save_best_wrong_samples(version_dir, epoch, eval_acc, wrong_samples)
                if getattr(self.args, "logger", None) is not None:
                    self.args.logger.info(
                        "Saved best validation wrong samples: epoch=%s, wrong_count=%s",
                        epoch + 1,
                        len(wrong_samples),
                    )

            if (epoch + 1) % 10 == 0 or (epoch + 1) == self.args.epochs:
                self.plot_metrics(history, version_dir)

            print(
                f"Epoch {epoch + 1}: train_loss={avg_train_loss:.4f}, "
                f"eval_loss={avg_eval_loss:.4f}, eval_acc={eval_acc:.4f}"
            )

        self.save_performance_stats(history, version_dir)
        print(f"Training completed. Best accuracy: {best_acc:.4f}")

    def evaluate(self, eval_loader, epoch):
        self.classifier.eval()
        eval_loss = 0.0
        total = 0
        correct = 0
        class_total = {cls_name: 0 for cls_name in CLASS_NAMES}
        wrong_count = {cls_name: 0 for cls_name in CLASS_NAMES}
        wrong_samples = []
        dataset_records = getattr(eval_loader.dataset, "audio_indexes", [])
        sample_offset = 0

        pbar = tqdm(eval_loader, total=len(eval_loader), ncols=100)
        pbar.set_description(f"Epoch {epoch + 1} Evaluating")
        with torch.no_grad():
            for waveform, log_mel, _, class_label in pbar:
                # 验证阶段不更新参数，只统计损失和分类结果。
                waveform = waveform.float().to(self.args.device)
                log_mel = log_mel.float().to(self.args.device)
                class_label = class_label.long().view(-1).to(self.args.device)

                logits, _, _, _ = self.classifier(waveform, log_mel)
                # 验证阶段沿用训练时的总损失，保证 loss 曲线含义一致。
                loss = self.compute_loss(logits, class_label)
                eval_loss += loss.item()

                pred = torch.argmax(logits, dim=1)
                correct += (pred == class_label).sum().item()
                total += class_label.size(0)

                true_labels = class_label.cpu().tolist()
                pred_labels = pred.cpu().tolist()
                for true_lbl, pred_lbl in zip(true_labels, pred_labels):
                    class_name = CLASS_NAMES[true_lbl]
                    class_total[class_name] += 1
                    if true_lbl != pred_lbl:
                        wrong_count[class_name] += 1

                # 验证集固定不打乱，因此可以按顺序回查当前 batch 对应的文件名和路径。
                batch_size_actual = len(true_labels)
                batch_records = dataset_records[sample_offset:sample_offset + batch_size_actual]
                for batch_index, (true_lbl, pred_lbl) in enumerate(zip(true_labels, pred_labels)):
                    if true_lbl == pred_lbl:
                        continue
                    record = batch_records[batch_index] if batch_index < len(batch_records) else {}
                    wrong_samples.append(
                        {
                            "dataset_index": sample_offset + batch_index,
                            "file_name": record.get("file_name", ""),
                            "path": record.get("path", ""),
                            "source": record.get("source", ""),
                            "class_name": record.get("class_name", ""),
                            "true_label": CLASS_NAMES[true_lbl],
                            "pred_label": CLASS_NAMES[pred_lbl],
                        }
                    )
                sample_offset += batch_size_actual

        avg_eval_loss = eval_loss / max(1, len(eval_loader))
        eval_acc = correct / max(1, total)
        class_acc = {
            cls_name: (class_total[cls_name] - wrong_count[cls_name]) / max(1, class_total[cls_name])
            for cls_name in CLASS_NAMES
        }
        print(f"\n--- Epoch {epoch + 1} error summary ---")
        for cls_name in CLASS_NAMES:
            print(f"{cls_name}: wrong {wrong_count[cls_name]}/{class_total[cls_name]} | acc {class_acc[cls_name]:.4f}")
        return avg_eval_loss, eval_acc, class_acc, wrong_samples

    @staticmethod
    def save_best_wrong_samples(save_dir, epoch, eval_acc, wrong_samples):
        # 同时保存 JSON 和 TXT 两份，便于后续脚本分析和人工排查。
        json_path = os.path.join(save_dir, "best_eval_wrong_samples.json")
        txt_path = os.path.join(save_dir, "best_eval_wrong_samples.txt")

        payload = {
            "best_epoch": int(epoch) + 1,
            "best_eval_acc": float(eval_acc),
            "wrong_count": len(wrong_samples),
            "wrong_samples": wrong_samples,
        }
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(payload, json_file, indent=2, ensure_ascii=False)

        with open(txt_path, "w", encoding="utf-8") as txt_file:
            txt_file.write("========================================\n")
            txt_file.write(" Best Eval Wrong Samples\n")
            txt_file.write("========================================\n")
            txt_file.write(f"Best epoch: {int(epoch) + 1}\n")
            txt_file.write(f"Best eval acc: {float(eval_acc):.6f}\n")
            txt_file.write(f"Wrong sample count: {len(wrong_samples)}\n\n")
            for item in wrong_samples:
                txt_file.write(
                    f"[{item['dataset_index']}] true={item['true_label']} pred={item['pred_label']} "
                    f"class={item['class_name']} file={item['file_name']} "
                    f"source={item['source']} path={item['path']}\n"
                )

    def plot_metrics(self, history, save_dir):
        epochs = range(1, len(history["train_loss"]) + 1)

        # 保存整体 loss 曲线。
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history["train_loss"], label="Train Loss")
        plt.plot(epochs, history["eval_loss"], label="Eval Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("SirenLite-GRU Loss Curve")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=300)
        plt.close()

        # 保存总体准确率和各类别准确率曲线。
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, history["eval_acc"], label="Overall Accuracy", linewidth=2)
        for cls_name, cls_values in history["class_acc"].items():
            plt.plot(epochs, cls_values, label=f"{cls_name} Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("SirenLite-GRU Accuracy Curve")
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=300)
        plt.close()

    def save_performance_stats(self, history, save_dir):
        stats_path = os.path.join(save_dir, "performance_stats.json")
        summary_path = os.path.join(save_dir, "performance_summary.txt")

        # 同时保存完整训练历史和一个便于查看的文本摘要。
        with open(stats_path, "w", encoding="utf-8") as json_file:
            json.dump(history, json_file, indent=4, ensure_ascii=False)

        with open(summary_path, "w", encoding="utf-8") as summary_file:
            summary_file.write("========================================\n")
            summary_file.write("      SirenLite-GRU Training Summary\n")
            summary_file.write("========================================\n")
            summary_file.write(f"Total epochs: {len(history['train_loss'])}\n")
            summary_file.write(f"Best eval acc: {max(history['eval_acc']):.4f}\n")
            summary_file.write(f"Final eval acc: {history['eval_acc'][-1]:.4f}\n\n")
            summary_file.write("Per-class accuracy:\n")
            for cls_name, cls_values in history["class_acc"].items():
                if cls_values:
                    summary_file.write(
                        f"{cls_name}: final={cls_values[-1]:.4f}, best={max(cls_values):.4f}, "
                        f"best_epoch={int(np.argmax(cls_values)) + 1}\n"
                    )
