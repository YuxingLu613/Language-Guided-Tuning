from __future__ import annotations

"""AdaptiveTrainer
===================
A unified trainer that delegates *training-strategy* decisions (loss function,
optimizer, learning-rate, etc.) to an LLM-powered `StrategyAgent` **between epochs**.

The class is designed to be **task-agnostic**:
- `task_type`="classification" → accuracy/precision/recall metrics & nn.CrossEntropyLoss
- `task_type`="regression" → mse/mae metrics & nn.MSELoss

It can be extended to segmentation, sequence modelling, etc. by adding new
`_get_loss_fn`, `_calculate_metrics` branches.

Example usage for MNIST classification:
>>> model = SimpleCNN(...)
>>> trainer = AdaptiveTrainer(model, "classification", "config/config.yaml")
>>> trainer.train(train_loader, val_loader)

For housing price regression:
>>> model = CustomModel(...)
>>> trainer = AdaptiveTrainer(model, "regression", "config/housing_config.yaml")

NOTE: This trainer intentionally *does not* integrate with the existing Advisor/
Judger flow to keep the example concise. It focuses on the dynamic-strategy
aspect requested by the user.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import yaml
import json

from ..llm.strategy_agent import StrategyAgent
from ..utils.metrics_logger import MetricsLogger
from ..utils.regression_metrics_logger import RegressionMetricsLogger
from ..llm.judger import LLMJudger
from ..llm.prompt_optimizer import LLMPromptOptimizer


class AdaptiveTrainer:
    SUPPORTED_OPTIMIZERS = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adamw": torch.optim.AdamW,
        "adagrad": torch.optim.Adagrad,
        "nadam": torch.optim.NAdam,
    }

    class _FocalLoss(nn.Module):
        """Simple focal loss implementation for multi-class or binary classification."""
        def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
            super().__init__()
            self.gamma = gamma
            self.weight = weight
            self.reduction = reduction

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
            ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.gamma) * ce_loss
            if self.reduction == "mean":
                return focal_loss.mean()
            elif self.reduction == "sum":
                return focal_loss.sum()
            else:
                return focal_loss

    class _DiceLoss(nn.Module):
        """Dice loss for (multi-)class classification."""

        def __init__(self, smooth: float = 1.0):
            super().__init__()
            self.smooth = smooth

        def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
            # one-hot encode targets
            num_classes = inputs.size(1)
            targets_onehot = torch.nn.functional.one_hot(targets, num_classes).float()
            inputs_prob = torch.softmax(inputs, dim=1)

            dims = (0,)
            intersection = (inputs_prob * targets_onehot).sum(dims)
            union = inputs_prob.sum(dims) + targets_onehot.sum(dims)
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            loss = 1 - dice.mean()
            return loss

    SUPPORTED_LOSSES = {
        # classification
        "cross_entropy": nn.CrossEntropyLoss,
        "nll": nn.NLLLoss,
        "focal": _FocalLoss,
        "dice": _DiceLoss,
        "dice_loss": _DiceLoss,
        "weighted_cross_entropy": "_custom",  # placeholder handled in _get_loss_fn
        # regression
        "mse": nn.MSELoss,
        "mae": nn.L1Loss,
        "huber": nn.HuberLoss,
        "log_cosh": "_custom_reg",
    }

    def __init__(self, model: nn.Module, task_type: str, config_path: Union[str, Path]):
        self.task_type = task_type.lower()
        if self.task_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported task_type: {task_type}")

        # Load YAML
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # Device
        if self.config["training"].get("device", "auto") == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config["training"]["device"])

        self.model = model.to(self.device)

        # ------------- 新增: 记录类别数供指标计算 ------------- #
        self.num_classes: int = self.config.get("dataset", {}).get("num_classes", 2)

        # ---------------- Strategy & Agents ---------------- #
        self.current_strategy: Dict[str, Any] = {
            "loss_function": self.config["training"].get("loss_function", "cross_entropy"),
            "optimizer": self.config["training"].get("optimizer", "adam"),
            "learning_rate": self.config["training"].get("learning_rate", 1e-3),
        }

        # For classification tasks, initialise default class weights (needed by MetricsLogger)
        if self.task_type == "classification":
            num_classes = self.config["dataset"].get("num_classes", 10)
            for i in range(num_classes):
                self.current_strategy.setdefault(f"weight_class_{i}", 1.0)

        self.loss_fn = self._get_loss_fn(self.current_strategy["loss_function"])
        self.optimizer = self._get_optimizer()

        self.agent = StrategyAgent(self.config["llm"])

        # ---------------- 新增：Judger & Prompt Optimizer ---------------- #
        self.judger = LLMJudger(self.config["llm"])
        self.prompt_optimizer = LLMPromptOptimizer(self.config["llm"])
        self.optimization_history: List[Dict[str, Any]] = []

        # ---------------- 独立日志目录 ---------------- #
        base_dir = Path(self.config["logging"]["log_dir"])
        ts = datetime.now().strftime("_strategy_%Y%m%d_%H%M%S")
        log_dir = base_dir / ts
        log_dir.mkdir(parents=True, exist_ok=True)
        # 更新配置中的路径，保证后续记录使用新目录
        self.config["logging"]["log_dir"] = str(log_dir)

        # Logging
        if self.task_type == "classification":
            self.metrics_logger = MetricsLogger(log_dir, self.config)
        else:
            self.metrics_logger = RegressionMetricsLogger(log_dir, self.config)

        # Agent output JSONL
        self._agent_log_file = log_dir / "strategy_agent_outputs.jsonl"

        # ---------------- 新增：通用 Agent 输出日志方法 ---------------- #

    def _log_agent_output(self, agent: str, payload: Dict[str, Any]):
        """Append agent output to corresponding JSONL file under log_dir."""
        log_file = Path(self.config["logging"]["log_dir"]) / f"{agent.lower()}_outputs.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_agent(self, payload: Dict[str, Any]):
        with open(self._agent_log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _get_loss_fn(self, name: str):
        name = name.lower()
        if name not in self.SUPPORTED_LOSSES:
            # 动态添加别名或自定义加权交叉熵
            if name in {"weighted_cross_entropy", "wce"}:
                def _wce(inputs: torch.Tensor, targets: torch.Tensor):
                    # 构造当前类别权重张量
                    weights = torch.tensor(
                        [self.current_strategy.get(f"weight_class_{i}", 1.0) for i in range(self.num_classes)],
                        device=inputs.device,
                    )
                    return nn.functional.cross_entropy(inputs, targets, weight=weights)

                return _wce
            if name == "log_cosh":
                def _log_cosh(preds: torch.Tensor, targets: torch.Tensor):
                    diff = preds.squeeze() - targets.squeeze()
                    return torch.mean(torch.log(torch.cosh(diff)))

                return _log_cosh
            raise ValueError(f"Unsupported loss function: {name}")
        if name in {"cross_entropy", "nll"}:
            # For classification, weights can be extended if needed
            return self.SUPPORTED_LOSSES[name]()
        if name == "log_cosh":
            def _log_cosh(preds: torch.Tensor, targets: torch.Tensor):
                diff = preds.squeeze() - targets.squeeze()
                return torch.mean(torch.log(torch.cosh(diff)))

            return _log_cosh
        if name in {"focal", "focal_loss"}:
            return self.SUPPORTED_LOSSES[name]()
        return self.SUPPORTED_LOSSES[name]()

    def _get_optimizer(self):
        optimizer_name = self.current_strategy["optimizer"].lower()
        lr = self.current_strategy["learning_rate"]
        if optimizer_name not in self.SUPPORTED_OPTIMIZERS:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        optim_cls = self.SUPPORTED_OPTIMIZERS[optimizer_name]
        wd = self.current_strategy.get('weight_decay', 0.0)

        def _ensure_sgd_keys(opt: torch.optim.Optimizer, default_m: float = 0.0):
            """确保SGD相关param_group包含 momentum/dampening 键"""
            for g in opt.param_groups:
                if 'momentum' not in g:
                    g['momentum'] = default_m
                if 'dampening' not in g:
                    g['dampening'] = 0.0
            return opt

        if optimizer_name == 'sgd':
            return _ensure_sgd_keys(torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.0, weight_decay=wd), 0.0)
        elif optimizer_name in {'sgd_with_momentum', 'sgd_momentum', 'sgdm'}:
            # 默认动量 0.9，可后续在 current_strategy 中加字段调整
            return _ensure_sgd_keys(torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd), 0.9)
        elif optimizer_name == 'adamw':
            return _ensure_sgd_keys(torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd))
        elif optimizer_name == 'adagrad':
            return _ensure_sgd_keys(torch.optim.Adagrad(self.model.parameters(), lr=lr, weight_decay=wd))
        elif optimizer_name == 'rmsprop':
            opt = torch.optim.RMSprop(self.model.parameters(), lr=lr, momentum=0.0, weight_decay=wd)
            # 手动初始化 square_avg
            for group in opt.param_groups:
                for p in group['params']:
                    state = opt.state[p]
                    if 'square_avg' not in state:
                        state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if 'momentum_buffer' not in state and group.get('momentum', 0.0) != 0.0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if 'step' not in state:
                        state['step'] = 0
            return _ensure_sgd_keys(opt, 0.0)
        opt_generic = optim_cls(self.model.parameters(), lr=lr, weight_decay=wd)
        return _ensure_sgd_keys(opt_generic)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def _calc_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        if self.task_type == "classification":
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy()
            return {
                "accuracy": float((preds == targets_np).mean()),
                "precision": float(precision_score(targets_np, preds, average="macro", zero_division=0)),
                "recall": float(recall_score(targets_np, preds, average="macro", zero_division=0)),
                "roc_auc": self._compute_roc_auc(targets_np, outputs, self.num_classes),
            }
        else:
            preds = outputs.squeeze().detach().cpu()
            t = targets.squeeze().detach().cpu()
            mse = torch.mean((preds - t) ** 2).item()
            mae = torch.mean(torch.abs(preds - t)).item()
            return {"mse": mse, "mae": mae}

    @staticmethod
    def _compute_roc_auc(y_true, logits, num_classes):
        """计算 ROC-AUC，分类数>2 时使用 macro OVR。失败返回 NaN."""
        try:
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            if num_classes == 2:
                return float(roc_auc_score(y_true, probs[:, 1]))
            return float(roc_auc_score(y_true, probs, multi_class='ovr', average='macro'))
        except Exception:
            return float('nan')

    # ------------------------------------------------------------------
    # Training / Validation loops
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        self.model.train(train)
        total_loss = 0.0
        outputs_list: List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []

        with torch.set_grad_enabled(train):
            with tqdm(loader, desc="train" if train else "val") as pbar:
                for b_idx, (data, target) in enumerate(pbar):
                    data, target = data.to(self.device), target.to(self.device)
                    if train:
                        self.optimizer.zero_grad()

                    logits = self.model(data)
                    if self.task_type == "classification":
                        loss = self.loss_fn(logits, target)
                    else:
                        # regression – ensure shapes compatible
                        loss = self.loss_fn(logits.squeeze(), target.squeeze())

                    if train:
                        loss.backward()
                        self.optimizer.step()

                    total_loss += loss.item()
                    outputs_list.append(logits.detach())
                    targets_list.append(target.detach())

                    pbar.set_postfix({"loss": total_loss / (b_idx + 1), "lr": self.current_strategy["learning_rate"]})

        outputs_tensor = torch.cat(outputs_list)
        targets_tensor = torch.cat(targets_list)
        metrics = self._calc_metrics(outputs_tensor, targets_tensor)
        metrics["loss"] = total_loss / len(loader)
        return metrics

    # ------------------------------------------------------------------
    # Public training interface
    # ------------------------------------------------------------------

    def _apply_strategy(self, strategy: Dict[str, Any]):
        """Update loss_fn & optimizer according to `strategy` (lr is **not** modified)."""
        # Loss function switch
        if "loss_function" in strategy and strategy["loss_function"] != self.current_strategy["loss_function"]:
            self.current_strategy["loss_function"] = strategy["loss_function"]
            self.loss_fn = self._get_loss_fn(strategy["loss_function"])

        # Optimizer change – recreate & drop old state (simple implementation)
        if "optimizer" in strategy and strategy["optimizer"] != self.current_strategy["optimizer"]:
            self.current_strategy["optimizer"] = strategy["optimizer"]
            self.optimizer = self._get_optimizer()

        # Learning-rate changes are intentionally ignored per user requirement

    # ------------------------------------------------------------------
    # Strategy optimisation evaluation (Judger + PromptOptimizer loop)
    # ------------------------------------------------------------------

    def _evaluate_strategy_optimization(
        self,
        train_loader,
        val_loader: Optional[DataLoader],
        new_strategy: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Follow Judger/PromptOptimizer logic to decide whether to adopt `new_strategy`."""
        # Step 0: snapshot current (end-of-epoch) states
        original_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        original_optimizer_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()}
        baseline_strategy = self.current_strategy.copy()

        # Step 1: baseline – train 1 extra epoch with **no change**
        baseline_train = self._run_epoch(train_loader, train=True)
        if val_loader is not None:
            baseline_val = self._run_epoch(val_loader, train=False)
            baseline_metrics: Dict[str, Any] = {"train": baseline_train, "val": baseline_val}
        else:
            baseline_metrics = {"train": baseline_train}

        baseline_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
        baseline_optimizer_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()}

        attempt_params = new_strategy or {}
        attempt_idx = 0
        final_use_opt = False
        final_reason = ""
        final_suggestion: Optional[str] = None
        final_metrics = baseline_metrics

        while attempt_idx < self.judger.max_attempts and attempt_params:
            attempt_idx += 1

            # Reset to original state before baseline epoch
            self.model.load_state_dict(original_model_state)
            # 回到基准策略
            self.current_strategy = baseline_strategy.copy()
            self.loss_fn = self._get_loss_fn(self.current_strategy["loss_function"])
            self.optimizer = self._get_optimizer()
            # 恢复优化器状态（与基准策略兼容）
            try:
                self.optimizer.load_state_dict(original_optimizer_state)
            except ValueError:
                # 如果优化器类型不兼容，忽略状态载入
                pass

            # Apply candidate strategy
            self._apply_strategy(attempt_params)

            # Train 1 epoch with candidate strategy
            cand_train = self._run_epoch(train_loader, train=True)
            if val_loader is not None:
                cand_val = self._run_epoch(val_loader, train=False)
                cand_metrics: Dict[str, Any] = {"train": cand_train, "val": cand_val}
            else:
                cand_metrics = {"train": cand_train}

            # Record history
            param_changes = {
                k: {"from": baseline_strategy.get(k), "to": attempt_params.get(k, baseline_strategy.get(k))}
                for k in set(baseline_strategy) | set(attempt_params)
            }
            self.optimization_history.append(
                {
                    "param_changes": param_changes,
                    "original_metrics": baseline_metrics,
                    "optimized_metrics": cand_metrics,
                }
            )

            # Judger evaluation
            use_opt, reason, suggestion = self.judger.judge_optimization(
                baseline_metrics,
                cand_metrics,
                baseline_strategy,
                attempt_params,
                self.optimization_history,
            )

            self._log_agent_output(
                "judger",
                {
                    "attempt": attempt_idx,
                    "use_opt": use_opt,
                    "reason": reason,
                    "suggestion": suggestion,
                },
            )

            if use_opt:
                final_use_opt = True
                final_reason = reason
                final_suggestion = suggestion
                final_metrics = cand_metrics
                baseline_metrics = cand_metrics
                baseline_strategy = self.current_strategy.copy()
                baseline_model_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                baseline_optimizer_state = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()}
                break

            # Not adopted – maybe get suggestion
            final_reason = reason
            final_suggestion = suggestion

            if suggestion:
                # PromptOptimizer refine
                refined_guidance = self.prompt_optimizer.refine_suggestion(
                    "strategy optimisation", suggestion
                )
                self._log_agent_output("prompt_optimizer", {"attempt": attempt_idx, "guidance": refined_guidance})

                # Ask StrategyAgent again with guidance
                raw_txt, next_params, should_update = self.agent.decide_strategy(
                    self.task_type,
                    baseline_strategy,
                    baseline_metrics,
                    self.config,
                    guidance=refined_guidance,
                )
                self._log_agent_output("strategy_agent", {"attempt": attempt_idx, "suggestion": raw_txt})

                if should_update and next_params != attempt_params:
                    attempt_params = next_params
                    continue

            # no further action
            break

        # Restore if not adopted
        if not final_use_opt:
            self.model.load_state_dict(baseline_model_state)
            self.optimizer = self._get_optimizer()  # recreate with baseline strategy
            self.optimizer.load_state_dict(baseline_optimizer_state)
            self.current_strategy = baseline_strategy.copy()
            self.loss_fn = self._get_loss_fn(self.current_strategy["loss_function"])

        return {
            "use_optimized": final_use_opt,
            "reason": final_reason,
            "suggestion": final_suggestion,
            "metrics": final_metrics,
        }

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        best_val_loss = float("inf")
        patience = 0
        epochs = self.config["training"].get("epochs", 10)
        early_patience = self.config["training"].get("early_stopping_patience", 5)
        query_interval = self.config["logging"].get("save_interval", 1)

        for epoch in range(epochs):
            train_metrics = self._run_epoch(train_loader, train=True)
            if val_loader is not None:
                val_metrics = self._run_epoch(val_loader, train=False)
                metrics = {"train": train_metrics, "val": val_metrics}

                # Early stopping (based on val loss)
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience = 0
                else:
                    patience += 1
                if patience >= early_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            else:
                metrics = {"train": train_metrics}

            # Log
            self.metrics_logger.update(epoch, metrics, self.current_strategy)

            # Strategy agent every `query_interval` epochs
            if (epoch + 1) % query_interval == 0:
                suggestion, new_strat, should_update = self.agent.decide_strategy(
                    self.task_type, self.current_strategy, metrics, self.config
                )
                self._log_agent({"epoch": epoch, "suggestion": suggestion})
                print(f"Epoch {epoch}: StrategyAgent suggestion -> {suggestion}")

                # 记录到 metrics_logger（若实现）
                if hasattr(self.metrics_logger, "add_llm_suggestion"):
                    try:
                        self.metrics_logger.add_llm_suggestion(epoch, suggestion)
                    except Exception:
                        pass

                if should_update and new_strat:
                    print("评估策略更新:", new_strat)
                    eval_result = self._evaluate_strategy_optimization(train_loader, val_loader, new_strat)

                    if eval_result["use_optimized"]:
                        print("采用优化后的策略:", self.current_strategy)
                    else:
                        print("保持原策略")
                        if eval_result["suggestion"]:
                            print("Judger改进建议:", eval_result["suggestion"])

                    # reset judger attempts for next epoch
                    self.judger.reset_attempts()
                    self.optimization_history = []

        # Save final model
        torch.save(self.model.state_dict(), Path(self.metrics_logger.log_dir) / "model.pt")
        print("Model saved to", Path(self.metrics_logger.log_dir) / "model.pt") 