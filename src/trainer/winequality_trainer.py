from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
from torch import nn
from torch.utils.data import DataLoader

import yaml
import datetime

from ..llm.advisor import LLMAdvisor
from ..llm.judger import LLMJudger
from ..llm.prompt_optimizer import LLMPromptOptimizer
from ..utils.regression_metrics_logger import RegressionMetricsLogger


class WineQualityTrainer:
    """Trainer for the Wine Quality regression task.

    The implementation mirrors `HousingTrainer` but uses a dedicated
    sub-directory prefix (`winequality_`) for logging so that results from
    different datasets are cleanly separated.
    """

    def __init__(self, model: nn.Module, config_path: Union[str, Path]):
        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # Create experiment-specific log dir, e.g. logs/winequality_20250727_123456/
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        root_log_dir = Path(self.config["logging"]["log_dir"])
        self.log_dir = root_log_dir / f"winequality_{ts}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        # Write back for downstream tools
        self.config["logging"]["log_dir"] = str(self.log_dir)

        # Device selection
        if self.config["training"]["device"] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config["training"]["device"])

        # Model & optimiser
        self.model = model.to(self.device)
        tr_cfg = self.config["training"]
        self.current_params: Dict[str, Any] = {
            "learning_rate": tr_cfg["learning_rate"],
        }

        # Optimiser choice
        opt_name = tr_cfg.get("optimizer", "adam").lower()
        opt_lr = self.current_params["learning_rate"]

        if opt_name == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=opt_lr, momentum=0.9)
        elif opt_name == "adagrad":
            self.optimizer = torch.optim.Adagrad(self.model.parameters(), lr=opt_lr)
        elif opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=opt_lr)
        else:  # default Adam
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt_lr)

        # Loss function (regression)
        loss_name = tr_cfg.get("loss_function", "mse").lower()
        if loss_name in ("mae", "l1"):
            self._loss_fn = nn.L1Loss()
        elif loss_name in ("huber", "smoothl1"):
            self._loss_fn = nn.SmoothL1Loss()
        else:
            self._loss_fn = nn.MSELoss()

        # LLM components (overridden later for baseline usage)
        self.advisor = LLMAdvisor(self.config["llm"])
        self.judger = LLMJudger(self.config["llm"])
        self.prompt_optimizer = LLMPromptOptimizer(self.config["llm"])

        # Metrics logger
        self.metrics_logger = RegressionMetricsLogger(self.log_dir, self.config)

        # Optimisation history (for Judger)
        self.optimization_history = []

    # ---------- Helper logging ---------- #

    def _log_agent_output(self, agent: str, payload: Dict[str, Any]):
        file = self.log_dir / f"{agent.lower()}_outputs.jsonl"
        with open(file, "a", encoding="utf-8") as f:
            import json as _json
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")

    # ---------- Metrics ---------- #

    def _calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        preds = outputs.squeeze().detach().cpu()
        t = targets.squeeze().detach().cpu()
        mse = torch.mean((preds - t) ** 2).item()
        mae = torch.mean(torch.abs(preds - t)).item()
        ss_res = torch.sum((t - preds) ** 2)  # 残差平方和
        ss_tot = torch.sum((t - torch.mean(t)) ** 2)  # 总平方和
        
        if ss_tot == 0:
            r2 = 1.0  # 如果所有目标值相同，完美预测时 R²=1
        else:
            r2 = 1 - (ss_res / ss_tot).item()
            # 限制 R² 在合理范围内，避免数值误差导致的异常值
            r2 = max(min(r2, 1.0), -100.0)  # 允许负值但限制最小值
        
        return {
            "mse": mse, 
            "mae": mae, 
            "r2": r2, 
            "loss": mse  # 保持 loss 与 mse 一致，便于现有逻辑
        }


    def _update_parameters(self, new_params: Dict[str, Any]):
        if "learning_rate" in new_params:
            self.current_params["learning_rate"] = new_params["learning_rate"]
            for g in self.optimizer.param_groups:
                g["lr"] = new_params["learning_rate"]
        # Extendable for more params (e.g. dropout)

    # ---------- Single epoch ---------- #

    def _run_one_epoch(self, loader: DataLoader, is_train: bool) -> Dict[str, float]:
        self.model.train(mode=is_train)

        total_loss = 0.0
        outputs_list, targets_list = [], []
        from tqdm import tqdm
        with torch.set_grad_enabled(is_train):
            for batch_idx, (data, target) in enumerate(tqdm(loader, desc="train" if is_train else "val")):
                data, target = data.to(self.device), target.to(self.device)

                if is_train:
                    self.optimizer.zero_grad()

                output = self.model(data)
                loss = self._loss_fn(output, target)

                if is_train:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                outputs_list.append(output.detach())
                targets_list.append(target.detach())

            outputs_tensor = torch.cat(outputs_list)
            targets_tensor = torch.cat(targets_list)
            metrics = self._calculate_metrics(outputs_tensor, targets_tensor)
            metrics["loss"] = total_loss / len(loader)
            return metrics

    # ---------- Optimisation evaluation ---------- #

    def _evaluate_optimization(self, train_loader: DataLoader, val_loader: Optional[DataLoader], new_params: Dict[str, Any]):
        baseline_state = {
            "model": {k: v.clone() for k, v in self.model.state_dict().items()},
            "optimizer": self.optimizer.state_dict(),
            "params": self.current_params.copy(),
        }

        baseline_train_metrics = self._run_one_epoch(train_loader, is_train=True)
        baseline_val_metrics = self._run_one_epoch(val_loader, is_train=False) if val_loader else {}
        baseline_metrics = {"train": baseline_train_metrics, "val": baseline_val_metrics}

        # Attempt new params
        self._update_parameters(new_params)
        opt_train_metrics = self._run_one_epoch(train_loader, is_train=True)
        opt_val_metrics = self._run_one_epoch(val_loader, is_train=False) if val_loader else {}
        optimized_metrics = {"train": opt_train_metrics, "val": opt_val_metrics}

        use_opt, reason, suggestion = self.judger.judge_optimization(
            baseline_metrics,
            optimized_metrics,
            baseline_state["params"],
            new_params,
            self.optimization_history,
        )

        if use_opt:
            self._log_agent_output("judger", {"attempt_params": new_params, "use_opt": True, "reason": reason})
            self.optimization_history.append(
                {
                    "param_changes": new_params,
                    "original_metrics": baseline_metrics,
                    "optimized_metrics": optimized_metrics,
                }
            )
            return True, reason, suggestion

        # Revert
        self._log_agent_output(
            "judger",
            {"attempt_params": new_params, "use_opt": False, "reason": reason, "suggestion": suggestion},
        )
        self.model.load_state_dict(baseline_state["model"])
        self.optimizer.load_state_dict(baseline_state["optimizer"])
        self.current_params = baseline_state["params"].copy()
        return False, reason, suggestion

    # ---------- Training loop ---------- #

    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        best_val_loss = float("inf")
        patience = 0

        for epoch in range(self.config["training"]["epochs"]):
            train_metrics = self._run_one_epoch(train_loader, is_train=True)
            if val_loader is not None:
                val_metrics = self._run_one_epoch(val_loader, is_train=False)
            else:
                val_metrics = {}

            metrics = {"train": train_metrics, "val": val_metrics}

            # Early stopping
            if val_loader is not None:
                if val_metrics.get("loss", float("inf")) < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    patience = 0
                else:
                    patience += 1
                if patience >= self.config["training"]["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Log metrics
            self.metrics_logger.update(epoch, metrics, self.current_params)

            # Advisor suggestion
            suggestion, new_params, should_update = self.advisor.get_advice(
                self.current_params, metrics, self.config
            )
            print(f"Epoch {epoch} advisor 建议: {suggestion}")
            self._log_agent_output("advisor", {"epoch": epoch, "suggestion": suggestion})

            if should_update and new_params:
                print("评估更新参数", new_params)
                use_opt, reason, sugg = self._evaluate_optimization(train_loader, val_loader, new_params)
                if use_opt:
                    print("采纳优化参数", new_params)
                else:
                    print("未采纳，原因:", reason)
                if sugg:
                    try:
                        refined = self.prompt_optimizer.refine_suggestion("", sugg)
                    except Exception:
                        refined = sugg
                    self._log_agent_output("prompt_optimizer", {"epoch": epoch, "refined_guidance": refined})

            # Reset Judger each epoch
            self.judger.reset_attempts()
            self.optimization_history = []

        # Save final model
        torch.save(self.model.state_dict(), self.log_dir / "model.pt")
        print("模型已保存到", self.log_dir / "model.pt") 