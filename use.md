# Codebase Quick Reference


## scripts/
- **optimize_structure.py**：使用 LLM/网格搜索优化模型结构；示例 `python scripts/optimize_structure.py --config CONFIG`。
- **plot_compare.py**：比较两组 `metrics_history.json` 并绘制曲线；示例 `python scripts/plot_compare.py --orig A.json --baseline B.json --output out.png`。
- **plot_compare_housing.py**：针对 Housing 任务的指标对比绘图脚本，用法同上。
- **train.py**：MNIST 通用训练入口（含 Advisor/Judger）；`python scripts/train.py --config config/mnist_aug.yaml [--model_path model.pt]`。
- **train_adaptive.py**：自适应 Trainer（动态调整参数/增强）；`python scripts/train_adaptive.py --config CONFIG`。
- **train_aug_transform.py**：图像任务按 epoch 动态应用 LLM 选择的增强；`python scripts/train_aug_transform.py --config CONFIG`。
- **train_augmented_noadvisor.py**：仅数据增强、无 Advisor 的 MNIST 训练；`python scripts/train_augmented_noadvisor.py --config config/mnist_aug.yaml`。
- **train_baseline.py**：MNIST 无 LLM 的基线训练；`python scripts/train_baseline.py --config CONFIG`。
- **train_housing_augmented.py**：房价回归 + LLM 表格增强 + Judger 评估；`python scripts/train_housing_augmented.py --config config/housing_config.yaml`。
- **train_housing_baseline.py**：房价回归基线训练；`python scripts/train_housing_baseline.py --config config/housing_config.yaml`。
- **train_housing.py**：房价回归完整 LLM 调优流程；`python scripts/train_housing.py --config config/housing_config.yaml [--model_path model.pt]`。

## src/llm/
- **__init__.py**：LLM 子包初始化。
- **advisor.py**：调用 LLM 给出超参数调整建议。
- **augmentation_chooser.py**：让 LLM 在允许列表中选择下个数据增强。
- **judger.py**：由 LLM 判断优化结果是否优于基线。
- **prompt_optimizer.py**：将 Judger 建议重写为更佳提示。
- **model_structure_optimizer.py**：自动探索/提议更优网络结构。
- **strategy_agent.py**：高层次代理，编排 Advisor/Judger/Prompt-Optimizer。 

## src/models/
- **__init__.py**：模型包初始化。
- **cnn_model.py**：可配置层数的 MNIST CNN 实现。
- **custom_model.py**：用于表格数据的多层感知机 (MLP)。
- **optimized_model.py**：经调参优化的 CNN 版本。
- **optimized_housing_model.py**：为 Housing 任务优化的 MLP 结构。

## src/trainer/
- **__init__.py**：Trainer 子包初始化。
- **trainer.py**：通用分类 Trainer，含 Advisor/Judger 逻辑。
- **adaptive_trainer.py**：支持动态参数/增强的适应式 Trainer。
- **augmentation_trainer.py**：占位基类，仅继承 `Trainer`。
- **augmentation_transform_trainer.py**：按 epoch 应用 LLM 选定图像增强的 Trainer。
- **augmentation_noadvisor_trainer.py**：去掉 Advisor 的数据增强 Trainer。
- **housing_trainer.py**：房价回归核心 Trainer（支持动态优化器/损失）。
- **tabular_aug_trainer.py**：在回归任务中引入 LLM 表格增强并用 Judger 评估。

## src/utils/
- **__init__.py**：工具包初始化。
- **augment_ops.py**：图像增强操作注册表。
- **augmented_dataset.py**：包装数据集以追加 LLM 生成样本。
- **tabular_aug_ops.py**：表格增强调度（gaussian_noise、feature_dropout、shift 等）。
- **housing_dataset.py**：读取 Housing CSV 并标准化输出。
- **metrics_logger.py**：通用指标记录器。
- **regression_metrics_logger.py**：回归指标专用记录器。
- **housing_metrics_logger.py**：绘制并记录 Housing 回归指标。

## src/__init__.py
- 标记 `src` 目录为 Python 包。 

