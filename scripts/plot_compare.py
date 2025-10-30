import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_metrics(json_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def avg_class_weights(metrics_dict):
    weights_keys = [k for k in metrics_dict.keys() if k.startswith('weight_class_')]
    if not weights_keys:
        return None
    # 假定每个key对应长度相同
    length = len(metrics_dict[weights_keys[0]])
    avg = [0.0] * length
    for k in weights_keys:
        vals = metrics_dict[k]
        avg = [a + v for a, v in zip(avg, vals)]
    avg = [v / len(weights_keys) for v in avg]
    return avg


def main(orig_path: Path, baseline_path: Path, output_path: Path):
    orig_metrics = load_metrics(orig_path)
    base_metrics = load_metrics(baseline_path)

    epochs_orig = list(range(len(orig_metrics['train_loss'])))
    epochs_base = list(range(len(base_metrics['train_loss'])))

    # 统一 epoch 数量，取较短部分
    max_len = min(len(epochs_orig), len(epochs_base))
    epochs = list(range(max_len))

    def trunc(lst):
        return lst[:max_len]

    plt.figure(figsize=(12, 10))

    # ----------- Loss -------------
    plt.subplot(2, 2, 1)
    plt.plot(epochs, trunc(orig_metrics['train_loss']), label='orig_train_loss')
    plt.plot(epochs, trunc(base_metrics['train_loss']), label='base_train_loss')
    if 'val_loss' in orig_metrics and 'val_loss' in base_metrics:
        plt.plot(epochs, trunc(orig_metrics['val_loss']), label='orig_val_loss')
        plt.plot(epochs, trunc(base_metrics['val_loss']), label='base_val_loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # ----------- Accuracy -------------
    plt.subplot(2, 2, 2)
    plt.plot(epochs, trunc(orig_metrics['train_accuracy']), label='orig_train_acc')
    plt.plot(epochs, trunc(base_metrics['train_accuracy']), label='base_train_acc')
    if 'val_accuracy' in orig_metrics and 'val_accuracy' in base_metrics:
        plt.plot(epochs, trunc(orig_metrics['val_accuracy']), label='orig_val_acc')
        plt.plot(epochs, trunc(base_metrics['val_accuracy']), label='base_val_acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # ----------- Learning Rate -------------
    plt.subplot(2, 2, 3)
    plt.plot(epochs, trunc(orig_metrics['learning_rate']), label='orig_lr')
    plt.plot(epochs, trunc(base_metrics['learning_rate']), label='base_lr')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()

    # ----------- Avg Class Weight -------------
    avg_w_orig = avg_class_weights(orig_metrics)
    avg_w_base = avg_class_weights(base_metrics)
    plt.subplot(2, 2, 4)
    if avg_w_orig and avg_w_base:
        plt.plot(epochs, trunc(avg_w_orig), label='orig_avg_cls_w')
        plt.plot(epochs, trunc(avg_w_base), label='base_avg_cls_w')
    plt.title('Average Class Weight')
    plt.xlabel('Epoch')
    plt.ylabel('Weight')
    if avg_w_orig and avg_w_base:
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f'Comparison plot saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare metrics between experiments')
    parser.add_argument('--orig', type=str, required=True, help='Path to original metrics_history.json')
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline metrics_history.json')
    parser.add_argument('--output', type=str, default='logs/compare_metrics.png', help='Output image path')
    args = parser.parse_args()

    main(Path(args.orig), Path(args.baseline), Path(args.output)) 