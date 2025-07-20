import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def load_metrics(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def truncate(lst, n):
    return lst[:n]


def main(exp_path: Path, baseline_path: Path, output: Path):
    exp = load_metrics(exp_path)
    base = load_metrics(baseline_path)

    n_epochs = min(len(exp['train_loss']), len(base['train_loss']))
    epochs = list(range(n_epochs))

    plt.figure(figsize=(12, 10))

    # Loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, truncate(exp['train_loss'], n_epochs), label='exp_train_loss')
    plt.plot(epochs, truncate(base['train_loss'], n_epochs), label='base_train_loss')
    if base['val_loss'] and exp['val_loss']:
        plt.plot(epochs, truncate(exp['val_loss'], n_epochs), label='exp_val_loss')
        plt.plot(epochs, truncate(base['val_loss'], n_epochs), label='base_val_loss')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()

    # MSE
    plt.subplot(2, 2, 2)
    if 'train_mse' in exp and 'train_mse' in base:
        plt.plot(epochs, truncate(exp['train_mse'], n_epochs), label='exp_train_mse')
        plt.plot(epochs, truncate(base['train_mse'], n_epochs), label='base_train_mse')
        if exp['val_mse'] and base['val_mse']:
            plt.plot(epochs, truncate(exp['val_mse'], n_epochs), label='exp_val_mse')
            plt.plot(epochs, truncate(base['val_mse'], n_epochs), label='base_val_mse')
        plt.title('MSE'); plt.xlabel('Epoch'); plt.ylabel('MSE'); plt.legend()

    # MAE
    plt.subplot(2, 2, 3)
    if 'train_mae' in exp and 'train_mae' in base:
        plt.plot(epochs, truncate(exp['train_mae'], n_epochs), label='exp_train_mae')
        plt.plot(epochs, truncate(base['train_mae'], n_epochs), label='base_train_mae')
        if exp['val_mae'] and base['val_mae']:
            plt.plot(epochs, truncate(exp['val_mae'], n_epochs), label='exp_val_mae')
            plt.plot(epochs, truncate(base['val_mae'], n_epochs), label='base_val_mae')
        plt.title('MAE'); plt.xlabel('Epoch'); plt.ylabel('MAE'); plt.legend()

    # Learning rate
    plt.subplot(2, 2, 4)
    plt.plot(epochs, truncate(exp['learning_rate'], n_epochs), label='exp_lr')
    plt.plot(epochs, truncate(base['learning_rate'], n_epochs), label='base_lr')
    plt.title('Learning Rate'); plt.xlabel('Epoch'); plt.ylabel('LR'); plt.legend()

    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300)
    print(f'Plot saved to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare housing regression metrics')
    parser.add_argument('--exp', required=True, help='Path to experiment metrics_history.json')
    parser.add_argument('--baseline', required=True, help='Path to baseline metrics_history.json')
    parser.add_argument('--output', default='logs/compare_housing_metrics.png', help='Output image path')
    args = parser.parse_args()

    main(Path(args.exp), Path(args.baseline), Path(args.output)) 