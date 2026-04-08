import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

import datetime
import argparse
import os
import random
import time
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
from sklearn.model_selection import KFold
from Dataset import DatasetGaze, Dataset_nogaze
from Engines import train_one_epoch, val_one_epoch
from models.GUIDE.GUIDE import GUIDE
from utils import misc


def get_args_parser():
    parser = argparse.ArgumentParser('5-fold CV hyperparameter tuning', add_help=False)
    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('--lr_drop', default=500, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--size', default=224, type=int)
    parser.add_argument('--output_dir', default='output/weights', help='path where to save')
    parser.add_argument('--data_dir', default='Data')
    parser.add_argument('--csv_path', default='Data/image_names.csv')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--n_splits', default=5, type=int)
    parser.add_argument('--patience', default=10, type=int)
    return parser


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_data = pd.read_csv(args.csv_path, encoding='gbk', dtype={'image_id': str})
    csv_train = csv_data[csv_data['train_test'] == 'train'].reset_index(drop=True)

    dataset_train_all = DatasetGaze(args.data_dir, args.csv_path, 'train', args.size)

    dataset_val_all = Dataset_nogaze(args.data_dir, args.csv_path, 'train', args.size)

    indices = np.arange(len(csv_train))
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    param_grid = [
        # lr = 1e-3
        {"lr": 1e-3, "batch_size": 16, "weight_decay": 1e-4},
        {"lr": 1e-3, "batch_size": 32, "weight_decay": 1e-4},
        {"lr": 1e-3, "batch_size": 16, "weight_decay": 1e-5},
        {"lr": 1e-3, "batch_size": 32, "weight_decay": 1e-5},

        # lr = 5e-4
        {"lr": 5e-4, "batch_size": 16, "weight_decay": 1e-4},
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 1e-4},
        {"lr": 5e-4, "batch_size": 16, "weight_decay": 1e-5},
        {"lr": 5e-4, "batch_size": 32, "weight_decay": 1e-5},

        # lr = 1e-4
        {"lr": 1e-4, "batch_size": 16, "weight_decay": 1e-4},
        {"lr": 1e-4, "batch_size": 32, "weight_decay": 1e-4},
        {"lr": 1e-4, "batch_size": 16, "weight_decay": 1e-5},
        {"lr": 1e-4, "batch_size": 32, "weight_decay": 1e-5},

        # lr = 5e-5
        {"lr": 5e-5, "batch_size": 16, "weight_decay": 1e-4},
        {"lr": 5e-5, "batch_size": 32, "weight_decay": 1e-4},
        {"lr": 5e-5, "batch_size": 16, "weight_decay": 1e-5},
        {"lr": 5e-5, "batch_size": 32, "weight_decay": 1e-5},
    ]

    best_param = None
    best_mean_acc = -1.0
    all_results = []

    print("Start 5-fold CV hyperparameter tuning...")
    start_time = time.time()

    for p_i, hp in enumerate(param_grid):
        print("\n" + "=" * 60)
        print(f"Param set {p_i+1}/{len(param_grid)}: {hp}")

        fold_accs = []
        fold_losses = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(indices), 1):
            print("-" * 40)
            print(f"Fold {fold}/{args.n_splits}")

            train_subset = Subset(dataset_train_all, train_idx.tolist())
            val_subset = Subset(dataset_val_all, val_idx.tolist())

            dataloader_train = DataLoader(
                train_subset,
                batch_size=hp["batch_size"],
                shuffle=True,
                num_workers=args.num_workers
            )
            dataloader_val = DataLoader(
                val_subset,
                batch_size=hp["batch_size"],
                shuffle=False,
                num_workers=args.num_workers
            )

            model = GUIDE(num_classes=2).to(device)
            model_without_ddp = model

            param_dicts = [{"params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad]}]
            optimizer = torch.optim.AdamW(param_dicts, lr=hp["lr"], weight_decay=hp["weight_decay"])
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

            writer = SummaryWriter(log_dir=str(output_dir / f"summary_p{p_i+1}_f{fold}"))

            best_val_loss = float("inf")
            best_val_acc = -1.0
            patience_counter = 0

            for epoch in range(0, args.epochs):
                train_one_epoch(model, dataloader_train, optimizer, device, epoch, args, writer)
                val_loss, val_acc = val_one_epoch(model, dataloader_val, device, epoch, writer)

                lr_scheduler.step()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            writer.close()

            fold_losses.append(best_val_loss)
            fold_accs.append(best_val_acc)

            print(f"Fold {fold} best: val_loss={best_val_loss:.6f}, val_acc={best_val_acc:.6f}")

        mean_acc = float(np.mean(fold_accs))
        mean_loss = float(np.mean(fold_losses))

        all_results.append({
            "hp": hp,
            "mean_val_acc": mean_acc,
            "mean_val_loss": mean_loss,
            "fold_accs": fold_accs,
            "fold_losses": fold_losses
        })

        print(f"Param set result: mean_val_acc={mean_acc:.6f}, mean_val_loss={mean_loss:.6f}")

        if mean_acc > best_mean_acc:
            best_mean_acc = mean_acc
            best_param = hp

    save_path = output_dir / "cv_tuning_results.txt"
    with open(save_path, "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(str(r) + "\n")
        f.write("\n")
        f.write(f"BEST_PARAM: {best_param}\n")
        f.write(f"BEST_MEAN_VAL_ACC: {best_mean_acc:.6f}\n")

    print("\nBest hyperparameters:", best_param)
    print("Best mean validation accuracy:", best_mean_acc)
    print("Saved to:", save_path)

    total_time = time.time() - start_time
    print('Tuning time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CV tuning script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(args.GPU_ids)

    main(args)