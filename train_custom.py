import argparse
import os
import torch
from train_helper_FFNet import Trainer
from Networks import FFNet
import random
import numpy as np


def set_seed(seed, use_cpu=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not use_cpu and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Custom Crowd Counting Dataset")

    # Dataset paths
    parser.add_argument(
        "--data-dir",
        default="/path/to/your/converted/dataset",
        help="path ke dataset yang sudah dikonversi",
    )

    # Model config
    parser.add_argument("--dataset", default="custom", help="gunakan custom dataset")
    parser.add_argument("--arch", default="FFNet", help="arsitektur model")
    parser.add_argument(
        "--crop-size", type=int, default=256, help="ukuran crop untuk training"
    )

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate awal")
    parser.add_argument(
        "--eta_min",
        type=float,
        default=1e-6,
        help="learning rate minimum untuk CosineAnnealingLR",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="batch size untuk training"
    )
    parser.add_argument(
        "--max-epoch", type=int, default=1000, help="maksimum epoch training"
    )

    # Loss weights
    parser.add_argument("--wot", type=float, default=0.1, help="weight untuk OT loss")
    parser.add_argument("--wtv", type=float, default=0.01, help="weight untuk TV loss")
    parser.add_argument(
        "--reg", type=float, default=10.0, help="regularisasi entropy dalam sinkhorn"
    )
    parser.add_argument(
        "--num-of-iter-in-ot", type=int, default=100, help="iterasi sinkhorn"
    )
    parser.add_argument(
        "--norm-cood",
        type=int,
        default=0,
        help="normalisasi koordinat saat menghitung distance",
    )

    # Training settings
    parser.add_argument(
        "--val-epoch", type=int, default=5, help="frekuensi validasi (setiap n epoch)"
    )
    parser.add_argument(
        "--val-start", type=int, default=50, help="epoch mulai validasi"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="jumlah worker untuk dataloader"
    )
    parser.add_argument("--device", default="0", help="GPU device")
    parser.add_argument(
        "--cpu", action="store_true", help="force training to use CPU instead of GPU"
    )

    # Resume training
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        help="path ke model checkpoint untuk resume training",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="otomatis resume dari checkpoint terbaru jika tersedia",
    )

    # Logging
    parser.add_argument(
        "--run-name", default="custom-crowd-counting", help="nama run untuk logging"
    )
    parser.add_argument(
        "--wandb", default=0, type=int, help="apakah menggunakan wandb logging"
    )
    parser.add_argument("--seed", default=42, type=int, help="random seed")

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed, use_cpu=args.cpu)

    # Set device visibility only if not using CPU
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.strip()
    else:
        # Force CPU usage by hiding CUDA devices
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Validasi path dataset
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Dataset directory {args.data_dir} tidak ditemukan!")

    required_files = ["train.list", "valid.list"]
    for file in required_files:
        if not os.path.exists(os.path.join(args.data_dir, file)):
            raise FileNotFoundError(f"File {file} tidak ditemukan di {args.data_dir}")

    print("=== Custom Crowd Counting Training ===")
    print(f"Dataset path: {args.data_dir}")
    print(f"Crop size: {args.crop_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Max epochs: {args.max_epoch}")
    print(f"OT weight: {args.wot}, TV weight: {args.wtv}")

    trainer = Trainer(args)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
