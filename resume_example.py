#!/usr/bin/env python3
"""
Contoh script untuk menggunakan fitur resume training
"""

import os
import subprocess
import sys


def find_latest_checkpoint(checkpoint_dir="./ckpts"):
    """Find the latest checkpoint file"""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = []
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith("_ckpt.tar"):
                full_path = os.path.join(root, file)
                # Extract epoch number from filename
                try:
                    epoch_num = int(file.split("_")[0])
                    checkpoint_files.append((epoch_num, full_path))
                except ValueError:
                    continue

    if not checkpoint_files:
        return None

    # Return the path of the checkpoint with the highest epoch number
    latest_checkpoint = max(checkpoint_files, key=lambda x: x[0])
    return latest_checkpoint[1]


def run_training(resume_mode="auto", checkpoint_path=None, max_epoch=100):
    """
    Run training dengan berbagai opsi resume

    Args:
        resume_mode: "auto", "specific", "fresh", "weights_only"
        checkpoint_path: path ke checkpoint (untuk mode "specific" atau "weights_only")
        max_epoch: maksimum epoch training
    """

    # Base training arguments
    base_args = [
        "python",
        "train_custom.py",
        "--data-dir",
        "./data",  # Sesuaikan path dataset Anda
        "--dataset",
        "custom",
        "--arch",
        "FFNet",
        "--crop-size",
        "256",
        "--batch-size",
        "8",
        "--lr",
        "1e-5",
        "--eta_min",
        "1e-6",
        "--weight-decay",
        "1e-4",
        "--max-epoch",
        str(max_epoch),
        "--wot",
        "0.1",
        "--wtv",
        "0.01",
        "--reg",
        "10.0",
        "--num-of-iter-in-ot",
        "100",
        "--norm-cood",
        "0",
        "--val-epoch",
        "5",
        "--val-start",
        "10",
        "--num-workers",
        "4",
        "--device",
        "0",
        "--run-name",
        "resume-training-example",
        "--wandb",
        "0",
        "--seed",
        "42",
    ]

    # Add resume arguments based on mode
    if resume_mode == "auto":
        print("🔄 Mode: Auto-resume dari checkpoint terbaru")
        base_args.extend(["--auto-resume"])

    elif resume_mode == "specific":
        if not checkpoint_path:
            latest = find_latest_checkpoint()
            if latest:
                checkpoint_path = latest
                print(f"🔄 Mode: Resume dari checkpoint otomatis: {checkpoint_path}")
            else:
                print("❌ Tidak ada checkpoint ditemukan untuk mode specific")
                return False
        else:
            print(f"🔄 Mode: Resume dari checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint tidak ditemukan: {checkpoint_path}")
            return False

        base_args.extend(["--resume", checkpoint_path])

    elif resume_mode == "weights_only":
        if not checkpoint_path:
            print("❌ Path checkpoint diperlukan untuk mode weights_only")
            return False

        print(f"🏋️  Mode: Load weights only dari: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"❌ Model file tidak ditemukan: {checkpoint_path}")
            return False

        base_args.extend(["--resume", checkpoint_path])

    elif resume_mode == "fresh":
        print("🌟 Mode: Fresh training (tidak menggunakan checkpoint)")

    else:
        print(f"❌ Mode tidak dikenal: {resume_mode}")
        return False

    # Print command
    print(f"📝 Command: {' '.join(base_args)}")
    print("-" * 60)

    # Run training
    try:
        process = subprocess.run(base_args, check=True)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        return False
    except KeyboardInterrupt:
        print("🛑 Training interrupted by user")
        return False


def main():
    """Main function dengan command line interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Resume Training Example Script")
    parser.add_argument(
        "--mode",
        choices=["auto", "specific", "fresh", "weights_only"],
        default="auto",
        help="Resume mode",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint file (for specific/weights_only mode)",
    )
    parser.add_argument("--max-epoch", type=int, default=100, help="Maximum epochs")
    parser.add_argument(
        "--list-checkpoints", action="store_true", help="List available checkpoints"
    )

    args = parser.parse_args()

    if args.list_checkpoints:
        print("📋 Available checkpoints:")
        checkpoint_dir = "./ckpts"
        if os.path.exists(checkpoint_dir):
            for root, dirs, files in os.walk(checkpoint_dir):
                for file in files:
                    if file.endswith("_ckpt.tar") or file.endswith(".pth"):
                        full_path = os.path.join(root, file)
                        size_mb = os.path.getsize(full_path) / (1024 * 1024)
                        print(f"  - {full_path} ({size_mb:.1f} MB)")
        else:
            print("  Tidak ada checkpoint ditemukan")
        return

    # Run training
    success = run_training(
        resume_mode=args.mode, checkpoint_path=args.checkpoint, max_epoch=args.max_epoch
    )

    if success:
        print("\n🎉 Training selesai! Hasil tersimpan di direktori ./ckpts/")
    else:
        print("\n💥 Training gagal. Periksa error di atas.")
        sys.exit(1)


if __name__ == "__main__":
    # Contoh penggunaan:
    print("=" * 60)
    print("🚀 Resume Training Example Script")
    print("=" * 60)
    print()
    print("Contoh penggunaan:")
    print("  python resume_example.py --mode auto")
    print(
        "  python resume_example.py --mode specific --checkpoint ./ckpts/path/25_ckpt.tar"
    )
    print(
        "  python resume_example.py --mode weights_only --checkpoint ./ckpts/path/best_model_mae.pth"
    )
    print("  python resume_example.py --mode fresh")
    print("  python resume_example.py --list-checkpoints")
    print()

    main()
