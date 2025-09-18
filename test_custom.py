import argparse
import torch
import os
import numpy as np
import pandas as pd
import datasets.crowd as crowd
import torch.nn.functional as F
from Networks import FFNet
from PIL import Image


def test_custom_dataset(args):
    """Test model pada custom dataset dan generate submission file"""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = FFNet.FFNet()
    model.to(device)
    model.load_state_dict(torch.load(args.model_path, device))
    model.eval()

    # Jika ada validation set, evaluasi dulu
    if args.evaluate_validation:
        print("Evaluating on validation set...")
        val_dataset = crowd.CustomDataset(
            args.data_path, args.crop_size, 8, method="valid"
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, 1, shuffle=False, num_workers=1, pin_memory=True
        )

        val_errors = []
        val_results = []

        for inputs, count, name in val_dataloader:
            pred_count = predict_single_image(model, inputs, device, args)
            error = count[0].item() - pred_count
            val_errors.append(error)
            val_results.append([name[0], count[0].item(), pred_count, error])
            print(
                f"Val - Image: {name[0]}, GT: {count[0].item()}, Pred: {pred_count:.2f}, Error: {error:.2f}"
            )

        val_errors = np.array(val_errors)
        mae = np.mean(np.abs(val_errors))
        mse = np.sqrt(np.mean(np.square(val_errors)))
        print(f"Validation Results - MAE: {mae:.2f}, MSE: {mse:.2f}")

        # Save validation results
        val_df = pd.DataFrame(
            val_results, columns=["image_name", "ground_truth", "prediction", "error"]
        )
        val_df.to_csv(
            os.path.join(args.output_dir, "validation_results.csv"), index=False
        )

    # Test pada test set
    print("Predicting on test set...")
    test_dataset = crowd.CustomDataset(args.data_path, args.crop_size, 8, method="test")
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 1, shuffle=False, num_workers=1, pin_memory=True
    )

    predictions = []

    for inputs, _, name in test_dataloader:
        pred_count = predict_single_image(model, inputs, device, args)
        predictions.append([name[0], int(round(pred_count))])
        print(f"Test - Image: {name[0]}, Predicted Count: {int(round(pred_count))}")

    # Generate submission file
    submission_df = pd.DataFrame(predictions, columns=["image_name", "count"])
    submission_path = os.path.join(args.output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission file saved to: {submission_path}")
    print(f"Total test images: {len(predictions)}")
    print(f"Average predicted count: {np.mean([p[1] for p in predictions]):.2f}")


def predict_single_image(model, inputs, device, args):
    """Prediksi untuk satu gambar menggunakan patch-based inference"""

    with torch.no_grad():
        inputs = inputs.to(device)
        crop_imgs, crop_masks = [], []
        b, c, h, w = inputs.size()
        rh, rw = args.crop_size, args.crop_size

        # Split image into patches
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                crop_imgs.append(inputs[:, :, gis:gie, gjs:gje])
                mask = torch.zeros([b, 1, h, w]).to(device)
                mask[:, :, gis:gie, gjs:gje].fill_(1.0)
                crop_masks.append(mask)

        crop_imgs, crop_masks = map(
            lambda x: torch.cat(x, dim=0), (crop_imgs, crop_masks)
        )

        # Process patches in batches
        crop_preds = []
        nz, bz = crop_imgs.size(0), args.batch_size
        for i in range(0, nz, bz):
            gs, gt = i, min(nz, i + bz)
            crop_pred, _ = model(crop_imgs[gs:gt])

            _, _, h1, w1 = crop_pred.size()
            crop_pred = (
                F.interpolate(
                    crop_pred,
                    size=(h1 * 8, w1 * 8),
                    mode="bilinear",
                    align_corners=True,
                )
                / 64
            )
            crop_preds.append(crop_pred)

        crop_preds = torch.cat(crop_preds, dim=0)

        # Reconstruct full image prediction
        idx = 0
        pred_map = torch.zeros([b, 1, h, w]).to(device)
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
                pred_map[:, :, gis:gie, gjs:gje] += crop_preds[idx]
                idx += 1

        # Average overlapping areas
        mask = crop_masks.sum(dim=0).unsqueeze(0)
        outputs = pred_map / mask

        return torch.sum(outputs).item()


def main():
    parser = argparse.ArgumentParser(description="Test Custom Crowd Counting Model")

    parser.add_argument(
        "--data-path",
        type=str,
        default="/path/to/your/converted/dataset",
        help="path ke dataset yang sudah dikonversi",
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="path ke model checkpoint (.pth)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="direktori untuk menyimpan hasil",
    )
    parser.add_argument("--device", default="0", help="GPU device")
    parser.add_argument(
        "--batch-size", type=int, default=8, help="batch size untuk inference"
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
        help="ukuran crop (harus sama dengan training)",
    )
    parser.add_argument(
        "--evaluate-validation",
        action="store_true",
        help="evaluasi pada validation set juga",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Validate paths
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset path {args.data_path} tidak ditemukan!")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path {args.model_path} tidak ditemukan!")

    test_custom_dataset(args)


if __name__ == "__main__":
    main()
