import argparse
from pathlib import Path

from .predictor import predict
from .settings import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions using private Hugging Face checkpoints.")
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--cybench-root", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "wrappers" / "data"),
        help="Directory to write the prediction CSV into.",
    )
    args = parser.parse_args()

    predictions_df = predict(
        model_type=args.model_type,
        country=args.country,
        crop=args.crop,
        checkpoint_name=args.checkpoint_name,
        cybench_root=args.cybench_root,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model_type}_{args.crop}_{args.country}_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
