import argparse
import json
import sys
from pathlib import Path

from .artifacts import ModelRegistry
from .predictor import predict
from .settings import REPO_ROOT
from .validation import validate_data


def _add_predict_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--cybench-root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "wrappers" / "data"),
        help="Directory to write the prediction CSV into.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YIELD-HUB SDK CLI")
    subparsers = parser.add_subparsers(dest="command")

    predict_parser = subparsers.add_parser("predict", help="Generate predictions from a trained checkpoint.")
    _add_predict_arguments(predict_parser)

    fetch_parser = subparsers.add_parser("fetch-model", help="Download a model checkpoint into the local HF cache.")
    fetch_parser.add_argument("--model-type", required=True)
    fetch_parser.add_argument("--country", required=True)
    fetch_parser.add_argument("--crop", required=True)
    fetch_parser.add_argument("--checkpoint-name", default=None)

    list_parser = subparsers.add_parser("list-models", help="List available model/crop/country combinations.")
    list_parser.add_argument("--model-type", default=None)

    validate_parser = subparsers.add_parser("validate-data", help="Validate a local CY-BENCH-style data folder.")
    validate_parser.add_argument("--crop", required=True)
    validate_parser.add_argument("--country", required=True)
    validate_parser.add_argument("--data-root", default=None)

    return parser


def _run_predict(args) -> None:
    predictions_df = predict(
        model_type=args.model_type,
        country=args.country,
        crop=args.crop,
        checkpoint_name=args.checkpoint_name,
        cybench_root=args.cybench_root,
        data_root=args.data_root,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model_type}_{args.crop}_{args.country}_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


def main() -> None:
    parser = _build_parser()

    legacy_predict_mode = len(sys.argv) > 1 and sys.argv[1] not in {"predict", "fetch-model", "list-models", "validate-data"}
    if legacy_predict_mode:
        args = parser.parse_args(["predict", *sys.argv[1:]])
    else:
        args = parser.parse_args()

    registry = ModelRegistry()

    if args.command == "fetch-model":
        checkpoint_path = registry.fetch_model(
            model_type=args.model_type,
            crop=args.crop,
            country=args.country,
            checkpoint_name=args.checkpoint_name,
        )
        print(checkpoint_path)
        return

    if args.command == "list-models":
        print(json.dumps(registry.list_models(model_type=args.model_type), indent=2))
        return

    if args.command == "validate-data":
        print(json.dumps(validate_data(data_root=args.data_root, crop=args.crop, country=args.country), indent=2))
        return

    _run_predict(args)
