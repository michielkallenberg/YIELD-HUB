import argparse
import ast
import os
import sys
from pathlib import Path

import pandas as pd
import torch

from lightning.pytorch import Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "process"))
sys.path.append(str(REPO_ROOT / "architectures"))

from cybench.config import KEY_YEAR
from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset
from linearLayer import create_model as create_linear_model
from loadData import DailyCYBenchSeqDataModule, calculate_fixed_split
from modelconfig import LinearModelConfig, TSTModelConfig
from tstLayer import create_model as create_tst_model

try:
    from .hf_artifacts import download_checkpoint, fetch_config_and_runid
except ImportError:
    from hf_artifacts import download_checkpoint, fetch_config_and_runid


TRANSFORMER_MODELS = {
    "autoformer",
    "patchtst",
    "tsmixer",
    "informer",
    "tst",
    "itransformer",
    "timexer",
    "timesnet",
}

TST_ONLY_ARGS = {"use_revin"}


def create_model_config(config_dict):
    if config_dict["model_type"] in TRANSFORMER_MODELS:
        return TSTModelConfig(**config_dict)

    filtered = {k: v for k, v in config_dict.items() if k not in TST_ONLY_ARGS}
    return LinearModelConfig(**filtered)


def create_model(model_config):
    if model_config.model_type in TRANSFORMER_MODELS:
        return create_tst_model(model_config)
    return create_linear_model(model_config)


def generate_predictions_wrapper(model_type, country, crop, checkpoint_name=None):
    fetched = fetch_config_and_runid(
        model_type=model_type,
        country=country,
        crop=crop,
        repo_root=REPO_ROOT,
    )
    config_dict = ast.literal_eval(fetched["config"])
    checkpoint_name = checkpoint_name or fetched["run_id"]
    checkpoint_path = download_checkpoint(
        model_type=config_dict["model_type"],
        crop=config_dict["crop"],
        country=config_dict["country"],
        checkpoint_name=checkpoint_name,
        repo_root=REPO_ROOT,
    )

    model_config = create_model_config(config_dict)

    df_y, dfs_x = load_dfs_crop(config_dict["crop"], [config_dict["country"]])
    if df_y is None or len(df_y) == 0:
        raise ValueError(f"No data found for {config_dict['crop']}-{config_dict['country']}")

    ds = CYDataset(config_dict["crop"], df_y, dfs_x)
    all_years = sorted({ds[i][KEY_YEAR] for i in range(len(ds))})
    fixed_splits = calculate_fixed_split(
        all_years,
        test_years=config_dict["test_years"],
        val_years=2,
    )

    dm = DailyCYBenchSeqDataModule(model_config)
    dm.setup(
        train_years=fixed_splits["train_years"],
        val_years=fixed_splits["val_years"],
        test_years=fixed_splits["test_years"],
    )

    model = create_model(model_config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.datamodule = dm
    model.trainer = trainer

    rows = []
    for batch in dm.test_dataloader():
        with torch.no_grad():
            result = model.predict(batch)

        for i in range(len(result["predictions"])):
            actual = result["targets"][i].item()
            pred = result["predictions"][i].item()
            error = pred - actual
            rows.append(
                {
                    "adm_id": result["adm_ids"][i],
                    "year": int(result["years"][i].item()),
                    "lat": float(result["lats"][i].item()),
                    "lon": float(result["lons"][i].item()),
                    "predicted_yield": pred,
                    "actual_yield": actual,
                    "error": error,
                    "abs_error": abs(error),
                    "pct_error": (error / actual * 100.0) if actual else None,
                    "checkpoint_name": checkpoint_name,
                    "model_type": model_type,
                    "country": country,
                    "crop": crop,
                }
            )

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate predictions using private Hugging Face checkpoints.")
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--country", required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "wrappers" / "data"),
        help="Directory to write the prediction CSV into.",
    )
    args = parser.parse_args()

    predictions_df = generate_predictions_wrapper(
        model_type=args.model_type,
        country=args.country,
        crop=args.crop,
        checkpoint_name=args.checkpoint_name,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.model_type}_{args.crop}_{args.country}_predictions.csv"
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
