from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from lightning.pytorch import Trainer

from .artifacts import TRANSFORMER_MODELS, download_checkpoint, fetch_config_and_runid
from .settings import REPO_ROOT, configure_runtime_paths

configure_runtime_paths()

from linearLayer import create_model as create_linear_model
from modelconfig import LinearModelConfig, TSTModelConfig
from tstLayer import create_model as create_tst_model
from .data import build_datamodule


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


class Predictor:
    """High-level prediction API for end-user and dashboard workflows."""

    def __init__(self, repo_root: Optional[Path] = None, cybench_root: Optional[str] = None):
        self.repo_root = Path(repo_root) if repo_root else REPO_ROOT
        self.cybench_root = cybench_root
        configure_runtime_paths(cybench_root=cybench_root, repo_root=self.repo_root)

    def predict(
        self,
        model_type: str,
        country: str,
        crop: str,
        checkpoint_name: Optional[str] = None,
    ) -> pd.DataFrame:
        fetched = fetch_config_and_runid(
            model_type=model_type,
            country=country,
            crop=crop,
            repo_root=self.repo_root,
        )
        config_dict = fetched["config_dict"]
        resolved_checkpoint_name = checkpoint_name or fetched["run_id"]
        checkpoint_path = download_checkpoint(
            model_type=config_dict["model_type"],
            crop=config_dict["crop"],
            country=config_dict["country"],
            checkpoint_name=resolved_checkpoint_name,
            repo_root=self.repo_root,
        )

        model_config = create_model_config(config_dict)
        dm, _ = build_datamodule(model_config)

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
                        "checkpoint_name": resolved_checkpoint_name,
                        "model_type": model_type,
                        "country": country,
                        "crop": crop,
                    }
                )

        return pd.DataFrame(rows)


def predict(
    model_type: str,
    country: str,
    crop: str,
    checkpoint_name: Optional[str] = None,
    repo_root: Optional[Path] = None,
    cybench_root: Optional[str] = None,
) -> pd.DataFrame:
    predictor = Predictor(repo_root=repo_root, cybench_root=cybench_root)
    return predictor.predict(
        model_type=model_type,
        country=country,
        crop=crop,
        checkpoint_name=checkpoint_name,
    )
