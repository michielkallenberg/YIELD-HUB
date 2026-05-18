import ast
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from huggingface_hub import hf_hub_download

from .settings import REPO_ROOT, get_hf_token


TRANSFORMER_REPO_ID = "Ambrosia2024/yield-transformers-cybench"
LINEAR_REPO_ID = "Ambrosia2024/yield-linear-cybench"

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
LINEAR_MODELS = {
    "nlinear",
    "dlinear",
    "xlinear",
    "rlinear",
    "olinear",
}
SUPPORTED_MODELS = sorted(TRANSFORMER_MODELS | LINEAR_MODELS)


def get_repo_id(model_type: str) -> str:
    return TRANSFORMER_REPO_ID if model_type in TRANSFORMER_MODELS else LINEAR_REPO_ID


def load_baseline_table(model_type: str, repo_root: Optional[Path] = None) -> pd.DataFrame:
    token = get_hf_token(repo_root or REPO_ROOT)
    repo_id = get_repo_id(model_type)
    csv_path = hf_hub_download(
        repo_id=repo_id,
        filename="config-and-results.csv",
        token=token,
        repo_type="model",
    )
    return pd.read_csv(csv_path)


def download_checkpoint(
    model_type: str,
    crop: str,
    country: str,
    checkpoint_name: str,
    repo_root: Optional[Path] = None,
) -> Path:
    token = get_hf_token(repo_root or REPO_ROOT)
    repo_id = get_repo_id(model_type)
    filename = f"{model_type}/{crop}/{country}/{checkpoint_name}.ckpt"
    checkpoint_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        repo_type="model",
    )
    return Path(checkpoint_path)


def fetch_config_and_runid(
    model_type: str,
    country: str,
    crop: str,
    repo_root: Optional[Path] = None,
) -> Dict:
    df = load_baseline_table(model_type, repo_root or REPO_ROOT)
    matched = df[
        (df["model_type"] == model_type)
        & (df["country"] == country)
        & (df["crop"] == crop)
    ]
    if matched.empty:
        raise ValueError(f"No config found for model={model_type}, crop={crop}, country={country}")

    row = matched.iloc[0]
    return {
        "config_text": row["config"],
        "config_dict": ast.literal_eval(row["config"]),
        "run_id": row["run_id"],
        "row": row,
    }
