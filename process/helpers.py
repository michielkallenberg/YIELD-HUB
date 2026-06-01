"""
--------------------
Author: XYZ
Description: Helper functions that I couldn't decide where to put. To-be-updated later.
Python version: 3.12.0
"""

import os
import re
import sys
import random
import hashlib
import logging
from typing import Union, Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import fields

import numpy as np
import pandas as pd

import torch
from lightning.pytorch import seed_everything

# Custom functions and classes
_helpers_dir = Path(__file__).parent.resolve()
sys.path.append(str(_helpers_dir.parent / 'architectures'))
from modelconfig import TSTModelConfig, LinearModelConfig


def verify_parameters(crop, model, country):
    """
    Verifies if the pipeline for selected crop, model, and country is implemented.
    """
    assert crop in ["maize", "wheat"]
    assert model in ["ridge", "svr", "rf", "gb", "mlp"]

    assert crop in ["maize", "wheat"]
    assert model in ["ridge", "svr", "rf", "gb", "mlp"]

    if crop == "maize":
        assert country in ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EL', 'ES', 'FR', 'HR', 'HU', 'IT', 'LT', 'NL', 'PL', 'PT', 'RO', 'SE', 'all']
    else:
        assert country in ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LV', 'NL', 'PL', 'PT', 'RO', 'SE', 'all']

def select_country(crop, country):
    """
    A function to collect all the countries if country is set to "all". Otherwise,
    it just returns [country]
    """
    if country == "all":
        if crop == "maize":
            country = ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EL', 'ES', 'FR', 'HR', 
                                'HU', 'IT', 'LT', 'NL', 'PL', 'PT', 'RO', 'SE']
        else:
            country = ['AT', 'BE', 'BG', 'CZ', 'DE', 'DK', 'EE', 'EL', 'ES', 'FI', 
                                'FR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LV', 'NL', 'PL', 'PT', 
                                'RO', 'SE']
    else:
        country = [country]
    return country

def seed_uniformly(seed):
    # Setting seed value for reproducibility    
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    seed_everything(seed)

def generate_checkpoint_name(args) -> str:
    """
    Generate a descriptive checkpoint filename with all hyperparameters.

    Args:
        args: ArgumentParser namespace with all hyperparameters

    Returns:
        Descriptive checkpoint filename without extension
    """
    base_name = f"{args.crop}_{args.country}"

    # Core hyperparameters
    hyperparams = [
        f"model:{args.model_type}",
        f"agg:{args.aggregation}",
        f"epochs:{args.epochs}",
        f"lr:{args.lr}",
        f"wd:{args.weight_decay}",
        f"batch:{args.batch_size}",
        f"seed:{args.seed}",
    ]

    # Feature flags
    if args.use_sota_features:
        hyperparams.append("sota")
    if args.include_spatial_features:
        hyperparams.append("spatial")
    if args.use_residual_trend:
        hyperparams.append("trend")
    hyperparams.append(f"lag:{args.lag_years}")

    # Combine all parts
    parts = [base_name] + hyperparams
    name = "_".join(parts)

    # Protect against OS filename length limits (typically 255 bytes)
    # Leave room for epoch/val_loss suffix (~20 chars) and .ckpt extension (~5 chars)
    MAX_NAME_LENGTH = 230
    if len(name.encode('utf-8')) > MAX_NAME_LENGTH:
        # Use hash for long names while preserving core info
        short_hash = hashlib.md5(name.encode('utf-8')).hexdigest()[:8]
        original_name = name
        hashed_name = f"{base_name}_run_{short_hash}"
        logging.warning(
            f"Checkpoint name exceeded {MAX_NAME_LENGTH} chars and was hashed:\n"
            f"  Original ({len(name.encode('utf-8'))} chars): {original_name}\n"
            f"  Hashed to: {hashed_name}\n"
            f"  To recover params: check wandb run config or args log"
        )
        name = hashed_name

    return name

def save_test_results_to_csv(
    config: Union[TSTModelConfig, LinearModelConfig],
    test_results: Dict[str, float],
    test_years: List[int],
    run_id: str,
    timestamp: str
):
    """Save test results to CSV files with per-year metrics."""
    os.makedirs(config.results_dir, exist_ok=True)
    
    base_data = {'timestamp': timestamp, 'run_id': run_id}
    # Exclude results_dir from CSV columns (it's metadata, not a model hyperparameter)
    for field in fields(config):
        if field.name != 'results_dir':
            base_data[field.name] = getattr(config, field.name)
    
    for metric in ['nrmse', 'mape', 'r2', 'rmse', 'mae', 'mse', 'smape']:
        csv_path = os.path.join(config.results_dir, f'{metric}.csv')
        year_columns = {str(year): test_results.get(f'{metric}_{year}', None) for year in test_years}
        year_columns['overall'] = test_results.get(f'{metric}_overall', None)
        row_data = {**base_data, **year_columns}

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, on_bad_lines='skip')
            new_df = pd.DataFrame([row_data])
            # pandas automatically aligns columns and adds new ones
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame([row_data])

        # Enforce consistent column order: metadata -> config -> years (sorted) -> overall
        def get_column_order(cols):
            metadata_cols = ['timestamp', 'run_id']
            year_cols = sorted([c for c in cols if c.isdigit() and len(c) == 4])
            other_cols = [c for c in cols if c not in metadata_cols + year_cols + ['overall']]
            return metadata_cols + other_cols + year_cols + ['overall']

        df = df[get_column_order(df.columns)]
        df.to_csv(csv_path, index=False)
        print(f"[CSV Results] Saved {metric} results to {csv_path}")


def load_best_hps(results_file: str) -> Dict[str, Any]:
    """Load best hyperparameters from a single HPO results file."""
    with open(results_file, 'r') as f:
        content = f.read()

    match = re.search(
        r'BEST TRIAL:.*?Hyperparameters:(.*?)ALL TRIALS',
        content, re.DOTALL
    )

    if not match:
        return {}

    hps = {}
    for line in match.group(1).split('\n'):
        if ':' in line and not line.strip().startswith('='):
            key, val = line.split(':', 1)
            key, val = key.strip(), val.strip()
            try:
                hps[key] = int(val) if '.' not in val else float(val)
            except ValueError:
                hps[key] = val
    return hps


def load_all_hps(results_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Load all HPO results from directory.
    Returns dict with keys like 'patchtst-AO-maize' -> hyperparameters
    """
    results_path = Path(results_dir)
    all_hps = {}

    for file_path in results_path.glob("**/*HPO_results*.txt"):
        folder_name = file_path.parent.parent.name
        parts = folder_name.replace('hpo_', '').split('_')

        model = parts[0]
        crop = parts[-1]
        region = '_'.join(parts[1:-1])

        key = f"{model}-{region}-{crop}"
        hps = load_best_hps(str(file_path))

        if hps:
            all_hps[key] = hps

    return all_hps


def get_hps_for_model_crop_region(
    results_dir: str,
    model: str,
    crop: str,
    region: str
) -> Dict[str, Any]:
    """Get hyperparameters for a specific model, crop, and region combination."""
    key = f"{model}-{region}-{crop}"
    all_hps = load_all_hps(results_dir)
    return all_hps.get(key, {})
