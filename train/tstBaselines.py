
# -*- coding: utf-8 -*-
"""
--------------------
Author: XYZ
Description: A time series transformers based crop yield prediction framework that combines state-of-the-art time architectures with agricultural domain knowledge. 
Python version: 3.12.0

--------------------
Architecture overview: This script implements a unified interface to seven transformer-based time series architectures from the HuggingFace Transformers library and nixtla implementations:
    • Autoformer: Auto-correlation mechanism for long-term dependency discovery (https://arxiv.org/abs/2106.13008)
    • PatchTST: Patch time series into sub-sequences for efficient processing (https://arxiv.org/pdf/2211.14730)
    • TSMixer: All-MLP architecture with mixing layers (via PatchTSMixer) (https://arxiv.org/pdf/2303.06053)
    • Informer: ProbSparse self-attention for long sequences (https://arxiv.org/pdf/2012.07436)
    • TST: Vanilla time series transformer with canonical attention (https://arxiv.org/pdf/2010.02803)
    • iTransformer: Inverted transformer treating channels as tokens for cross-variable dependencies (https://arxiv.org/abs/2310.06625)
    • TimeXer: Cross-attention transformer with endogenous patching and exogenous inverted embedding (https://arxiv.org/abs/2402.19072)

All models share a common base class (BaseTimeSeriesModel) ensuring consistent training, evaluation, and inference interfaces.

--------------------
Data pipeline: 
The script processes agricultural data through multiple stages:

1. INPUT FEATURES
   - Weather: tmin, tmax, tavg, precipitation, radiation, (optional: cwb)
   - Remote Sensing: NDVI, FPAR, SSM, RSM
   - Soil Properties: Available water capacity, organic carbon, pH, texture
   - Location: Country, state, latitude, longitude
   - Crop Calendar: Start/end of season with cyclic encoding (sin/cos)
   - Temporal Encoding: Fourier features (sin/cos of day-of-year, month)
   - Historical Lags: Yield from previous years (1-2 years, configurable)

2. TEMPORAL AGGREGATION
   - daily:   365 time steps (raw daily data)
   - weekly:  52 time steps  (weekly averages, Monday-start)
   - dekad:   36 time steps  (10-day periods, standard in ag monitoring)

3. GROWTH-STAGE PROCESSING
   Weather data is masked to only include observations between crop start-of-season (SOS) and end-of-season (EOS) dates, ensuring the model focuses on the growth period.

4. NORMALIZATION
   - Time series: Per-feature min-max scaling
   - Static features: Mean-centering and scaling
   - Targets: Normalized to zero mean, unit variance

--------------------
Other optional/advanced features:

1. RESIDUAL TREND MODELING (--use_residual_trend)
   Uses Mann-Kendall trend detection to identify significant linear trends in training yields, then models residuals (yield - trend) to improve forecasting for datasets with strong yield progression over time.

2. RECURSIVE LAG PREDICTION (--use_recursive_lags)
   For true out-of-sample testing: uses predicted yields as lag features during test set evaluation instead of ground truth, preventing data leakage.

3. SPATIAL FEATURES (--include_spatial_features)
    Adds explicit latitude/longitude as static features (beyond location embeddings).

4. FEATURE ABLATION TOGGLES
   --use_cwb_feature: Include climate water balance
   --drop_tavg: Drop average temperature if dataset computes it as (tmin+tmax)/2
   --use_gdd : Adds cumulative GDD as a time series channel
   --use_heat_stress_days: Adds heat/frost/dry stress day counts as static features
   --use_rue: Adds RUE (Radiation Use Efficiency) index as a time series channel
   --use_farquhar: Adds Farquhar photosynthesis proxy as a time series channel

-------------
Training workflow:
1. Data module handles train/val/test splits and normalization
2. Lightning trainer manages GPU distribution, mixed precision, checkpoints
3. Early stopping on validation loss with patience monitoring
4. Model checkpointing saves best model based on validation loss
5. WandB logging tracks metrics, hyperparameters, and artifacts

Evaluation metrics:
    • MSE, MAE, RMSE: Standard error metrics
    • R²: Coefficient of determination
    • MAPE, SMAPE: Percentage-based error metrics
    • NRMSE: Normalized RMSE (test set only)

------
Output generated:
    Checkpoints: Saved to checkpoints/ with descriptive filenames
    Results CSV: Detailed predictions with actuals, errors, metadata
    WandB: Full experiment tracking with metrics, parameters, artifacts
    
--------------------
Hyperparameters:
Key hyperparameters:
    --lr:              Learning rate (default: 1e-4)
    --weight_decay:    L2 regularization (default: 1e-5)
    --batch_size:      Training batch size (default: 16)
    --lag_years:       Historical yield lags (1 or 2, default: 1)
    --aggregation:     Temporal resolution (daily/weekly/dekad, default: dekad)
    --seed:            Random seed for reproducibility (default: 42)

--------------
Usage:
# Basic training with Autoformer: python tstBaselines.py --crop maize --country NL --model_type autoformer --epochs 50 --aggregation daily

# Use all SOTA features (Fourier encoding + residual trend + recursive lags)
    python tstBaselines.py --crop maize --country NL --model_type tst --use_sota_features --use_residual_trend --use_recursive_lags --use_cwb_feature --aggregation daily

# Quick test run (5 epochs)
    python tstBaselines.py --crop wheat --country NL --model_type timesnet --epochs 5 --aggregation daily --lag_years 0 --test_years 5 --results_dir checkpoints-test/results --wandb_project test-and-delete-later

# HPO with Optuna
python tstBaselines.py --crop maize --country NL --model_type patchtst --find_HPO --n_trials 10 --epochs 3 --hpo_objective multi --hpo_results_file checkpoints-test/results/HPO/octuna_file.txt --hpo_study_name test-and-delete-later

------------
Core dependencies:
    - torch>=2.0: PyTorch for model implementation
    - lightning: PyTorch Lightning for training framework
    - transformers>=4.30: Time series model architectures
    - torchmetrics: Evaluation metrics
    - wandb: Experiment tracking
    - pymannkendall: Trend detection for residual modeling
    - pandas, numpy: Data manipulation

Internal dependencies:
    - cybench.datasets: Agricultural data loading utilities
    - cybench.config: Domain constants and configurations
"""

# %% Loading libraries
import os
import sys
import random
import argparse
import logging
import uuid
import csv

from datetime import datetime
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna

# Configure module-level logger BEFORE any imports that might use it
logger = logging.getLogger(__name__)

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

# CY-BENCH Dependencies
from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset
from cybench.config import (LOCATION_PROPERTIES, SOIL_PROPERTIES,
    FORECAST_LEAD_TIME, KEY_LOC, KEY_YEAR, KEY_TARGET, KEY_DATES, KEY_CROP_SEASON,
    CROP_CALENDAR_DATES
)

# Loading custom functions and classes
sys.path.append('../process/')
from helpers import generate_checkpoint_name, save_test_results_to_csv
from validateModel import print_metrics_table
from loadData import calculate_fixed_split, DailyCYBenchSeqDataModule

sys.path.append('../architectures/')
from modelconfig import TSTModelConfig
from tstLayer import create_model

# Setting precision
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 8:  # Ampere or newer
        torch.set_float32_matmul_precision('high')
        logger.info(f"Enabled high matmul precision (GPU capability {capability})")
    else:
        logger.info(f"Keeping default matmul precision (GPU capability {capability} < 8.0)")
else:
    logger.info("Running on CPU, matmul precision setting has no effect")

# Boilerplate code
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CY-BENCH Time Series Yield Forecasting with Expanding Window CV")
    parser.add_argument('--crop', default="maize")
    parser.add_argument('--country', default="NL")
    parser.add_argument('--model_type', default="autoformer",
                        choices=['autoformer', 'patchtst', 'tsmixer', 'informer', 'tst', 'itransformer', 'timexer', 'timesnet'])
    parser.add_argument('--aggregation', default="dekad",
                        choices=['daily', 'weekly', 'dekad'])
    parser.add_argument('--use_sota_features', action='store_true')
    parser.add_argument('--include_spatial_features', action='store_true')
    parser.add_argument('--lag_years', type=int, default=1, choices=[0, 1, 2],
                        help='Number of lagged yield years (max 2, default: 1)')
    parser.add_argument('--use_recursive_lags', action='store_true',
                        help='Use predicted yields as lags during testing for true out-of-sample evaluation '
                             '(default: False, uses observed test-set yields as lags)')
    # Domain feature engineering flags
    parser.add_argument('--use_gdd', action='store_true',
                        help='Add cumulative GDD as a time series channel. '
                             'Uses crop-specific base/upper thresholds from cybench.config. '
                             'GDD = max(min(Tavg, Tupper) - Tbase, 0), then cumsum.')
    parser.add_argument('--use_heat_stress_days', action='store_true',
                        help='Add heat/frost/dry stress day counts as static features. '
                             'Captures threshold exceedance events missed by averages.')
    parser.add_argument('--use_rue', action='store_true',
                        help='Add RUE (Radiation Use Efficiency) index as a time series channel. '
                             'RUE = cumPAR * T_stress * W_stress. Experimental.')
    parser.add_argument('--use_farquhar', action='store_true',
                        help='Add Farquhar photosynthesis proxy as a time series channel. '
                             'Based on FvCB C3 model. Seasonal-scale approximation only.')
    parser.add_argument('--load_checkpoint', default=None,
                        help='Path to checkpoint to load for fine-tuning')
    parser.add_argument('--save_checkpoint_dir', default='checkpoints-test',
                        help='Directory to save model checkpoints (default: checkpoints/)')
    parser.add_argument('--results_dir', default='checkpoints/results',
                        help='Directory to save CSV results (default: checkpoints/results/)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_residual_trend', action='store_true')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='DataLoader workers. Default: auto-calculated as min(cpu_count//4, 8). '
                             'For 3 concurrent scripts, this balances CPU usage. Set manually to override.')
    parser.add_argument('--test_years', type=int, default=3,
                        help='Number of years for final test set (default: 3)')
    # Feature configuration flags for ablation studies
    parser.add_argument('--use_cwb_feature', action='store_true',
                        help='Include crop water balance (cwb) as a feature. '
                             'Note: cwb is derived from prec and ET0 (depends on temperature), '
                             'so it may be redundant with existing weather features.')
    parser.add_argument('--drop_tavg', action='store_true',
                        help='Drop tavg feature if dataset computes it as (tmin+tmax)/2, '
                             'which carries no additional information beyond tmin/tmax.')
    parser.add_argument('--lr_decay_every', type=int, default=None,
                        help='Decay learning rate by half every N epochs (default: None, no decay)')
    parser.add_argument('--wandb_project', default=None,
                        help='Custom WandB project name (default: CYBENCH-LSTF-AAAI2027)')
    parser.add_argument('--wandb_run_name', default=None,
                        help='Custom WandB run name (default: model_type-crop-country)')
    # PatchTST-specific hyperparameters (only used when model_type='patchtst')
    parser.add_argument('--patchtst_d_model', type=int, default=64,
                        help='PatchTST: dimension of the transformer hidden states (default: 64)')
    parser.add_argument('--patchtst_num_attention_heads', type=int, default=4,
                        help='PatchTST: number of parallel attention heads (default: 4)')
    parser.add_argument('--patchtst_ffn_dim', type=int, default=256,
                        help='PatchTST: dimension of the feed-forward network (default: 256)')
    parser.add_argument('--patchtst_num_layers', type=int, default=3,
                        help='PatchTST: number of transformer encoder layers (default: 3)')
    parser.add_argument('--patchtst_dropout', type=float, default=0.1,
                        help='PatchTST: dropout probability for regularization (default: 0.1)')
    # Optuna HPO arguments
    parser.add_argument('--find_HPO', action='store_true',
                        help='Enable Optuna hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--hpo_storage', type=str, default=None,
                        help='Optuna storage URL for distributed optimization (e.g., sqlite:///optuna.db)')
    parser.add_argument('--hpo_study_name', type=str, default=None,
                        help='Optuna study name (default: auto-generated)')
    parser.add_argument('--hpo_results_file', type=str, default=None,
                        help='Path to save HPO results text file (default: auto-generated in results_dir)')
    parser.add_argument('--hpo_objective', type=str, default='nrmse',
                        choices=['nrmse', 'r2', 'multi'],
                        help='Optimization objective: nrmse (minimize), r2 (maximize), or multi (both)')
    args = parser.parse_args()

    # Dynamically set num_workers if not specified
    if args.num_workers is None:
        cpu_count = os.cpu_count() or 1
        # For 3 concurrent scripts: divide by 4, cap at 8 for balance
        # This gives good parallelism without overwhelming the system
        args.num_workers = min(cpu_count // 4, 8)
        print(f"[Auto-config] Setting num_workers={args.num_workers} based on {cpu_count} CPU cores")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Generate unique run identifier and timestamp for CSV tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(uuid.uuid4())[:8]  # Short unique identifier

    print(f"\n{'=' * 70}")
    print(f"CY-BENCH  |  {args.model_type.upper()}  |  {args.crop}-{args.country}  "
          f"|  {args.aggregation.upper()}")
    print(f"  SOTA={args.use_sota_features}  Spatial={args.include_spatial_features}  "
          f"Lag={args.lag_years}")
    print(f"  Domain features: GDD={args.use_gdd}  HeatStress={args.use_heat_stress_days}  "
          f"RUE={args.use_rue}  Farquhar={args.use_farquhar}")
    print(f"  TestYears={args.test_years}")
    print(f"  lr={args.lr}  wd={args.weight_decay}  epochs={args.epochs}  "
          f"batch={args.batch_size}  seed={args.seed}")
    print(f"{'=' * 70}\n")

    # Create LR scheduler lambda if requested
    lr_scheduler_lambda = None
    if args.lr_decay_every is not None:
        def lr_scheduler_lambda(epoch):
            decay_factor = args.lr_decay_every
            decay_steps = epoch // decay_factor
            return 0.5 ** decay_steps

    config = TSTModelConfig(
        crop=args.crop, country=args.country,
        model_type=args.model_type, aggregation=args.aggregation,
        use_sota_features=args.use_sota_features,
        include_spatial_features=args.include_spatial_features,
        lag_years=args.lag_years,
        load_checkpoint=args.load_checkpoint,
        seed=args.seed, batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        test_years=args.test_years,
        use_residual_trend=args.use_residual_trend,
        use_cwb_feature=args.use_cwb_feature,
        drop_tavg=args.drop_tavg,
        use_recursive_lags=args.use_recursive_lags,
        use_gdd=args.use_gdd,
        use_heat_stress_days=args.use_heat_stress_days,
        use_rue=args.use_rue,
        use_farquhar=args.use_farquhar,
        results_dir=args.results_dir,
        lr_scheduler_lambda=lr_scheduler_lambda,
        patchtst_d_model=args.patchtst_d_model,
        patchtst_num_attention_heads=args.patchtst_num_attention_heads,
        patchtst_ffn_dim=args.patchtst_ffn_dim,
        patchtst_num_layers=args.patchtst_num_layers,
        patchtst_dropout=args.patchtst_dropout,
    )

    # No longer mutating global WEATHER_FEATURES - using config.weather_features property instead
    # This prevents timing issues where model init captures the wrong value
    print(f"[Feature Config] Weather features: {config.weather_features}")
    print(f"[Feature Config] Total time series vars ({len(config.time_series_vars)}): {config.time_series_vars}")

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.save_checkpoint_dir, exist_ok=True)
    print(f"\n[Checkpoint Config]")
    print(f"  Save directory: {args.save_checkpoint_dir}")
    if args.load_checkpoint:
        print(f"  Load checkpoint: {args.load_checkpoint}")

    # Get available years for this country-crop combination
    df_y, dfs_x = load_dfs_crop(config.crop, [config.country])
    if df_y is None or len(df_y) == 0:
        print(f"[ERROR] No data for {config.crop}-{config.country}")
        sys.exit(1)

    ds = CYDataset(config.crop, df_y, dfs_x)
    all_years = sorted(set([ds[i][KEY_YEAR] for i in range(len(ds))]))
    print(f"[Data] Available years: {all_years}")

    # Calculate fixed train/val/test split
    fixed_splits = calculate_fixed_split(
        all_years,
        test_years=args.test_years,
        val_years=2
    )

    print(f"\n[Split Config - Fixed]")
    print(f"  Total years: {fixed_splits['total_years']}")
    print(f"  Train years ({len(fixed_splits['train_years'])}): {sorted(fixed_splits['train_years'])}")
    print(f"  Val years ({len(fixed_splits['val_years'])}): {sorted(fixed_splits['val_years'])}")
    print(f"  Test years ({len(fixed_splits['test_years'])}): {sorted(fixed_splits['test_years'])}")

    # ==================== OPTUNA HPO INTEGRATION ====================
    if args.find_HPO:
        print(f"\n{'=' * 70}")
        print(f"OPTUNA HYPERPARAMETER OPTIMIZATION")
        print(f"{'=' * 70}\n")

        # Auto-generate HPO results file path if not provided
        if args.hpo_results_file is None:
            hpo_results_dir = os.path.join(args.results_dir, "HPO")
            os.makedirs(hpo_results_dir, exist_ok=True)
            hpo_results_file = os.path.join(hpo_results_dir,
                f"{args.model_type}_{args.crop}_{args.country}_HPO_results_{timestamp}.txt")
        else:
            hpo_results_file = args.hpo_results_file
            # Create directory if it doesn't exist
            hpo_results_dir = os.path.dirname(hpo_results_file)
            if hpo_results_dir:  # Only create if there's a directory component
                os.makedirs(hpo_results_dir, exist_ok=True)

        # Auto-generate study name if not provided
        if args.hpo_study_name is None:
            study_name = f"{args.model_type}_{args.crop}_{args.country}_{args.aggregation}"
        else:
            study_name = args.hpo_study_name

        print(f"[HPO Config]")
        print(f"  Objective: {args.hpo_objective}")
        print(f"  Trials: {args.n_trials}")
        print(f"  Study name: {study_name}")
        print(f"  Results file: {hpo_results_file}")
        print(f"  Storage: {args.hpo_storage if args.hpo_storage else 'In-memory'}")

        def optuna_objective(trial):
            """Optuna objective function for hyperparameter optimization"""

            # Suggest hyperparameters based on model type
            if args.model_type == 'patchtst':
                patchtst_d_model = trial.suggest_categorical('patchtst_d_model', [32, 64, 128, 256])
                patchtst_num_attention_heads = trial.suggest_categorical('patchtst_num_attention_heads', [2, 4, 8, 16])
                patchtst_ffn_dim = trial.suggest_categorical('patchtst_ffn_dim', [128, 256, 512, 1024])
                patchtst_num_layers = trial.suggest_categorical('patchtst_num_layers', [2, 3, 4, 6, 8])
                patchtst_dropout = trial.suggest_float('patchtst_dropout', 0.0, 0.5)

                # Shared hyperparameters
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
                lr = trial.suggest_float('lr', 5e-5, 5e-4, log=True)
                weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)

                # Create config with suggested parameters
                hpo_config = TSTModelConfig(
                    crop=args.crop, country=args.country,
                    model_type=args.model_type, aggregation=args.aggregation,
                    use_sota_features=args.use_sota_features,
                    include_spatial_features=args.include_spatial_features,
                    lag_years=args.lag_years,
                    load_checkpoint=None,  # Never load checkpoint during HPO
                    seed=args.seed, batch_size=batch_size,
                    num_workers=args.num_workers,
                    max_epochs=args.epochs, lr=lr, weight_decay=weight_decay,
                    test_years=args.test_years,
                    use_residual_trend=args.use_residual_trend,
                    use_cwb_feature=args.use_cwb_feature,
                    drop_tavg=args.drop_tavg,
                    use_recursive_lags=args.use_recursive_lags,
                    use_gdd=args.use_gdd,
                    use_heat_stress_days=args.use_heat_stress_days,
                    use_rue=args.use_rue,
                    use_farquhar=args.use_farquhar,
                    results_dir=args.results_dir,
                    lr_scheduler_lambda=lr_scheduler_lambda,
                    patchtst_d_model=patchtst_d_model,
                    patchtst_num_attention_heads=patchtst_num_attention_heads,
                    patchtst_ffn_dim=patchtst_ffn_dim,
                    patchtst_num_layers=patchtst_num_layers,
                    patchtst_dropout=patchtst_dropout,
                )
            else:
                # For non-patchtst models, only tune shared hyperparameters
                batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
                lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
                weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)

                hpo_config = TSTModelConfig(
                    crop=args.crop, country=args.country,
                    model_type=args.model_type, aggregation=args.aggregation,
                    use_sota_features=args.use_sota_features,
                    include_spatial_features=args.include_spatial_features,
                    lag_years=args.lag_years,
                    load_checkpoint=None,
                    seed=args.seed, batch_size=batch_size,
                    num_workers=args.num_workers,
                    max_epochs=args.epochs, lr=lr, weight_decay=weight_decay,
                    test_years=args.test_years,
                    use_residual_trend=args.use_residual_trend,
                    use_cwb_feature=args.use_cwb_feature,
                    drop_tavg=args.drop_tavg,
                    use_recursive_lags=args.use_recursive_lags,
                    use_gdd=args.use_gdd,
                    use_heat_stress_days=args.use_heat_stress_days,
                    use_rue=args.use_rue,
                    use_farquhar=args.use_farquhar,
                    results_dir=args.results_dir,
                    lr_scheduler_lambda=lr_scheduler_lambda,
                    patchtst_d_model=args.patchtst_d_model,
                    patchtst_num_attention_heads=args.patchtst_num_attention_heads,
                    patchtst_ffn_dim=args.patchtst_ffn_dim,
                    patchtst_num_layers=args.patchtst_num_layers,
                    patchtst_dropout=args.patchtst_dropout,
                )

            # Create datamodule
            dm = DailyCYBenchSeqDataModule(hpo_config)
            dm.setup(
                train_years=fixed_splits['train_years'],
                val_years=fixed_splits['val_years'],
                test_years=fixed_splits['test_years']
            )

            # Create model
            model = create_model(hpo_config)

            # Setup callbacks for HPO (faster training)
            hpo_callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=False),
                LearningRateMonitor(logging_interval='epoch'),
            ]

            # WandB logger for each trial (disabled by default for speed)
            trial_logger = CSVLogger("logs/", name=f"cybench-hpo-{trial.number}")

            # Create trainer
            trainer = Trainer(
                max_epochs=hpo_config.max_epochs,
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                callbacks=hpo_callbacks,
                logger=[trial_logger],
                log_every_n_steps=50,
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
            )

            # Train
            print(f"\n[Trial {trial.number}] Training with hyperparameters...")
            try:
                trainer.fit(model, dm)

                # Get validation metrics
                val_metrics = trainer.validate(model, dm)
                if val_metrics:
                    val_results = val_metrics[0]

                    # Handle different objectives
                    if args.hpo_objective == 'nrmse':
                        return val_results.get('val/nrmse', float('inf'))
                    elif args.hpo_objective == 'r2':
                        return val_results.get('val/r2', -float('inf'))  # Return r2 directly for maximization
                    elif args.hpo_objective == 'multi':
                        # Multi-objective: return tuple (nrmse, r2)
                        nrmse = val_results.get('val/nrmse', float('inf'))
                        r2 = val_results.get('val/r2', -float('inf'))
                        return nrmse, r2  # Return r2 directly for maximization
                else:
                    if args.hpo_objective == 'multi':
                        return float('inf'), float('inf')
                    else:
                        return float('inf')
            except Exception as e:
                print(f"[Trial {trial.number}] Failed: {e}")
                if args.hpo_objective == 'multi':
                    return float('inf'), float('inf')
                else:
                    return float('inf')

        # Create and run study with pruning
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)

        if args.hpo_objective == 'multi':
            study = optuna.create_study(
                study_name=study_name,
                storage=args.hpo_storage,
                load_if_exists=True,
                directions=['minimize', 'maximize'],  # Minimize nrmse, maximize r2
                pruner=pruner
            )
        elif args.hpo_objective == 'r2':
            study = optuna.create_study(
                study_name=study_name,
                storage=args.hpo_storage,
                load_if_exists=True,
                direction='maximize',  # Maximize r2
                pruner=pruner
            )
        else:  # nrmse
            study = optuna.create_study(
                study_name=study_name,
                storage=args.hpo_storage,
                load_if_exists=True,
                direction='minimize',  # Minimize nrmse
                pruner=pruner
            )

        print(f"\n{'=' * 70}")
        print(f"Starting Optuna optimization with {args.n_trials} trials...")
        print(f"{'=' * 70}\n")

        # Enqueue default baseline configuration for comparison
        if args.model_type == 'patchtst':
            baseline_params = {
                'patchtst_d_model': 64,
                'patchtst_num_attention_heads': 4,
                'patchtst_ffn_dim': 256,
                'patchtst_num_layers': 3,
                'patchtst_dropout': 0.1,
                'batch_size': 16,
                'lr': 1e-4,
                'weight_decay': 1e-5,
            }
            study.enqueue_trial(baseline_params)
            print(f"[HPO] Enqueued baseline configuration: {baseline_params}")

        # Add median pruning to stop unpromising trials early
        study.optimize(optuna_objective, n_trials=args.n_trials, show_progress_bar=True)

        # Print and save results
        print(f"\n{'=' * 70}")
        print(f"HPO COMPLETED")
        print(f"{'=' * 70}\n")

        if args.hpo_objective == 'multi':
            print("Best trials (multi-objective):")
            # Find best trials for each objective
            best_nrmse_trial = min(study.best_trials, key=lambda t: t.values[0])
            best_r2_trial = max(study.best_trials, key=lambda t: t.values[1])

            print(f"\nBest NRMSE (Trial {best_nrmse_trial.number}):")
            print(f"  NRMSE: {best_nrmse_trial.values[0]:.4f}")
            print(f"  R²: {best_nrmse_trial.values[1]:.4f}")
            print(f"  Params: {best_nrmse_trial.params}")

            print(f"\nBest R² (Trial {best_r2_trial.number}):")
            print(f"  NRMSE: {best_r2_trial.values[0]:.4f}")
            print(f"  R²: {best_r2_trial.values[1]:.4f}")
            print(f"  Params: {best_r2_trial.params}")
        else:
            print(f"Best trial (Trial {study.best_trial.number}):")
            print(f"  Value: {study.best_value:.4f}")
            print(f"  Params: {study.best_params}")

        # Save results to text file
        with open(hpo_results_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write(f"OPTUNA HPO RESULTS - {args.model_type.upper()} | {args.crop}-{args.country}\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Study Name: {study_name}\n")
            f.write(f"Objective: {args.hpo_objective}\n")
            f.write(f"Total Trials: {len(study.trials)}\n")
            f.write(f"Timestamp: {timestamp}\n\n")

            if args.hpo_objective == 'multi':
                f.write("BEST TRIALS (Multi-Objective):\n")
                f.write("-" * 70 + "\n")

                best_nrmse_trial = min(study.best_trials, key=lambda t: t.values[0])
                f.write(f"\nBest NRMSE (Trial {best_nrmse_trial.number}):\n")
                f.write(f"  NRMSE: {best_nrmse_trial.values[0]:.6f}\n")
                f.write(f"  R²: {best_nrmse_trial.values[1]:.6f}\n")
                f.write(f"  Hyperparameters:\n")
                for param, value in best_nrmse_trial.params.items():
                    f.write(f"    {param}: {value}\n")

                best_r2_trial = max(study.best_trials, key=lambda t: t.values[1])
                f.write(f"\nBest R² (Trial {best_r2_trial.number}):\n")
                f.write(f"  NRMSE: {best_r2_trial.values[0]:.6f}\n")
                f.write(f"  R²: {best_r2_trial.values[1]:.6f}\n")
                f.write(f"  Hyperparameters:\n")
                for param, value in best_r2_trial.params.items():
                    f.write(f"    {param}: {value}\n")
            else:
                f.write("BEST TRIAL:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Trial Number: {study.best_trial.number}\n")
                f.write(f"Objective Value: {study.best_value:.6f}\n")
                f.write(f"Hyperparameters:\n")
                for param, value in study.best_params.items():
                    f.write(f"    {param}: {value}\n")

            f.write("\n" + "=" * 70 + "\n")
            f.write("ALL TRIALS\n")
            f.write("=" * 70 + "\n\n")

            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    if args.hpo_objective == 'multi':
                        f.write(f"Trial {trial.number}: NRMSE={trial.values[0]:.6f}, R²={trial.values[1]:.6f}\n")
                    else:
                        f.write(f"Trial {trial.number}: Value={trial.value:.6f}\n")
                    f.write(f"  Params: {trial.params}\n\n")

        print(f"\n[HPO] Results saved to: {hpo_results_file}")

        # Save best hyperparameters to CSV files
        if args.hpo_objective == 'multi':
            # Create CSV files in the save_checkpoint_dir
            csv_rmse_path = os.path.join(args.save_checkpoint_dir, 'optuna_rmse.csv')
            csv_r2_path = os.path.join(args.save_checkpoint_dir, 'optuna_r2.csv')

            # Get best trials for each metric
            best_rmse_trial = min(study.best_trials, key=lambda t: t.values[0])
            best_r2_trial = min(study.best_trials, key=lambda t: t.values[1])

            # Save RMSE-optimized hyperparameters
            with open(csv_rmse_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['hyperparameter', 'value'])
                for param, value in best_rmse_trial.params.items():
                    writer.writerow([param, value])
                writer.writerow(['nrmse', best_rmse_trial.values[0]])
                writer.writerow(['r2', -best_rmse_trial.values[1]])

            # Save R²-optimized hyperparameters
            with open(csv_r2_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['hyperparameter', 'value'])
                for param, value in best_r2_trial.params.items():
                    writer.writerow([param, value])
                writer.writerow(['nrmse', best_r2_trial.values[0]])
                writer.writerow(['r2', -best_r2_trial.values[1]])

            print(f"[HPO] Best RMSE hyperparameters saved to: {csv_rmse_path}")
            print(f"[HPO] Best R² hyperparameters saved to: {csv_r2_path}")

        # Skip regular training when in HPO mode
        print(f"\n[HPO] Hyperparameter optimization complete. Exiting.")
        sys.exit(0)

    # ==================== REGULAR TRAINING (no HPO) ====================
    # PHASE 3: Final Model Training and Evaluation
    print(f"\n{'=' * 70}")
    print(f"PHASE 3: Final Model Training and Evaluation")
    print(f"{'=' * 70}\n")

    # Create datamodule for final model
    dm_final = DailyCYBenchSeqDataModule(config)
    dm_final.setup(
        train_years=fixed_splits['train_years'],
        val_years=fixed_splits['val_years'],
        test_years=fixed_splits['test_years']
    )

    # Create final model
    model_final = create_model(config)

    # WandB logger for final model
    try:
        wandb_project = args.wandb_project if args.wandb_project else "CYBENCH-LSTF-AAAI2027"
        wandb_run_name = args.wandb_run_name if args.wandb_run_name else f"{args.model_type}-{args.crop}-{args.country}"
        wandb_logger = WandbLogger(
            project=wandb_project,
            name=wandb_run_name,
            config=vars(args),
            group=f"{args.crop}-{args.country}"
        )
        loggers = [wandb_logger]
    except Exception as e:
        print(f"[WandB Warning] Could not initialise WandB logger: {e}")
        loggers = [CSVLogger("logs/", name="cybench-tst")]

    # Setup callbacks
    final_callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=True),
        ModelCheckpoint(
            monitor='val_loss',
            save_top_k=1,
            mode='min',
            dirpath=args.save_checkpoint_dir,
            filename=f'{generate_checkpoint_name(args)}_{{epoch:02d}}_{{val_loss:.4f}}_runid:{run_id}',
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    if args.lr_decay_every is not None:
        print(f"[LR Schedule] Enabled: LR will halve every {args.lr_decay_every} epochs")

    trainer = Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=final_callbacks,
        logger=loggers,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=False,
    )

    print("\nTraining final model...")
    trainer.fit(model_final, dm_final)

    print("\nEvaluating final model...")
    test_results = trainer.test(model_final, dm_final, ckpt_path="best")
    # Use same pattern as CV folds - Lightning returns float dict, not torchmetrics tensors
    if test_results:
        r = test_results[0]
        final_metrics = {
            'mse': r.get('test/mse'),
            'mae': r.get('test/mae'),
            'rmse': r.get('test/rmse'),
            'r2': r.get('test/r2'),
            'mape': r.get('test/mape'),
            'smape': r.get('test/smape'),
            'nrmse': r.get('test/nrmse'),
        }
    else:
        final_metrics = {}

    # Save test results to CSV files with per-year metrics
    # The per-year metrics are now computed during trainer.test() in on_test_epoch_end()
    print(f"\n[CSV Results] Retrieving per-year metrics from test results...")

    # trainer.test() modifies model_final in-place when loading checkpoint weights.
    # _test_results_per_year is set during on_test_epoch_end() which runs during trainer.test(),
    # so it is available on model_final after trainer.test() returns.
    if hasattr(model_final, '_test_results_per_year') and model_final._test_results_per_year:
        per_year_metrics = model_final._test_results_per_year
    else:
        print(f"[CSV Results] Warning: No per-year metrics found on model. Using overall metrics only.")
        per_year_metrics = {}

    # Log per-year metrics to console
    print(f"\n[CSV Results] Per-Year Test Metrics:")
    for year in sorted(fixed_splits['test_years']):
        print(f"  Year {year}:")
        for metric in ['nrmse', 'mape', 'r2']:
            if f'{metric}_{year}' in per_year_metrics:
                print(f"    {metric.upper()}: {per_year_metrics[f'{metric}_{year}']:.4f}")

    print(f"  Overall:")
    for metric in ['nrmse', 'mape', 'r2']:
        if f'{metric}_overall' in per_year_metrics:
            print(f"    {metric.upper()}: {per_year_metrics[f'{metric}_overall']:.4f}")

    # Save to CSV - extract actual years from test results (not from fixed_splits)
    # This handles cases where the datamodule gets reconfigured during final training
    actual_test_years = set()
    for key in per_year_metrics.keys():
        if key.endswith('_overall'):
            continue
        # Extract year from keys like 'nrmse_2015', 'mape_2017', etc.
        parts = key.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            actual_test_years.add(int(parts[1]))

    save_test_results_to_csv(
        config=config,
        test_results=per_year_metrics,
        test_years=sorted(actual_test_years),
        run_id=run_id,
        timestamp=timestamp
    )

    # Print split summary
    print(f"\n{'=' * 70}")
    print(f"SPLIT SUMMARY: {args.crop}-{args.country}")
    print(f"{'=' * 70}")
    print(f"\n  Available years ({len(all_years)}): {all_years}")
    print(f"  Train years ({len(fixed_splits['train_years'])}): {sorted(fixed_splits['train_years'])}")
    print(f"  Val years ({len(fixed_splits['val_years'])}): {sorted(fixed_splits['val_years'])}")
    print(f"  Test years ({len(fixed_splits['test_years'])}): {sorted(fixed_splits['test_years'])}")

    # Print final results with all metrics
    print_metrics_table(
        f"FINAL RESULTS: {args.crop}-{args.country}",
        final_metrics
    )
    
    # Print experiment completion message
    print(f"\n{'=' * 70}")
    print(f"Experiment complete: {args.crop}-{args.country}")
    print(f"  Model: {args.model_type}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Test years: {fixed_splits['test_years']}")
    print(f"{'=' * 70}\n")
