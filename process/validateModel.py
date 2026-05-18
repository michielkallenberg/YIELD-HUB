import os, sys
from tqdm import tqdm
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error

from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset

import torch
import torchmetrics
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

# Custom library
from loadData import prepare_features_and_targets

def evaluate_predictions_by_year(y_true, y_pred, years, metrics=None, min_samples_per_year=2, epsilon=1e-6):
    """
    Compute regression metrics overall and per year with robustness to small sample sizes
    and zero/near-zero variance.

    Args:
        y_true: np.ndarray of true targets
        y_pred: np.ndarray of predicted targets
        years: list or np.ndarray of corresponding years
        metrics: list of metrics to compute ['r2', 'rmse', 'mape', 'normalized_rmse']
        min_samples_per_year: minimum number of samples to compute metrics per year
        epsilon: small value to avoid division by zero in normalized RMSE

    Returns:
        dict: nested dictionary {overall: {...}, year1: {...}, ...}
    """

    if metrics is None:
        metrics = ['r2', 'rmse', 'mape', 'normalized_rmse']

    EPSILON = epsilon
    results = {}

    # --- Metric functions ---
    def compute_r2_or_nan(y_t, y_p):
        if len(y_t) < min_samples_per_year or np.std(y_t) < EPSILON:
            return np.nan
        return r2_score(y_t, y_p)

    def compute_rmse(y_t, y_p):
        if len(y_t) == 0:
            return np.nan
        return np.sqrt(mean_squared_error(y_t, y_p))

    def compute_mape(y_t, y_p):
        if len(y_t) == 0:
            return np.nan
        return mean_absolute_percentage_error(y_t, y_p)

    def compute_normalized_rmse(y_t, y_p):
        if len(y_t) < min_samples_per_year:
            return compute_rmse(y_t, y_p)
        denom = np.max(y_t) - np.min(y_t)
        return compute_rmse(y_t, y_p) / (denom + EPSILON) * 100

    # --- Overall metrics ---
    results['overall'] = {}
    for metric in metrics:
        if metric == 'r2':
            results['overall']['r2'] = compute_r2_or_nan(y_true, y_pred)
        elif metric == 'rmse':
            results['overall']['rmse'] = compute_rmse(y_true, y_pred)
        elif metric == 'mape':
            results['overall']['mape'] = compute_mape(y_true, y_pred)
        elif metric == 'normalized_rmse':
            results['overall']['normalized_rmse'] = compute_normalized_rmse(y_true, y_pred)

    # --- Per-year metrics ---
    years = np.array(years)
    for year in sorted(set(years)):
        mask = np.where(years == year)[0]
        y_t = y_true[mask]
        y_p = y_pred[mask]

        results[year] = {}
        for metric in metrics:
            if metric == 'r2':
                results[year]['r2'] = compute_r2_or_nan(y_t, y_p)
            elif metric == 'rmse':
                results[year]['rmse'] = compute_rmse(y_t, y_p)
            elif metric == 'mape':
                results[year]['mape'] = compute_mape(y_t, y_p)
            elif metric == 'normalized_rmse':
                results[year]['normalized_rmse'] = compute_normalized_rmse(y_t, y_p)

    return results


def store_model_results(results_dict, model_name, country, crop, file_path="../output/myoutputs/sklearn_model_results.csv"):
    rows = []
    for year, metrics in results_dict.items():
        for metric_name, value in metrics.items():
            rows.append({
                "model": str(model_name),
                "country": str(country[0]),
                "crop": str(crop),
                "year": str(year),
                "metric": str(metric_name),
                "value": value
            })

    new_df = pd.DataFrame(rows)

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df

    combined_df = combined_df.drop_duplicates(subset=["model", "country", "crop", "year", "metric"])
    combined_df.to_csv(file_path, index=False)
    # print(f"Results stored successfully. Total records: {len(combined_df)}")
    return combined_df

def evaluate_OOD_results_from_countries(crop, model_name, pipeline, file_path):
    """
    Evaluates the trained model on EU countries in CY-BENCH dataset.
    """

    countries_to_evaluate = ["AT", "BE", "BG", "CZ", "DE", "DK", "EE", "EL", "ES", "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LV", "NL", "PL", "PT", "RO", "SE", "SK"]

    if crop == "maize":
        countries_not_of_interest = ["EE", "FI", "IE", "LV"]
    elif crop == "wheat":
        countries_not_of_interest = ["SK"]
    else:
        raise Exception("Crop can either be maize or wheat.")

    for country in countries_not_of_interest:
        countries_to_evaluate.remove(country)
    
    # for country in tqdm(countries_to_evaluate):
    pbar = tqdm(countries_to_evaluate, desc="Evaluating")
    for country in pbar:
        pbar.set_description(f"Evaluating {model_name} model on {country} country")

        df_y, dfs_x = load_dfs_crop(crop, countries=[country])
        ds = CYDataset(crop=crop, data_target=df_y, data_inputs=dfs_x)

        years_sorted = sorted(list(ds.years))
        train_years = [y for y in years_sorted if y <= 2017]
        test_years  = [y for y in years_sorted if y >= 2018]
        _, test_ds = ds.split_on_years((train_years, test_years))

        X_test, y_test, years_test = prepare_features_and_targets(test_ds)
        y_pred = pipeline.predict(X_test)

        results_by_year = evaluate_predictions_by_year(y_test, y_pred, years_test)
        _ = store_model_results(results_by_year, model_name, country, crop, file_path)


class ModelMetrics(nn.Module):
    """
    Comprehensive metrics for agricultural yield forecasting.

    Wraps torchmetrics to provide MSE, MAE, RMSE, R², MAPE, SMAPE, NRMSE.
    Prefix (train/val/test) is used for wandb logging namespacing.

    NOTE: Inherits from nn.Module so metrics are moved to device automatically
    when the parent LightningModule is moved to GPU.
    """

    def __init__(self, prefix: str = "test", include_nrmse: bool = True):
        super().__init__()
        self.prefix = prefix
        metrics_dict = {
            'mse': torchmetrics.MeanSquaredError(),
            'mae': torchmetrics.MeanAbsoluteError(),
            'r2': torchmetrics.R2Score(),
            'mape': torchmetrics.MeanAbsolutePercentageError(),
            'smape': torchmetrics.SymmetricMeanAbsolutePercentageError(),
        }
        # Only include NRMSE if requested (exclude for training since targets are normalized)
        if include_nrmse:
            metrics_dict['nrmse'] = torchmetrics.NormalizedRootMeanSquaredError(normalization='mean')
        self.metrics = torchmetrics.MetricCollection(metrics_dict)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.metrics.update(preds, targets)

    def compute(self) -> Dict:
        return self.metrics.compute()

    def reset(self):
        self.metrics.reset()

    def log_results(self, step: str = "val"):
        results = self.compute()
        print(f"\n{'-' * 60}")
        print(f"{step.upper()} METRICS ({self.prefix.upper()}):")
        print(f"  MSE:   {results['mse']:.4f}")
        print(f"  MAE:   {results['mae']:.4f}")
        print(f"  RMSE:  {torch.sqrt(results['mse']):.4f}")
        print(f"  R²:    {results['r2']:.4f}")
        # MAPE/SMAPE reported as fractions
        print(f"  MAPE:  {results['mape']:.4f}")
        print(f"  SMAPE: {results['smape']:.4f}")
        # Only print NRMSE if it exists (excluded for training metrics)
        if 'nrmse' in results:
            print(f"  NRMSE: {results['nrmse']:.4f}")
        print(f"{'-' * 60}")

def format_metrics_dict(results: Dict) -> Dict[str, float]:
    """
    Convert torchmetrics tensor results to a clean dict of floats.

    Args:
        results: Dict with tensor metrics from ModelMetrics.compute()

    Returns:
        Dict with all metrics as Python floats
    """
    return {
        'mse': float(results['mse'].item()) if 'mse' in results else None,
        'mae': float(results['mae'].item()) if 'mae' in results else None,
        'rmse': float(torch.sqrt(results['mse']).item()) if 'mse' in results else None,
        'r2': float(results['r2'].item()) if 'r2' in results else None,
        'mape': float(results['mape'].item()) if 'mape' in results else None,
        'smape': float(results['smape'].item()) if 'smape' in results else None,
        'nrmse': float(results['nrmse'].item()) if 'nrmse' in results else None,
    }

def print_metrics_table(title: str, metrics: Dict, step: str = "test"):
    """
    Print a nicely formatted table of all metrics.

    Args:
        title: Section title (e.g., "CV Fold 1 Results")
        metrics: Dict from format_metrics_dict()
        step: Step name for context
    """
    print(f"\n{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}")

    if metrics.get('mse') is not None:
        print(f"  MSE:   {metrics['mse']:.4f}")
    if metrics.get('mae') is not None:
        print(f"  MAE:   {metrics['mae']:.4f}")
    if metrics.get('rmse') is not None:
        print(f"  RMSE:  {metrics['rmse']:.4f}")
    if metrics.get('r2') is not None:
        print(f"  R²:    {metrics['r2']:.4f}")
    if metrics.get('mape') is not None:
        print(f"  MAPE:  {metrics['mape']:.2f}%")
    if metrics.get('smape') is not None:
        print(f"  SMAPE: {metrics['smape']:.2f}%")
    if metrics.get('nrmse') is not None:
        print(f"  NRMSE: {metrics['nrmse']:.4f}")

    print(f"{'-' * 70}")


def save_walk_forward_results(
    results_matrix: List[Dict],
    all_years: List[int],
    test_years: int,
    config,
    run_id: str,
    timestamp: str
) -> str:
    """
    Save walk-forward results to CSV in a structured format.

    Creates a CSV with columns:
    - fold: Fold index (0 to test_years-1)
    - train_end_year: Last year of training data
    - test_year: Year being tested
    - mse, mae, rmse, r2, mape, smape, nrmse: Metric values

    Also saves aggregated metrics per-year and overall.

    Args:
        results_matrix: List of dicts, each containing fold metrics by year
        all_years: All available years in the dataset
        test_years: Number of test years (number of folds)
        config: Model config object with crop, country, etc.
        run_id: Unique run identifier
        timestamp: Timestamp string

    Returns:
        Path to saved CSV file
    """
    os.makedirs(config.results_dir, exist_ok=True)

    # Build flattened DataFrame with config info
    rows = []
    for fold_result in results_matrix:
        fold_idx = fold_result['fold_idx']
        train_end_year = fold_result['train_end_year']

        for test_year, metrics in fold_result['yearly_metrics'].items():
            row = {
                'run_id': run_id,
                'crop': config.crop,
                'country': config.country,
                'model_type': getattr(config, 'model_type', 'unknown'),
                'aggregation': getattr(config, 'aggregation', 'unknown'),
                'fold': fold_idx,
                'train_end_year': train_end_year,
                'test_year': test_year,
                'mse': metrics.get('mse'),
                'mae': metrics.get('mae'),
                'rmse': metrics.get('rmse'),
                'r2': metrics.get('r2'),
                'mape': metrics.get('mape'),
                'smape': metrics.get('smape'),
                'nrmse': metrics.get('nrmse'),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Save detailed results
    csv_path = os.path.join(
        config.results_dir,
        f"{config.crop}_{config.country}_walkforward_{timestamp}_{run_id}.csv"
    )
    df.to_csv(csv_path, index=False)

    # Calculate and save per-year aggregates
    metric_cols = ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape', 'nrmse']
    per_year_agg = df.groupby('test_year').agg({
        **{col: ['mean', 'std'] for col in metric_cols}
    }).reset_index()
    per_year_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] for col in per_year_agg.columns]

    # Add config columns to aggregated
    per_year_agg['run_id'] = run_id
    per_year_agg['crop'] = config.crop
    per_year_agg['country'] = config.country
    per_year_agg['model_type'] = getattr(config, 'model_type', 'unknown')
    per_year_agg['aggregation'] = getattr(config, 'aggregation', 'unknown')

    agg_path = os.path.join(
        config.results_dir,
        f"{config.crop}_{config.country}_walkforward_aggregated_{timestamp}_{run_id}.csv"
    )
    per_year_agg.to_csv(agg_path, index=False)

    # Calculate overall averages
    overall_avg = df[metric_cols].mean()

    print(f"\n[Walk-Forward] CSV Results saved to:")
    print(f"  Detailed: {csv_path}")
    print(f"  Aggregated: {agg_path}")
    print(f"\n[Walk-Forward] Overall Average Across All Folds and Years:")
    print(f"  MSE:   {overall_avg['mse']:.4f}")
    print(f"  MAE:   {overall_avg['mae']:.4f}")
    print(f"  RMSE:  {overall_avg['rmse']:.4f}")
    print(f"  R2:    {overall_avg['r2']:.4f}")
    print(f"  MAPE:  {overall_avg['mape']:.2f}%")
    print(f"  SMAPE: {overall_avg['smape']:.2f}%")
    print(f"  NRMSE: {overall_avg['nrmse']:.4f}")

    return csv_path


def run_walk_forward_validation(
    all_years: List[int],
    test_years: int,
    config,
    create_model_fn,
    datamodule_class,
    trainer_class,
    early_stopping_epoch: int,
    es_patience: int,
    es_min_delta: float,
    loggers: List,
    run_id: str,
    timestamp: str
) -> Dict:
    """
    Run walk-forward validation where each fold is tested on ALL future years.

    For example, with test_years=5 and years 2000-2020:
    - Fold 0 (train up to 2015): test on 2016, 2017, 2018, 2019, 2020
    - Fold 1 (train up to 2016): test on 2017, 2018, 2019, 2020
    - Fold 2 (train up to 2017): test on 2018, 2019, 2020
    - Fold 3 (train up to 2018): test on 2019, 2020
    - Fold 4 (train up to 2019): test on 2020

    Args:
        all_years: Sorted list of all available years
        test_years: Number of years to walk forward
        config: Model configuration object
        create_model_fn: Function that creates a model from config
        datamodule_class: Class for creating datamodules
        trainer_class: PyTorch Lightning Trainer class
        early_stopping_epoch: Max epochs for early stopping
        es_patience: Early stopping patience
        es_min_delta: Early stopping min delta
        loggers: List of loggers (WandB, CSV)
        run_id: Unique run identifier
        timestamp: Timestamp string

    Returns:
        Dict containing:
        - results_matrix: List of fold results with yearly metrics
        - overall_metrics: Average metrics across all folds/years
        - per_year_metrics: Aggregated metrics per test year
    """
    from cybench.process.loadData import generate_walk_forward_splits

    print(f"\n{'=' * 70}")
    print(f"WALK-FORWARD VALIDATION (Test on All Future Years)")
    print(f"{'=' * 70}\n")

    # Generate walk-forward splits
    wf_splits = generate_walk_forward_splits(all_years, test_years)

    print(f"[Walk-Forward Config]")
    print(f"  Number of folds: {len(wf_splits)}")
    print(f"  Early stopping epoch limit (E): {early_stopping_epoch}")
    print(f"  Early stopping patience: {es_patience}")
    print(f"  Early stopping min_delta: {es_min_delta}")
    print(f"\n[Walk-Forward Splits]")
    for s in wf_splits:
        print(f"  Fold {s['fold_idx']}: train_years={len(s['train_years'])} ({min(s['train_years'])}-{max(s['train_years'])})")

    # Store results from all folds
    results_matrix = []

    for fold_idx, split in enumerate(wf_splits):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold_idx + 1}/{len(wf_splits)}")
        print(f"{'=' * 70}")
        print(f"  Train years: {split['train_years']}")
        print(f"  Train end year: {max(split['train_years'])}")

        # Create config for this fold
        fold_config = _update_config_for_fold(
            config, split['train_years'], early_stopping_epoch
        )

        # Create datamodule and model
        dm_fold = datamodule_class(fold_config)
        dm_fold.setup(
            train_years=split['train_years'],
            val_years=[],
            test_years=[]
        )

        model_fold = create_model_fn(fold_config)

        # Setup callbacks
        fold_callbacks = [
            EarlyStopping(
                monitor='train_loss',
                patience=es_patience,
                min_delta=es_min_delta,
                mode='min',
                verbose=True,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ]

        # Create trainer
        trainer_fold = trainer_class(
            max_epochs=fold_config.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            callbacks=fold_callbacks,
            logger=loggers,
            log_every_n_steps=10,
            enable_progress_bar=True,
            enable_model_summary=False,
        )

        print(f"\n[Fold {fold_idx}] Training...")
        trainer_fold.fit(model_fold, dm_fold)

        # Test on ALL future years
        train_end_year = max(split['train_years'])
        future_years = [y for y in all_years if y > train_end_year]

        print(f"\n[Fold {fold_idx}] Testing on {len(future_years)} future years: {future_years}")

        fold_yearly_metrics = {}

        for test_year in future_years:
            # Create test datamodule for this single year
            dm_test = datamodule_class(fold_config)
            # Copy normalization statistics from training datamodule to avoid NaN
            dm_test.copy_normalization_from(dm_fold)
            dm_test.setup(
                train_years=[],
                val_years=[],
                test_years=[test_year]
            )

            test_results = trainer_fold.test(model_fold, dm_test, verbose=False)

            if test_results:
                r = test_results[0]
                metrics = {
                    'mse': r.get('test/mse'),
                    'mae': r.get('test/mae'),
                    'rmse': r.get('test/rmse'),
                    'r2': r.get('test/r2'),
                    'mape': r.get('test/mape'),
                    'smape': r.get('test/smape'),
                    'nrmse': r.get('test/nrmse'),
                }
                fold_yearly_metrics[test_year] = metrics

                print(f"  Year {test_year}: NRMSE={metrics['nrmse']:.4f}, R2={metrics['r2']:.4f}")

        results_matrix.append({
            'fold_idx': fold_idx,
            'train_end_year': train_end_year,
            'yearly_metrics': fold_yearly_metrics,
        })

    # Aggregate results
    aggregated = _aggregate_walk_forward_results(results_matrix, all_years, test_years)

    # Save to CSV
    csv_path = save_walk_forward_results(
        results_matrix, all_years, test_years, config, run_id, timestamp
    )

    # Log to WandB (overall averages with run_id)
    _log_walk_forward_to_wandb(loggers, aggregated, run_id)

    return {
        'results_matrix': results_matrix,
        'overall_metrics': aggregated['overall'],
        'per_year_metrics': aggregated['per_year'],
        'csv_path': csv_path,
    }


def _update_config_for_fold(config, train_years, max_epochs):
    """Update config for a walk-forward fold."""
    # Create a copy of the config with updated test_years and max_epochs
    fold_config = config.__class__(
        crop=config.crop,
        country=config.country,
        model_type=getattr(config, 'model_type', None),
        aggregation=getattr(config, 'aggregation', None),
        use_sota_features=getattr(config, 'use_sota_features', False),
        include_spatial_features=getattr(config, 'include_spatial_features', False),
        lag_years=getattr(config, 'lag_years', 1),
        load_checkpoint=None,
        seed=config.seed,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        max_epochs=max_epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        test_years=1,
        use_residual_trend=getattr(config, 'use_residual_trend', False),
        use_recursive_lags=getattr(config, 'use_recursive_lags', False),
        use_gdd=getattr(config, 'use_gdd', False),
        use_heat_stress_days=getattr(config, 'use_heat_stress_days', False),
        use_rue=getattr(config, 'use_rue', False),
        use_farquhar=getattr(config, 'use_farquhar', False),
        use_cwb_feature=getattr(config, 'use_cwb_feature', False),
        drop_tavg=getattr(config, 'drop_tavg', False),
        use_revin=getattr(config, 'use_revin', False),
        results_dir=config.results_dir,
        lr_scheduler_lambda=getattr(config, 'lr_scheduler_lambda', None),
    )

    # Add model-specific hyperparameters if they exist
    for attr in ['xlinear_hidden_size', 'xlinear_temporal_ff', 'xlinear_channel_ff', 'xlinear_dropout',
                 'patchtst_d_model', 'patchtst_num_attention_heads', 'patchtst_ffn_dim',
                 'patchtst_num_layers', 'patchtst_dropout']:
        if hasattr(config, attr):
            setattr(fold_config, attr, getattr(config, attr))

    return fold_config


def _aggregate_walk_forward_results(results_matrix: List[Dict], all_years: List[int], test_years: int) -> Dict:
    """Aggregate walk-forward results into overall and per-year metrics."""
    # Collect all metric values
    all_metric_values = {m: [] for m in ['mse', 'mae', 'rmse', 'r2', 'mape', 'smape', 'nrmse']}
    per_year_values = {}

    for fold_result in results_matrix:
        for test_year, metrics in fold_result['yearly_metrics'].items():
            for metric_name, value in metrics.items():
                if value is not None:
                    all_metric_values[metric_name].append(value)

                    # Track per-year values
                    if test_year not in per_year_values:
                        per_year_values[test_year] = {m: [] for m in all_metric_values}
                    per_year_values[test_year][metric_name].append(value)

    # Calculate overall averages
    overall = {}
    for metric_name, values in all_metric_values.items():
        if values:
            overall[metric_name] = np.mean(values)
            overall[f'{metric_name}_std'] = np.std(values)
        else:
            overall[metric_name] = None
            overall[f'{metric_name}_std'] = None

    # Calculate per-year averages
    per_year = {}
    for year, metrics_dict in sorted(per_year_values.items()):
        per_year[year] = {}
        for metric_name, values in metrics_dict.items():
            if values:
                per_year[year][metric_name] = np.mean(values)
                per_year[year][f'{metric_name}_std'] = np.std(values)
            else:
                per_year[year][metric_name] = None
                per_year[year][f'{metric_name}_std'] = None

    return {'overall': overall, 'per_year': per_year}


def _log_walk_forward_to_wandb(loggers: List, aggregated: Dict, run_id: str):
    """Log walk-forward aggregated metrics to WandB."""
    for logger in loggers:
        if hasattr(logger, 'experiment'):  # WandB logger
            metrics_to_log = {
                'walk_forward/run_id': run_id,
            }
            for metric_name, value in aggregated['overall'].items():
                if value is not None and not metric_name.endswith('_std'):
                    metrics_to_log[f'walk_forward/{metric_name}'] = value
                    std_key = f'{metric_name}_std'
                    if std_key in aggregated['overall']:
                        metrics_to_log[f'walk_forward/{metric_name}_std'] = aggregated['overall'][std_key]

            logger.experiment.log(metrics_to_log)