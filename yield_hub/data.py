from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .settings import REPO_ROOT, configure_runtime_paths, resolve_data_root

configure_runtime_paths()

import cybench.config as cybench_config
from cybench_compat import KEY_LOC, KEY_YEAR, LOCATION_PROPERTIES
import cybench.datasets.configured as configured_loader
from cybench.datasets.configured import load_dfs_crop
from cybench.datasets.dataset import Dataset as CYDataset
from loadData import DailyCYBenchSeqDataModule, calculate_fixed_split

def configure_data_root(data_root: str | None = None) -> Path:
    resolved = resolve_data_root(data_root=data_root, repo_root=REPO_ROOT)
    cybench_config.PATH_DATA_DIR = str(resolved)
    configured_loader.PATH_DATA_DIR = str(resolved)
    return resolved


def _inject_location_df(crop: str, country: str, dfs_x: Dict, data_root: Path) -> Dict:
    """Load location CSV when the upstream CY-BENCH loader does not expose it."""
    if "location" in dfs_x:
        return dfs_x

    location_path = data_root / crop / country / f"location_{crop}_{country}.csv"
    if not location_path.exists():
        return dfs_x

    df_loc = pd.read_csv(location_path)
    required_cols = [KEY_LOC] + [c for c in LOCATION_PROPERTIES if c in df_loc.columns]
    if len(required_cols) <= 1:
        return dfs_x

    dfs_x = dict(dfs_x)
    dfs_x["location"] = df_loc[required_cols].set_index(KEY_LOC)
    return dfs_x


def load_dataset(crop: str, country: str, data_root: str | None = None) -> Tuple[CYDataset, list]:
    resolved_data_root = configure_data_root(data_root)
    df_y, dfs_x = load_dfs_crop(crop, [country])
    if df_y is None or len(df_y) == 0:
        target_dir = Path(cybench_config.PATH_DATA_DIR) / crop / country
        raise ValueError(
            f"No data found for {crop}-{country}. "
            f"cybench.config={cybench_config.__file__} | "
            f"PATH_DATA_DIR={cybench_config.PATH_DATA_DIR} | "
            f"target_dir={target_dir} | "
            f"target_exists={target_dir.exists()}"
        )

    dfs_x = _inject_location_df(crop, country, dfs_x, resolved_data_root)
    ds = CYDataset(crop, df_y, dfs_x)
    all_years = sorted({ds[i][KEY_YEAR] for i in range(len(ds))})
    return ds, all_years


def build_datamodule(model_config, data_root: str | None = None) -> Tuple[DailyCYBenchSeqDataModule, Dict]:
    _, all_years = load_dataset(model_config.crop, model_config.country, data_root=data_root)
    fixed_splits = calculate_fixed_split(
        all_years,
        test_years=model_config.test_years,
        val_years=2,
    )

    dm = DailyCYBenchSeqDataModule(model_config)
    dm.setup(
        train_years=fixed_splits["train_years"],
        val_years=fixed_splits["val_years"],
        test_years=fixed_splits["test_years"],
    )
    return dm, fixed_splits
