from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .settings import REPO_ROOT, resolve_data_root


DATA_CONTRACT_VERSION = "1.0"

REQUIRED_FILES = {
    "yield": "yield_{crop}_{country}.csv",
    "soil": "soil_{crop}_{country}.csv",
    "crop_calendar": "crop_calendar_{crop}_{country}.csv",
    "meteo": "meteo_{crop}_{country}.csv",
    "fpar": "fpar_{crop}_{country}.csv",
    "ndvi": "ndvi_{crop}_{country}.csv",
    "soil_moisture": "soil_moisture_{crop}_{country}.csv",
    "location": "location_{crop}_{country}.csv",
}

REQUIRED_COLUMNS = {
    "yield": {"adm_id", "harvest_year", "yield"},
    "soil": {"adm_id", "awc", "bulk_density"},
    "crop_calendar": {"adm_id", "sos", "eos"},
    "meteo": {"adm_id", "date", "tmin", "tmax", "prec", "rad", "tavg", "cwb"},
    "fpar": {"adm_id", "date", "fpar"},
    "ndvi": {"adm_id", "date", "ndvi"},
    "soil_moisture": {"adm_id", "date", "ssm"},
    "location": {"adm_id", "latitude", "longitude"},
}

OPTIONAL_COLUMNS = {
    "yield": set(),
    "soil": set(),
    "crop_calendar": set(),
    "meteo": set(),
    "fpar": set(),
    "ndvi": set(),
    "soil_moisture": set(),
    "location": set(),
}

UNIQUE_KEYS = {
    "yield": ["adm_id", "harvest_year"],
    "soil": ["adm_id"],
    "crop_calendar": ["adm_id"],
    "location": ["adm_id"],
}

DATE_COLUMNS = {
    "meteo": "date",
    "fpar": "date",
    "ndvi": "date",
    "soil_moisture": "date",
}

NUMERIC_RULES = {
    "yield": {"yield": {"min": 0.0}},
    "soil": {
        "awc": {"min": 0.0},
        "bulk_density": {"min": 0.0},
    },
    "crop_calendar": {
        "sos": {"min": 1.0, "max": 366.0},
        "eos": {"min": 1.0, "max": 366.0},
    },
    "meteo": {},
    "fpar": {"fpar": {"min": 0.0, "max": 100.0, "preferred_max": 1.0}},
    "ndvi": {"ndvi": {"min": -1.0, "max": 1.0}},
    "soil_moisture": {"ssm": {"min": 0.0, "max": 100.0, "preferred_max": 1.0}},
    "location": {
        "latitude": {"min": -90.0, "max": 90.0},
        "longitude": {"min": -180.0, "max": 180.0},
    },
}


def _to_serializable_int_list(values: Iterable[int]) -> list[int]:
    return [int(v) for v in values]


def _count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def _coerce_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    if parsed.notna().sum() == 0:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def _validate_numeric_column(series: pd.Series, rule: Dict) -> Dict:
    numeric = pd.to_numeric(series, errors="coerce")
    invalid_non_numeric = int(numeric.isna().sum() - series.isna().sum())
    invalid_below_min = 0
    invalid_above_max = 0
    scale_warning = None

    if "min" in rule:
        invalid_below_min = int((numeric < rule["min"]).fillna(False).sum())
    if "max" in rule:
        invalid_above_max = int((numeric > rule["max"]).fillna(False).sum())
    if "preferred_max" in rule and invalid_non_numeric == 0:
        observed_max = numeric.max(skipna=True)
        if pd.notna(observed_max) and observed_max > rule["preferred_max"]:
            scale_warning = (
                f"values exceed preferred normalized max {rule['preferred_max']} "
                f"but remain within accepted max {rule['max']}"
            )

    return {
        "non_numeric": invalid_non_numeric,
        "below_min": invalid_below_min,
        "above_max": invalid_above_max,
        "scale_warning": scale_warning,
        "ok": invalid_non_numeric == 0 and invalid_below_min == 0 and invalid_above_max == 0,
    }


def _validate_file(key: str, path: Path, crop: str, country: str) -> Dict:
    file_info = {
        "path": str(path),
        "exists": path.exists(),
        "required_columns": sorted(REQUIRED_COLUMNS[key]),
        "optional_columns": sorted(OPTIONAL_COLUMNS[key]),
        "missing_columns": [],
        "unexpected_columns": [],
        "row_count": None,
        "non_empty": False,
        "duplicate_key_rows": 0,
        "invalid_dates": 0,
        "detected_years": [],
        "missing_harvest_years": [],
        "numeric_checks": {},
        "warnings": [],
        "ok": True,
    }

    if not path.exists():
        file_info["ok"] = False
        return file_info

    df = pd.read_csv(path)
    file_info["row_count"] = len(df)
    file_info["non_empty"] = len(df) > 0
    file_info["unexpected_columns"] = sorted(
        set(df.columns) - REQUIRED_COLUMNS[key] - OPTIONAL_COLUMNS[key]
    )

    if not file_info["non_empty"]:
        file_info["ok"] = False
        return file_info

    missing_columns = sorted(REQUIRED_COLUMNS[key] - set(df.columns))
    file_info["missing_columns"] = missing_columns
    if missing_columns:
        file_info["ok"] = False
        return file_info

    if key in UNIQUE_KEYS:
        dup_count = int(df.duplicated(subset=UNIQUE_KEYS[key]).sum())
        file_info["duplicate_key_rows"] = dup_count
        if dup_count > 0:
            file_info["ok"] = False

    if key in DATE_COLUMNS:
        parsed = _coerce_datetime(df[DATE_COLUMNS[key]])
        invalid_dates = int(parsed.isna().sum())
        file_info["invalid_dates"] = invalid_dates
        if invalid_dates > 0:
            file_info["ok"] = False
        detected_years = sorted(parsed.dropna().dt.year.unique().tolist())
        file_info["detected_years"] = _to_serializable_int_list(detected_years)

    if key == "yield":
        harvest_years = pd.to_numeric(df["harvest_year"], errors="coerce").dropna().astype(int)
        detected_years = sorted(harvest_years.unique().tolist())
        file_info["detected_years"] = _to_serializable_int_list(detected_years)
        if detected_years:
            expected = set(range(min(detected_years), max(detected_years) + 1))
            missing = sorted(expected - set(detected_years))
            file_info["missing_harvest_years"] = _to_serializable_int_list(missing)

    for column, rule in NUMERIC_RULES[key].items():
        if column not in df.columns:
            continue
        check = _validate_numeric_column(df[column], rule)
        file_info["numeric_checks"][column] = check
        if check["scale_warning"]:
            file_info["warnings"].append(f"{column}: {check['scale_warning']}")
        if not check["ok"]:
            file_info["ok"] = False

    if key == "crop_calendar" and {"sos", "eos"}.issubset(df.columns):
        sos = pd.to_numeric(df["sos"], errors="coerce")
        eos = pd.to_numeric(df["eos"], errors="coerce")
        invalid_order = int((eos < sos).fillna(False).sum())
        file_info["numeric_checks"]["season_order"] = {
            "invalid_rows": invalid_order,
            "ok": invalid_order == 0,
        }
        if invalid_order > 0:
            file_info["ok"] = False

    return file_info


def validate_data(data_root: str | None, crop: str, country: str) -> dict:
    resolved_root = resolve_data_root(data_root=data_root, repo_root=REPO_ROOT)
    target_dir = Path(resolved_root) / crop / country
    report = {
        "schema_version": DATA_CONTRACT_VERSION,
        "crop": crop,
        "country": country,
        "data_root": str(resolved_root),
        "target_dir": str(target_dir),
        "target_exists": target_dir.exists(),
        "required_files": sorted(REQUIRED_FILES.keys()),
        "files": {},
        "summary": {
            "missing_files": [],
            "files_with_errors": [],
        },
        "ok": True,
    }

    if not target_dir.exists():
        report["ok"] = False
        return report

    yield_years = None
    for key, pattern in REQUIRED_FILES.items():
        filename = pattern.format(crop=crop, country=country)
        path = target_dir / filename
        file_info = _validate_file(key, path, crop=crop, country=country)
        report["files"][key] = file_info

        if not file_info["exists"]:
            report["summary"]["missing_files"].append(key)
            report["ok"] = False
            continue

        if not file_info["ok"]:
            report["summary"]["files_with_errors"].append(key)
            report["ok"] = False

        if key == "yield" and file_info["detected_years"]:
            yield_years = set(file_info["detected_years"])

    if yield_years:
        for key, file_info in report["files"].items():
            if key == "yield" or not file_info["exists"] or not file_info["detected_years"]:
                continue
            overlap = sorted(yield_years.intersection(file_info["detected_years"]))
            file_info["year_overlap_with_yield"] = _to_serializable_int_list(overlap)
            if not overlap:
                file_info["ok"] = False
                report["ok"] = False
                if key not in report["summary"]["files_with_errors"]:
                    report["summary"]["files_with_errors"].append(key)

    report["summary"]["missing_files"] = sorted(report["summary"]["missing_files"])
    report["summary"]["files_with_errors"] = sorted(report["summary"]["files_with_errors"])
    return report
