import os
import sys
from pathlib import Path
from typing import Optional


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DEFAULT_LOCAL_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_CYBENCH_CANDIDATES = [
    REPO_ROOT.parent / "crop_yield_prediction" / "cybench" / "AgML-CY-BENCH",
    REPO_ROOT.parent / "AgML-CY-BENCH",
    REPO_ROOT.parent.parent / "crop_yield_prediction" / "cybench" / "AgML-CY-BENCH",
]


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def load_repo_env(repo_root: Optional[Path] = None) -> None:
    root = repo_root or REPO_ROOT
    env_path = root / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'").strip('"'))


def get_hf_token(repo_root: Optional[Path] = None) -> str:
    load_repo_env(repo_root)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Hugging Face token. Set HF_TOKEN or HUGGINGFACE_HUB_TOKEN in .env."
        )
    return token


def resolve_data_root(data_root: Optional[str] = None, repo_root: Optional[Path] = None) -> Path:
    root = repo_root or REPO_ROOT
    load_repo_env(root)

    configured = data_root or os.environ.get("YIELD_HUB_DATA_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.append(DEFAULT_LOCAL_DATA_ROOT)

    for candidate in candidates:
        if (candidate / "maize").exists() or (candidate / "wheat").exists():
            return candidate

    return DEFAULT_LOCAL_DATA_ROOT


def resolve_cybench_root(cybench_root: Optional[str] = None, repo_root: Optional[Path] = None) -> Path:
    load_repo_env(repo_root)
    configured = cybench_root or os.environ.get("CYBENCH_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend(DEFAULT_CYBENCH_CANDIDATES)

    for candidate in candidates:
        if (candidate / "cybench" / "config.py").exists():
            return candidate

    raise ModuleNotFoundError(
        "Could not locate the external CY-BENCH package. Set CYBENCH_ROOT in .env "
        "to the AgML-CY-BENCH repository root containing cybench/config.py."
    )


def configure_runtime_paths(cybench_root: Optional[str] = None, repo_root: Optional[Path] = None) -> Path:
    root = repo_root or REPO_ROOT
    load_repo_env(root)

    runtime_paths = [
        str(root),
        str(root / "process"),
        str(root / "architectures"),
    ]
    for path in reversed(runtime_paths):
        _prepend_sys_path(Path(path))

    resolved_cybench_root = resolve_cybench_root(cybench_root=cybench_root, repo_root=root)
    _prepend_sys_path(resolved_cybench_root)

    return resolved_cybench_root
