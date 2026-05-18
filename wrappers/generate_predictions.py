import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from yield_hub.cli import main
from yield_hub.predictor import predict as generate_predictions_wrapper

__all__ = ["generate_predictions_wrapper", "main"]


if __name__ == "__main__":
    main()
