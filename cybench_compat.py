"""Compatibility layer for differing AgML-CY-BENCH config APIs.

YIELD-HUB was written against a CY-BENCH variant that exposed symbols such as
`LOCATION_PROPERTIES`. The current upstream checkout on this machine does not
define all of those names. This module normalizes that surface so the rest of
the repo can import a stable set of config values.
"""

from cybench import config as _cfg


def _get(name, default):
    return getattr(_cfg, name, default)


GDD_BASE_TEMP = _get("GDD_BASE_TEMP", {"maize": 10.0, "wheat": 0.0})
GDD_UPPER_LIMIT = _get("GDD_UPPER_LIMIT", {"maize": 30.0, "wheat": 26.0})
SOIL_PROPERTIES = _get("SOIL_PROPERTIES", [])
LOCATION_PROPERTIES = _get("LOCATION_PROPERTIES", ["latitude", "longitude"])
FORECAST_LEAD_TIME = _get("FORECAST_LEAD_TIME", "middle-of-season")
KEY_LOC = _get("KEY_LOC", "adm_id")
KEY_YEAR = _get("KEY_YEAR", "year")
KEY_TARGET = _get("KEY_TARGET", "yield")
KEY_DATES = _get("KEY_DATES", "dates")
KEY_CROP_SEASON = _get("KEY_CROP_SEASON", "crop_season")
CROP_CALENDAR_DATES = _get("CROP_CALENDAR_DATES", ["sos_date", "eos_date", "cutoff_date"])
