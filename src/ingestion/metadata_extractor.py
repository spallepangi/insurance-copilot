"""
Extract plan name and other metadata from file path and content.
"""

import re
from pathlib import Path


# Map filename stems to display plan names
PLAN_NAME_MAP = {
    "bronze": "Bronze",
    "silver": "Silver",
    "gold": "Gold",
    "platinum": "Platinum",
    "bbbronze": "Bronze",
    "bbsilver": "Silver",
    "bbgold": "Gold",
    "bbplatinum": "Platinum",
}


def extract_metadata_from_path(pdf_path: Path) -> dict:
    """
    Derive plan name from file path.
    Returns dict with 'plan_name' and 'source_file'.
    """
    stem = pdf_path.stem.lower()
    plan_name = PLAN_NAME_MAP.get(stem)
    if not plan_name:
        # Try to match bronze/silver/gold/platinum in name
        for key, value in PLAN_NAME_MAP.items():
            if key in stem:
                plan_name = value
                break
    if not plan_name:
        plan_name = stem.replace("_", " ").replace("-", " ").title()
    return {
        "plan_name": plan_name,
        "source_file": pdf_path.name,
    }
