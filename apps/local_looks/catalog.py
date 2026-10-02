"""Small immutable runtime catalog; contains bindings, never LUT pixel tables.

The candidate inventory is not proof of official availability or host recipes.
The UI can use a plain preview flag without displaying implementation notes.
Resource hashes/grid sizes are provenance metadata, not a requirement imposed
on explicitly supplied synthetic or custom test resources. No cube is opened
when this module is imported, the engine is constructed, or a photo is loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType


@dataclass(frozen=True)
class LookSpec:
    id: str
    title: str
    default_strength: int
    adjustable: bool


CATALOG_PATH = Path(__file__).with_name("assets") / "look-catalog.json"


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _validate_cube(cube):
    name = cube["filename"]
    if (not isinstance(name, str) or not name.endswith(".cube") or name.startswith(".")
            or any(character in name for character in ("/", "\\", ":", "\x00"))):
        raise ValueError("Catalog cube must be a plain .cube filename within resource_dir")
    if (type(cube["grid_size"]) is not int or cube["grid_size"] < 2
            or type(cube["restore_half"]) is not bool):
        raise ValueError("Invalid catalog cube grid/precision policy")
    if len(cube["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in cube["sha256"]):
        raise ValueError("Invalid reference cube SHA-256")


def _load_catalog():
    try:
        raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        if raw["schema"] != 1:
            raise ValueError("Unsupported catalog schema")
        groups = {group["id"] for group in raw["groups"]}
        filters = {item["id"] for item in raw["color_filters"]}
        if len(filters) != len(raw["color_filters"]):
            raise ValueError("Duplicate catalog color filter")
        for item in raw["color_filters"]:
            _validate_cube(item["cube"])
        seen = set()
        for item in raw["looks"]:
            if item["id"] in seen or item["group"] not in groups:
                raise ValueError("Duplicate Look or unknown group")
            seen.add(item["id"])
            if (item["input_space"] not in ("srgb", "display-p3")
                    or item["output_space"] not in ("srgb", "display-p3")
                    or item["recipe"] not in ("identity", "source_to_primary", "secondary_to_primary")):
                raise ValueError("Unsupported catalog domain/recipe")
            if not set(item["filter_options"]).issubset(filters):
                raise ValueError("Unknown color-filter attachment")
            if item["recipe"] == "identity":
                if item["id"] != "original" or item["primary_cube"] is not None or item["secondary_cube"] is not None:
                    raise ValueError("Only Original can have the identity recipe")
            else:
                _validate_cube(item["primary_cube"])
                if item["recipe"] == "secondary_to_primary":
                    _validate_cube(item["secondary_cube"])
                elif item["secondary_cube"] is not None:
                    raise ValueError("Source blend must have exactly one primary table")
        if "original" not in seen:
            raise ValueError("Missing Original entry")
        return _freeze(raw)
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise RuntimeError(f"Local Looks runtime catalog is missing or invalid: {error}") from error


CATALOG = _load_catalog()
LOOK_BINDINGS = MappingProxyType({item["id"]: item for item in CATALOG["looks"]})
LOOKS = tuple(LookSpec(item["id"], item["title"], item["default_strength"], item["adjustable"])
              for item in CATALOG["looks"])
LOOK_BY_ID = MappingProxyType({look.id: look for look in LOOKS})
LOOK_GROUPS = MappingProxyType({item["id"]: item["group"] for item in CATALOG["looks"]})
GROUP_TITLES = MappingProxyType({item["id"]: item["title"] for item in CATALOG["groups"]})
LOOK_PREVIEW = MappingProxyType({item["id"]: item["preview"] for item in CATALOG["looks"]})
LOOK_STRENGTH_RULES = MappingProxyType({item["id"]: item["strength_rule"] for item in CATALOG["looks"]})
LOOK_FILTER_OPTIONS = MappingProxyType({item["id"]: item["filter_options"] for item in CATALOG["looks"]})
COLOR_FILTERS = MappingProxyType({item["id"]: item for item in CATALOG["color_filters"]})


def look_preview(look_id: str, strength: float | None = None, color_filter: str | None = None) -> bool:
    """Preview status for a selected recipe; LOOK_PREVIEW covers the whole range.

    Old Eternal/Vivid100 remain verified without attachments; intermediate
    source blends and all new candidates are preview. Steve's continuous
    interpolation remains the existing observed two-endpoint rule.
    """
    binding = LOOK_BINDINGS[look_id]
    if color_filter is not None:
        return True
    if strength is None:
        strength = binding["default_strength"]
    return bool(binding["preview_default"] or
                (binding["preview_unverified_strengths"] and strength not in binding["verified_strengths"]))
