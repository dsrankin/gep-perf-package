
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .core import RunConfig, null_selector, boosted_truth_selector, hh_mass_window_selector


# Selector registry for YAML -> callable resolution
SELECTORS = {
    "null_selector": null_selector,
    "boosted_truth_selector": boosted_truth_selector,
    "hh_mass_window_selector": hh_mass_window_selector,
}


def _parse_bins(obj: Any) -> np.ndarray:
    """
    Accept either:
      - list/tuple of numbers -> np.array
      - dict {start, stop, step} -> np.arange(start, stop + 1e-9, step)
    """
    if isinstance(obj, (list, tuple)):
        return np.asarray(obj, dtype=np.float32)
    if isinstance(obj, dict):
        start = float(obj["start"])
        stop = float(obj["stop"])
        step = float(obj["step"])
        # include stop if it lands on the grid
        n = int(np.floor((stop - start) / step + 1e-9)) + 1
        return (start + step * np.arange(n, dtype=np.float32)).astype(np.float32)
    raise TypeError(f"Unsupported bins format: {type(obj)}")


def _parse_selectors(obj: Any):
    """
    YAML format:
      selectors:
        - name: null_selector
          kwargs: {}
        - name: boosted_truth_selector
          kwargs: {dr_threshold: 0.7}

    Returns (sels, labels)
    """
    if obj is None:
        return [], []
    sels = []
    labels = []
    for item in obj:
        if isinstance(item, str):
            name = item
            kwargs = {}
            label = name
        elif isinstance(item, dict):
            name = item["name"]
            kwargs = item.get("kwargs", {}) or {}
            label = item.get("label", name)
        else:
            raise TypeError(f"Invalid selector entry: {item!r}")

        if name not in SELECTORS:
            raise ValueError(
                f"Unknown selector '{name}'. Known selectors: {sorted(SELECTORS)}"
            )
        fn = SELECTORS[name]
        if kwargs:
            # bind kwargs via a small wrapper (keeps signature simple)
            def _wrapped(pairs, _fn=fn, _kwargs=kwargs):
                return _fn(pairs, **_kwargs)
            sels.append(_wrapped)
        else:
            sels.append(fn)
        labels.append(label)
    return sels, labels


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    data: Dict[str, Any] = yaml.safe_load(path.read_text())

    selectors = data.pop("selectors", None)
    sels, sel_labels = _parse_selectors(selectors)

    # bins
    if "truth_pt_bins" in data:
        data["truth_pt_bins"] = _parse_bins(data["truth_pt_bins"])
    if "truth_eta_bins" in data:
        data["truth_eta_bins"] = _parse_bins(data["truth_eta_bins"])

    # Allow sel_labels override (but keep YAML selectors as source of truth by default)
    data.setdefault("sel_labels", sel_labels)
    data.setdefault("sels", sels)

    # default empty dicts/lists
    data.setdefault("match_dict", {})
    data.setdefault("extra_vars", [])
    data.setdefault("truth_suffix", "")

    # Some YAML authors may provide scalars where lists are expected
    for k in ["signal_files", "background_files", "background_weights", "reco_prefixes", "nobjs", "rates", "triggers"]:
        if k in data and not isinstance(data[k], list):
            data[k] = [data[k]]

    cfg = RunConfig(**data)
    return cfg
