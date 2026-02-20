
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .core import (
    RunConfig,
    null_selector,
    boosted_truth_selector,
    hh_mass_window_selector,
    eratio_selector,
    truth_pt_turnon_var,
    dijet_mass_turnon_var,
)


# Selector registry for YAML -> callable resolution
SELECTORS = {
    "null_selector": null_selector,
    "boosted_truth_selector": boosted_truth_selector,
    "hh_mass_window_selector": hh_mass_window_selector,
    "eratio_selector": eratio_selector,
}

# Turn-on variable registry for YAML -> callable resolution
TURNON_VARIABLES = {
    "truth_pt": truth_pt_turnon_var,
    "dijet_mass": dijet_mass_turnon_var,
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


def _normalize_extra_vars(obj: Any, reco_prefixes: List[str]) -> Dict[str, List[str]]:
    if obj is None:
        return {prefix: [] for prefix in reco_prefixes}
    if isinstance(obj, dict):
        normalized: Dict[str, List[str]] = {}
        for prefix in reco_prefixes:
            value = obj.get(prefix, [])
            if value is None:
                normalized[prefix] = []
            elif isinstance(value, (list, tuple)):
                normalized[prefix] = list(value)
            else:
                normalized[prefix] = [str(value)]
        return normalized
    if isinstance(obj, (list, tuple)):
        return {prefix: list(obj) for prefix in reco_prefixes}
    if isinstance(obj, str):
        return {prefix: [obj] for prefix in reco_prefixes}
    raise TypeError(f"Unsupported extra_vars format: {type(obj)}")



def _split_extra_var_entry(raw_name: str, reco_prefix: str) -> tuple[str, str]:
    branch_name = str(raw_name)
    if branch_name.startswith(f"{reco_prefix}_"):
        branch_name = branch_name[len(reco_prefix) + 1:]
    return str(raw_name), branch_name


def _expand_reco_prefixes_from_extra_vars(
    reco_prefixes: List[str],
    reco_labels: List[str],
    extra_vars: Dict[str, List[str]],
):
    expanded_prefixes: List[str] = []
    expanded_labels: List[str] = []
    expanded_extra_vars: Dict[str, List[str]] = {}
    reco_sources: Dict[str, str] = {}
    extra_var_branches: Dict[str, Dict[str, str]] = {}

    for reco_prefix, reco_label in zip(reco_prefixes, reco_labels):
        extras = extra_vars.get(reco_prefix, [])

        parsed = []
        for raw_name in extras:
            logical_name, branch_name = _split_extra_var_entry(raw_name, reco_prefix)
            parsed.append((logical_name, branch_name))

        grouped_variants: Dict[str, Dict[str, str]] = {}
        for logical_name, branch_name in parsed:
            if "_" not in branch_name:
                continue
            var_name, variant = branch_name.split("_", 1)
            grouped_variants.setdefault(var_name, {})[variant] = branch_name

        split_candidates = [var_name for var_name, variants in grouped_variants.items() if len(variants) > 1]

        if len(split_candidates) != 1:
            expanded_prefixes.append(reco_prefix)
            expanded_labels.append(reco_label)
            expanded_extra_vars[reco_prefix] = [logical_name for logical_name, _ in parsed]
            reco_sources[reco_prefix] = reco_prefix
            extra_var_branches[reco_prefix] = {
                logical_name: branch_name for logical_name, branch_name in parsed
            }
            continue

        split_var = split_candidates[0]
        split_variants = grouped_variants[split_var]

        static_entries = [
            (logical_name, branch_name)
            for logical_name, branch_name in parsed
            if not ("_" in branch_name and branch_name.split("_", 1)[0] == split_var)
        ]

        expanded_prefixes.append(reco_prefix)
        expanded_labels.append(reco_label)
        expanded_extra_vars[reco_prefix] = [name for name, _ in static_entries]
        reco_sources[reco_prefix] = reco_prefix
        extra_var_branches[reco_prefix] = {
            name: branch for name, branch in static_entries
        }

        for variant, branch_name in split_variants.items():
            new_prefix = f"{reco_prefix}_{variant}"
            expanded_prefixes.append(new_prefix)
            expanded_labels.append(f"{reco_label} {variant}" if reco_label else variant)
            expanded_extra_vars[new_prefix] = [split_var] + [name for name, _ in static_entries]
            reco_sources[new_prefix] = reco_prefix
            extra_var_branches[new_prefix] = {
                split_var: branch_name,
                **{name: branch for name, branch in static_entries},
            }

    return expanded_prefixes, expanded_labels, expanded_extra_vars, reco_sources, extra_var_branches


def _parse_turnon_vars(obj: Any, bins_lookup: Dict[str, np.ndarray]):
    """
    YAML format:
      turnon_variables:
        - name: truth_pt
          label: Truth $p_T$ [GeV]
          bins: truth_pt_bins
        - name: dijet_mass
          label: Truth $m_{jj}$ [GeV]
          bins: {start: 0, stop: 500, step: 10}

    Returns (turnon_vars, labels, bins)
    """
    if obj is None:
        return [""], [truth_pt_turnon_var], ["Truth $p_T$ [GeV]"], [bins_lookup["truth_pt_bins"]]

    vars_in = []
    vars_out = []
    labels = []
    bins_out = []
    for item in obj:
        if isinstance(item, str):
            name = item
            kwargs = {}
            label = name
            bins_spec = None
        elif isinstance(item, dict):
            name = item["name"]
            kwargs = item.get("kwargs", {}) or {}
            label = item.get("label", name)
            bins_spec = item.get("bins")
        else:
            raise TypeError(f"Invalid turn-on variable entry: {item!r}")

        if name not in TURNON_VARIABLES:
            raise ValueError(
                f"Unknown turn-on variable '{name}'. Known variables: {sorted(TURNON_VARIABLES)}"
            )
        vars_in.append("" if name=="truth_pt" else "_"+name)
        fn = TURNON_VARIABLES[name]
        if kwargs:
            def _wrapped(pairs, nobj, _fn=fn, _kwargs=kwargs):
                return _fn(pairs, nobj, **_kwargs)
            vars_out.append(_wrapped)
        else:
            vars_out.append(fn)
        labels.append(label)

        if bins_spec is None:
            bins_out.append(bins_lookup["truth_pt_bins"])
        elif isinstance(bins_spec, str):
            if bins_spec not in bins_lookup:
                raise ValueError(
                    f"Unknown bins reference '{bins_spec}'. Known bins: {sorted(bins_lookup)}"
                )
            bins_out.append(bins_lookup[bins_spec])
        else:
            bins_out.append(_parse_bins(bins_spec))

    return vars_in, vars_out, labels, bins_out


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
    data.setdefault("extra_vars", {})
    data.setdefault("truth_suffix", "")
    data.setdefault("reco_labels", data.get("reco_prefixes", []))

    # Some YAML authors may provide scalars where lists are expected
    for k in ["signal_files", "background_files", "background_weights", "reco_prefixes", "reco_labels", "nobjs", "rates", "triggers"]:
        if k in data and not isinstance(data[k], list):
            data[k] = [data[k]]

    original_reco_prefixes = list(data.get("reco_prefixes", []))

    data["extra_vars"] = _normalize_extra_vars(
        data.get("extra_vars"),
        original_reco_prefixes,
    )

    original_pt_reco_names = data.get("match_dict", {}).get("pt_reco_names")

    (
        data["reco_prefixes"],
        data["reco_labels"],
        data["extra_vars"],
        data["reco_sources"],
        data["extra_var_branches"],
    ) = _expand_reco_prefixes_from_extra_vars(
        original_reco_prefixes,
        data.get("reco_labels", []),
        data["extra_vars"],
    )

    data["match_dict"] = dict(data.get("match_dict", {}))
    if original_pt_reco_names is not None:
        if not isinstance(original_pt_reco_names, (list, tuple)):
            raise TypeError(
                f"match_dict.pt_reco_names must be a list/tuple when provided, got: {type(original_pt_reco_names)}"
            )
        if len(original_pt_reco_names) != len(original_reco_prefixes):
            raise ValueError(
                "match_dict.pt_reco_names must match the number of reco_prefixes before extra_vars expansion "
                f"({len(original_pt_reco_names)} != {len(original_reco_prefixes)})"
            )

        pt_name_by_prefix = {
            reco_prefix: str(pt_name)
            for reco_prefix, pt_name in zip(original_reco_prefixes, original_pt_reco_names)
        }
        data["match_dict"]["pt_reco_names"] = [
            pt_name_by_prefix[data["reco_sources"].get(prefix, prefix)]
            for prefix in data["reco_prefixes"]
        ]

    turnon_variables = data.pop("turnon_variables", None)
    turnon_vars, turnon_fns, turnon_labels, turnon_bins = _parse_turnon_vars(
        turnon_variables,
        {
            "truth_pt_bins": data["truth_pt_bins"],
            "truth_eta_bins": data["truth_eta_bins"],
        },
    )
    data.setdefault("turnon_fns", turnon_fns)
    data.setdefault("turnon_vars", turnon_vars)
    data.setdefault("turnon_var_labels", turnon_labels)
    data.setdefault("turnon_bins", turnon_bins)

    cfg = RunConfig(**data)
    return cfg
