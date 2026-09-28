import uproot
import awkward as ak
import numpy as np
import vector
from tqdm import tqdm
import math
import gc
import matplotlib.pyplot as plt

vector.register_awkward()

from dataclasses import dataclass, field, replace
from typing import Callable, Optional

import os

DEFAULT_PLOTDIR = 'perf_plots'
DEFAULT_RESDIR = 'perf_results'

plotdir = os.environ.get('GEP_PERF_PLOTDIR', DEFAULT_PLOTDIR)
resdir = os.environ.get('GEP_PERF_RESDIR', DEFAULT_RESDIR)
os.makedirs(plotdir, exist_ok=True)
os.makedirs(plotdir+'/debug', exist_ok=True)
os.makedirs(resdir, exist_ok=True)

EFFICIENCY_MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*')
MARKER_SIZE_SCALE = 1.5
DEFAULT_MARKER_SIZE = 4 * MARKER_SIZE_SCALE
DEFAULT_SCATTER_SIZE = (plt.rcParams['lines.markersize'] * MARKER_SIZE_SCALE) ** 2
LEGEND_TOP_MARGIN = 0.78


def marker_alpha(alpha: float) -> float:
    if alpha in (0.0, 1.0):
        return alpha
    return 1 - (1 - alpha) * 0.5


def place_legend_above(ax=None, *, title=None, max_cols=3):
    """
    Place the legend in a centered block above the axes, wrapping into
    multiple rows (at most ``max_cols`` columns) instead of forcing every
    entry into one expanded row: with more than a couple of long labels
    (as in the per-collection overlay plots) an expand-to-full-width single
    row overlaps entries on top of each other. Column widths are sized by
    matplotlib from the actual label text, so nothing is cropped or overlaps.
    """
    ax = ax or plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    ncol = min(len(handles), max_cols)
    nrows = -(-len(handles) // ncol)  # ceil division
    legend = ax.legend(
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        borderaxespad=0,
        title=title,
        fontsize=8 if len(handles) > max_cols else None,
    )
    # more rows need more headroom above the axes
    top = LEGEND_TOP_MARGIN - 0.10 * (nrows - 1)
    ax.figure.subplots_adjust(top=max(top, 0.35))
    return legend


def sanitize_plot_component(name: str) -> str:
    return str(name).replace(os.sep, '_')


def ensure_plot_parent(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def build_plot_path(*parts: str) -> str:
    return ensure_plot_parent(os.path.join(plotdir, *parts))


def build_debug_plot_path(collection: Optional[str], filename: str) -> str:
    if collection:
        return build_plot_path('debug', sanitize_plot_component(collection), filename)
    return build_plot_path('debug', filename)


def get_efficiency_marker(index: int) -> str:
    return EFFICIENCY_MARKERS[index % len(EFFICIENCY_MARKERS)]

# this will also swap from processing batches with awkward to a standard event loop

@dataclass
class RunConfig:
    name: str
    signal_files: list[str]
    background_files: list[str]
    background_weights: list[float]
    reco_prefixes: list[str]
    reco_labels: Optional[list[str]]
    truth_prefix: str
    truth_suffix: str
    match_dict: dict
    nobjs: list[int]
    sels: list[Callable[[ak.Array], np.ndarray]] #the selections should take in and return an awkward array
    sel_labels: list[str]
    rate_sels: list[Callable[[ak.Array], np.ndarray]]
    rate_sel_labels: list[str]
    truth_pt_bins: np.ndarray
    truth_eta_bins: np.ndarray
    do_rho_sub: bool
    rates: list[float]
    triggers :list[list[float]]
    extra_vars: dict[str, list[str]]
    turnon_vars: list[str]
    turnon_fns: list[Callable[[ak.Array, int], ak.Array]]
    turnon_var_labels: list[str]
    turnon_bins: list[np.ndarray]
    spline_lambdas: dict[str, float] = field(default_factory=dict)
    reco_sources: dict[str, str] = field(default_factory=dict)
    extra_var_branches: dict[str, dict[str, str]] = field(default_factory=dict)
    tree: str = "ntuple"
    dr_max: float = 0.2
    reco_pt_min: float = 10.
    truth_pt_min: float = 10.
    pt_min: float = 10.
    reco_iso_dr: float = 0.4
    truth_iso_dr: float = 0.4
        
    def __post_init__(self):
        if self.reco_labels is None:
            self.reco_labels = list(self.reco_prefixes)
        if len(self.background_files)!=len(self.background_weights):
            raise ValueError(f"Background files ({len(self.background_files)}) and weights ({len(self.background_weights)}) must be the same length")
        if len(self.reco_labels)!=len(self.reco_prefixes):
            raise ValueError(f"Reconstruction labels ({len(self.reco_labels)}) and reco prefixes ({len(self.reco_prefixes)}) must be the same length")
        if len(self.nobjs)!=len(self.rates):
            raise ValueError(f"Number of objects ({len(self.nobjs)}) and rates ({len(self.rates)}) must be the same length")
        if len(self.sels)!=len(self.sel_labels):
            raise ValueError(f"Number of selectors ({len(self.sels)}) and selector labels ({len(self.sel_labels)}) must be the same length")
        if len(self.rate_sels)!=len(self.rate_sel_labels):
            raise ValueError(f"Number of selectors ({len(self.rate_sels)}) and selector labels ({len(self.rate_sel_labels)}) must be the same length")
        if len(self.nobjs)!=len(self.triggers):
            raise ValueError(f"Number of objects ({len(self.nobjs)}) and triggers ({len(self.triggers)}) must be the same length")
        if len(self.turnon_vars)!=len(self.turnon_fns):
            raise ValueError(f"Number of turn-on variables ({len(self.turnon_vars)}) and functions ({len(self.turnon_fns)}) must be the same length")
        if len(self.turnon_vars)!=len(self.turnon_var_labels):
            raise ValueError(f"Number of turn-on variables ({len(self.turnon_vars)}) and labels ({len(self.turnon_var_labels)}) must be the same length")
        if len(self.turnon_vars)!=len(self.turnon_bins):
            raise ValueError(f"Number of turn-on variables ({len(self.turnon_vars)}) and bins ({len(self.turnon_bins)}) must be the same length")

@dataclass
class RunResult:
    name: str
    reco: str
    reco_label: str
    nobj:int
    sel_label: str
    rate_sel_label: str
    fixrate: bool
    threshold: float
    rate: float
    truth_pt_bins: np.ndarray
    truth_eta_bins: np.ndarray
    signal_efficiency: np.ndarray
    signal_efficiency_error: np.ndarray
    full_sig_efficiency: np.ndarray
    full_sig_efficiency_error: np.ndarray
    full_bkg_efficiency: np.ndarray
    full_bkg_efficiency_error: np.ndarray
    response_uncorr: dict
    response_corr: dict
    resol_uncorr: dict
    resol_corr: dict
    turnon_var: str
    turnon_label: str
    turnon_bins: np.ndarray


def result_reco_label(result: RunResult) -> str:
    return getattr(result, "reco_label", result.reco)

def delta_phi(phi1,phi2):
    dphi = phi1-phi2
    dphi = dphi+2*np.pi*(dphi<=-np.pi).astype(np.float32)
    dphi = dphi-2*np.pi*(dphi>np.pi).astype(np.float32)
    return dphi


def delta_phi_awkward(phi1, phi2):
    """Awkward array version of delta_phi"""
    dphi = phi1 - phi2
    return ak.where(dphi > np.pi, dphi - 2*np.pi,
                    ak.where(dphi < -np.pi, dphi + 2*np.pi, dphi))


def _wrap_dphi(dphi):
    """Wrap a numpy array of phi differences into (-pi, pi] (same convention as delta_phi_awkward)."""
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    return np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)


def _jagged_to_flat(arr, scale=None, dtype=np.float32):
    """
    Flatten a jagged (events x var) awkward array into (flat numpy array, counts per event).
    The flat array is converted to ``dtype`` and optionally divided by ``scale``.
    """
    counts = np.asarray(ak.to_numpy(ak.num(arr, axis=1)), dtype=np.int64)
    flat = np.asarray(ak.to_numpy(ak.flatten(arr, axis=1)))
    if scale is not None:
        flat = flat / scale
    return np.asarray(flat, dtype=dtype), counts


def _apply_flat_mask(mask, counts, *arrays):
    """Apply a per-object boolean mask to flat arrays and recompute the per-event counts."""
    nev = counts.shape[0]
    ev = np.repeat(np.arange(nev, dtype=np.int64), counts)
    new_counts = np.bincount(ev[mask], minlength=nev).astype(np.int64)
    return tuple(a[mask] for a in arrays) + (new_counts,)


def compute_isolation_flat(eta, phi, counts, iso_dr):
    """
    Isolation mask for flat (eta, phi) arrays grouped by ``counts`` per event.
    Returns True if no OTHER object in the same event is within ``iso_dr``
    (pairs at exactly zero distance are ignored, as in the original 3D version).

    Objects are sorted by eta inside each event so that only neighbours with
    |d_eta| < iso_dr are ever compared: O(n * window) instead of O(n^2) per event.
    """
    n = eta.shape[0]
    if n == 0 or iso_dr <= 0.:
        return np.ones(n, dtype=bool)

    ev = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    order = np.lexsort((eta, ev))
    eta_s = eta[order]
    phi_s = phi[order]
    ev_s = ev[order]

    has_nearby = np.zeros(n, dtype=bool)
    iso2 = iso_dr * iso_dr
    k = 1
    while k < n:
        d_eta = eta_s[k:] - eta_s[:-k]
        cand = np.flatnonzero((ev_s[k:] == ev_s[:-k]) & (d_eta < iso_dr))
        if cand.size == 0:
            # eta is sorted within each event, so larger offsets cannot be closer
            break
        d_phi = _wrap_dphi(phi_s[cand + k] - phi_s[cand])
        dr2 = d_eta[cand] ** 2 + d_phi ** 2
        close = cand[(dr2 < iso2) & (dr2 > 0)]
        has_nearby[close] = True
        has_nearby[close + k] = True
        k += 1

    isolated = np.empty(n, dtype=bool)
    isolated[order] = ~has_nearby
    return isolated


def match_flat(r_pt, r_eta, r_phi, r_counts, t_pt, t_eta, t_phi, t_counts,
               dr_max, riso, tiso, ptmin, k=4, extra_vars=None):
    """
    Non-greedy truth->reco matching on flat numpy arrays (events described by counts).
    Each truth object matches its closest isolated reco object within dr_max.

    Returns a dict of flat float32 arrays plus ``counts`` per event. Per event the
    output rows are: one row per truth object (reco values NaN when unmatched),
    followed by up to ``k`` leading-pt unmatched reco objects (truth values NaN).
    """
    nan32 = np.float32(np.nan)
    nev = r_counts.shape[0]
    n_r = r_pt.shape[0]
    n_t = t_pt.shape[0]

    r_start = np.cumsum(r_counts) - r_counts
    t_start = np.cumsum(t_counts) - t_counts
    t_ev = np.repeat(np.arange(nev, dtype=np.int64), t_counts)

    # ---- truth x reco pairs (per truth: all reco in the same event) ----
    seg_len = r_counts[t_ev]                       # reco multiplicity of each truth's event
    seg_start = np.cumsum(seg_len) - seg_len
    n_pairs = int(seg_len.sum())
    p_t = np.repeat(np.arange(n_t, dtype=np.int64), seg_len)
    p_r = r_start[t_ev[p_t]] + (np.arange(n_pairs, dtype=np.int64) - seg_start[p_t])

    d_eta = t_eta[p_t] - r_eta[p_r]
    d_phi = _wrap_dphi(t_phi[p_t] - r_phi[p_r])
    dr2 = d_eta ** 2 + d_phi ** 2
    del d_eta, d_phi
    valid = (dr2 < (dr_max ** 2)) & riso[p_r] & tiso[p_t]
    dr2 = np.where(valid, dr2, np.inf)
    del valid

    # ---- closest reco per truth (first minimum, +inf means no match) ----
    min_dr2 = np.full(n_t, np.inf, dtype=dr2.dtype)
    arg_pair = np.zeros(n_t, dtype=np.int64)
    has_seg = seg_len > 0
    if n_pairs > 0:
        st = seg_start[has_seg]
        min_dr2[has_seg] = np.minimum.reduceat(dr2, st)
        is_min = dr2 == np.repeat(min_dr2[has_seg], seg_len[has_seg])
        arg_pair[has_seg] = np.minimum.reduceat(
            np.where(is_min, np.arange(n_pairs, dtype=np.int64), n_pairs), st
        )
        del is_min
    del dr2
    has_match = min_dr2 < np.inf
    # global reco index matched to each truth (0 placeholder where unmatched)
    match_r = p_r[arg_pair] if n_pairs > 0 else np.zeros(n_t, dtype=np.int64)

    def gather(a, idx):
        return a[idx] if a.shape[0] > 0 else np.full(idx.shape[0], np.nan, dtype=np.float32)

    matched_pt = gather(r_pt, match_r)
    has_match &= matched_pt > ptmin
    matched_pt = np.where(has_match, matched_pt, nan32)
    matched_eta = np.where(has_match, gather(r_eta, match_r), nan32)
    matched_phi = np.where(has_match, gather(r_phi, match_r), nan32)

    reco_was_matched = np.zeros(n_r, dtype=bool)
    reco_was_matched[match_r[has_match]] = True

    # ---- unmatched reco: keep the k leading in pt per event ----
    keep_idx = np.flatnonzero((~reco_was_matched) & (r_pt > ptmin))
    r_ev = np.repeat(np.arange(nev, dtype=np.int64), r_counts)
    u_ev = r_ev[keep_idx]
    order = np.lexsort((-r_pt[keep_idx], u_ev))   # stable: event, then descending pt
    keep_idx = keep_idx[order]
    u_ev = u_ev[order]
    u_all = np.bincount(u_ev, minlength=nev)
    rank = np.arange(keep_idx.shape[0], dtype=np.int64) - (np.cumsum(u_all) - u_all)[u_ev]
    top = rank < k
    u_idx = keep_idx[top]
    u_ev = u_ev[top]
    u_rank = rank[top]
    u_counts = np.minimum(u_all, k).astype(np.int64)

    # ---- assemble output rows: truth rows first, then unmatched reco rows ----
    out_counts = t_counts + u_counts
    out_start = np.cumsum(out_counts) - out_counts
    t_pos = out_start[t_ev] + (np.arange(n_t, dtype=np.int64) - t_start[t_ev])
    u_pos = out_start[u_ev] + t_counts[u_ev] + u_rank
    n_out = int(out_counts.sum())
    u_nan = np.full(u_idx.shape[0], np.nan, dtype=np.float32)

    def assemble(t_vals, u_vals):
        out = np.empty(n_out, dtype=np.float32)
        out[t_pos] = t_vals
        out[u_pos] = u_vals
        return out

    output = {
        'counts': out_counts,
        'reco_pt': assemble(matched_pt, r_pt[u_idx]),
        'reco_eta': assemble(matched_eta, r_eta[u_idx]),
        'reco_phi': assemble(matched_phi, r_phi[u_idx]),
        'truth_pt': assemble(t_pt, u_nan),
        'truth_eta': assemble(t_eta, u_nan),
        'truth_phi': assemble(t_phi, u_nan),
    }
    if extra_vars:
        for extra_name, r_extra in extra_vars.items():
            matched_extra = np.where(has_match, gather(r_extra, match_r), nan32)
            output[f'reco_{extra_name}'] = assemble(matched_extra, r_extra[u_idx])
    return output


def match_chunk_vectorized(chunk, reco_prefixes, reco_branches, truth_branches, dr_max,
                          reco_iso_dr=0.4, truth_iso_dr=0.4,
                          reco_pt_min=None, truth_pt_min=None, pt_min=None,
                          extra_vars_by_prefix=None, reco_extra_branches=None):
    """
    Matching for an entire chunk of events.
    Non-greedy: each truth object matches to its closest reco object within dr_max.
    Returns {reco_prefix: {'counts': ..., 'reco_pt': ..., ..., 'reco_<extra>': ...}}
    of flat float32 numpy arrays (see match_flat).
    """
    # Extract truth arrays once (shared across all reco prefixes); pt is MeV -> GeV
    t_pt, t_counts = _jagged_to_flat(chunk[truth_branches[0]], scale=1000.0)
    t_eta, _ = _jagged_to_flat(chunk[truth_branches[1]])
    t_phi, _ = _jagged_to_flat(chunk[truth_branches[2]])

    if pt_min is None:
        pt_min = -1.

    # Apply truth pT cuts if specified
    if truth_pt_min is not None:
        t_pt, t_eta, t_phi, t_counts = _apply_flat_mask(t_pt > truth_pt_min, t_counts, t_pt, t_eta, t_phi)

    # Truth isolation
    t_isolated = compute_isolation_flat(t_eta, t_phi, t_counts, truth_iso_dr)

    if extra_vars_by_prefix is None:
        extra_vars_by_prefix = {}
    if reco_extra_branches is None:
        reco_extra_branches = {
            reco_prefix: {
                extra_name: f"{reco_prefix}_{extra_name}"
                for extra_name in extra_vars_by_prefix.get(reco_prefix, [])
            }
            for reco_prefix in reco_prefixes
        }

    results = {}
    for reco_prefix in reco_prefixes:
        r_pt, r_counts = _jagged_to_flat(chunk[reco_branches[reco_prefix][0]], scale=1000.0)
        r_eta, _ = _jagged_to_flat(chunk[reco_branches[reco_prefix][1]])
        r_phi, _ = _jagged_to_flat(chunk[reco_branches[reco_prefix][2]])
        extra_names = list(extra_vars_by_prefix.get(reco_prefix, []))
        r_extra_list = [
            _jagged_to_flat(chunk[reco_extra_branches[reco_prefix][extra_name]])[0]
            for extra_name in extra_names
        ]

        # Apply pT cut
        r_mask = r_pt > pt_min
        r_pt, r_eta, r_phi, *r_extra_list, r_counts = _apply_flat_mask(
            r_mask, r_counts, r_pt, r_eta, r_phi, *r_extra_list
        )
        del r_mask

        # Reco isolation (+ minimum reco pt for matching)
        r_isolated = compute_isolation_flat(r_eta, r_phi, r_counts, reco_iso_dr)
        if reco_pt_min is not None:
            r_isolated &= r_pt > reco_pt_min

        results[reco_prefix] = match_flat(
            r_pt, r_eta, r_phi, r_counts,
            t_pt, t_eta, t_phi, t_counts,
            dr_max, r_isolated, t_isolated, pt_min,
            extra_vars=dict(zip(extra_names, r_extra_list)),
        )

    return results


def _normalize_extra_vars(reco_prefixes, extra_vars):
    if extra_vars is None:
        return {prefix: [] for prefix in reco_prefixes}
    if isinstance(extra_vars, dict):
        normalized = {}
        for prefix in reco_prefixes:
            value = extra_vars.get(prefix, [])
            if value is None:
                normalized[prefix] = []
            elif isinstance(value, (list, tuple)):
                normalized[prefix] = list(value)
            else:
                normalized[prefix] = [str(value)]
        return normalized
    if isinstance(extra_vars, (list, tuple)):
        return {prefix: list(extra_vars) for prefix in reco_prefixes}
    if isinstance(extra_vars, str):
        return {prefix: [extra_vars] for prefix in reco_prefixes}
    raise TypeError(f"Unsupported extra_vars format: {type(extra_vars)}")



def match_reco_truth(
    files,
    weights=None,
    reco_prefixes=["AntiKt4GEPCellsTowerAlgJets"],
    truth_prefix="AntiKt4TruthJets",
    truth_suffix="",
    pt_reco_names=None,
    pt_truth_name="pt",
    reco_pt_min=20.,
    truth_pt_min=20.,
    pt_min=20.,
    reco_iso_dr=0.4,
    truth_iso_dr=0.4,
    eta_name="eta",
    phi_name="phi",
    dr_max=0.2,
    tree_name="ntuple",
    extra_vars=None,
    reco_sources=None,
    extra_var_branches=None,
    met_mode=False,
    met_reco_ex_name="ex",
    met_reco_ey_name="ey",
    met_special_prefixes=None,
    step_size=10000,
):

    if weights is None:
        weights = [1.0] * len(files)

    if pt_reco_names is None:
        default_pt_name = "et" if met_mode else "pt"
        pt_reco_names = [default_pt_name] * len(reco_prefixes)
    elif not isinstance(pt_reco_names, (list, tuple)):
        raise TypeError(f"pt_reco_names must be a list or tuple, got: {type(pt_reco_names)}")
    elif len(pt_reco_names) != len(reco_prefixes):
        raise ValueError(
            f"pt_reco_names ({len(pt_reco_names)}) and reco prefixes ({len(reco_prefixes)}) must be the same length"
        )

    extra_vars_by_prefix = _normalize_extra_vars(reco_prefixes, extra_vars)
    reco_sources = reco_sources or {}
    extra_var_branches = extra_var_branches or {}
    special_met_prefixes = set(met_special_prefixes or [])

    truth_branches = [
        f"{truth_prefix}_{pt_truth_name}{truth_suffix}",
        f"{truth_prefix}_{eta_name}{truth_suffix}",
        f"{truth_prefix}_{phi_name}{truth_suffix}",
    ]

    with uproot.open(files[0]) as ftmp:
        available_branches = set(ftmp[tree_name].keys())

    reco_extra_branches = {}
    reco_branches = {}
    reco_metphi_mode = {}
    for i, reco_prefix in enumerate(reco_prefixes):
        mapped_source_prefix = reco_sources.get(reco_prefix, reco_prefix)
        source_prefix = mapped_source_prefix

        mapped_pt_branch = f"{mapped_source_prefix}_{pt_reco_names[i]}"
        expanded_pt_branch = f"{reco_prefix}_{pt_reco_names[i]}"
        if mapped_source_prefix != reco_prefix:
            if expanded_pt_branch in available_branches or mapped_pt_branch not in available_branches:
                source_prefix = reco_prefix

        extra_branch_map = {}
        for extra_var in extra_vars_by_prefix[reco_prefix]:
            source_extra_var = extra_var_branches.get(reco_prefix, {}).get(extra_var, extra_var)
            extra_branch = f"{source_prefix}_{source_extra_var}"
            if extra_branch not in available_branches and mapped_source_prefix != source_prefix:
                fallback_branch = f"{mapped_source_prefix}_{source_extra_var}"
                if fallback_branch in available_branches:
                    extra_branch = fallback_branch
            extra_branch_map[extra_var] = extra_branch
        reco_extra_branches[reco_prefix] = extra_branch_map

        reco_branches[reco_prefix] = [
            f"{source_prefix}_{pt_reco_names[i]}",
            f"{source_prefix}_{eta_name}",
            f"{source_prefix}_{phi_name}",
        ]
        if met_mode:
            reco_branches[reco_prefix][1] = f"{source_prefix}_{met_reco_ex_name}"
            reco_branches[reco_prefix][2] = f"{source_prefix}_{met_reco_ey_name}"
            reco_metphi_mode[reco_prefix] = False
            if reco_branches[reco_prefix][1] not in available_branches:
                met_branch = f"{source_prefix}_met"
                met_phi_branch = f"{source_prefix}_metPhi"
                if (
                    (reco_prefix in special_met_prefixes or source_prefix in special_met_prefixes)
                    and met_branch in available_branches
                    and met_phi_branch in available_branches
                ):
                    reco_branches[reco_prefix][0] = met_branch
                    reco_branches[reco_prefix][1] = met_phi_branch
                    reco_branches[reco_prefix][2] = met_phi_branch
                    reco_metphi_mode[reco_prefix] = True

    branches = truth_branches + ["weight", "gFEX_rho"]
    for reco_prefix in reco_prefixes:
        branches.extend(reco_branches[reco_prefix])
        branches.extend(reco_extra_branches[reco_prefix].values())
    branches = list(dict.fromkeys(branches))

    # Accumulators: per reco prefix, lists of flat numpy arrays (one per chunk) + per-event counts
    base_fields = ["reco_pt", "reco_eta", "reco_phi", "truth_pt", "truth_eta", "truth_phi"]
    fields_by_prefix = {
        reco_prefix: base_fields + [f"reco_{extra_name}" for extra_name in extra_vars_by_prefix[reco_prefix]]
        for reco_prefix in reco_prefixes
    }
    acc = {
        reco_prefix: {name: [] for name in ["counts"] + fields_by_prefix[reco_prefix]}
        for reco_prefix in reco_prefixes
    }
    event_ids = []
    event_weights = []
    event_rhos = []

    def process_file(filename, weight):
        with uproot.open(filename) as ftmp:
            n_events = ftmp[tree_name].num_entries
        total_chunks = math.ceil(n_events / step_size)
        print(f"{filename}: {total_chunks} chunks")

        it = uproot.iterate(
            f"{filename}:{tree_name}",
            branches,
            step_size=step_size,
            library="ak",
        )

        event_offset = 0
        file_weights = []

        for chunk in tqdm(it, total=total_chunks, desc=f"{filename}"):
            n_events_chunk = len(chunk[truth_branches[0]])

            if met_mode:
                truth_pt = ak.values_astype(chunk[truth_branches[0]] / 1000.0, np.float32)
                truth_phi = ak.values_astype(chunk[truth_branches[2]], np.float32)
                truth_px = truth_pt * np.cos(truth_phi)
                truth_py = truth_pt * np.sin(truth_phi)
                truth_met_x = -ak.sum(truth_px, axis=1)
                truth_met_y = -ak.sum(truth_py, axis=1)
                truth_met = np.asarray(ak.to_numpy(np.sqrt(truth_met_x**2 + truth_met_y**2)), dtype=np.float32)
                truth_met_phi = np.asarray(ak.to_numpy(np.arctan2(truth_met_y, truth_met_x)), dtype=np.float32)
                truth_eta = np.zeros(n_events_chunk, dtype=np.float32)
                ones = np.ones(n_events_chunk, dtype=np.int64)

                results = {}
                for reco_prefix in reco_prefixes:
                    reco_et = np.asarray(ak.to_numpy(chunk[reco_branches[reco_prefix][0]] / 1000.0), dtype=np.float32)
                    if reco_metphi_mode.get(reco_prefix, False):
                        reco_phi = np.asarray(ak.to_numpy(-chunk[reco_branches[reco_prefix][1]]), dtype=np.float32)
                    else:
                        reco_ex = np.asarray(ak.to_numpy(chunk[reco_branches[reco_prefix][1]]), dtype=np.float32)
                        reco_ey = np.asarray(ak.to_numpy(chunk[reco_branches[reco_prefix][2]]), dtype=np.float32)
                        reco_et = np.asarray(np.sqrt(np.power(reco_ex, 2) + np.power(reco_ey, 2)), dtype=np.float32)
                        reco_phi = np.asarray(np.arctan2(reco_ey, reco_ex), dtype=np.float32)

                    out = {
                        "counts": ones,
                        "reco_pt": reco_et,
                        "reco_eta": truth_eta,
                        "reco_phi": reco_phi,
                        "truth_pt": truth_met,
                        "truth_eta": truth_eta,
                        "truth_phi": truth_met_phi,
                    }
                    for extra_name, extra_branch in reco_extra_branches[reco_prefix].items():
                        out[f"reco_{extra_name}"] = np.asarray(ak.to_numpy(chunk[extra_branch]), dtype=np.float32)
                    results[reco_prefix] = out
            else:
                results = match_chunk_vectorized(
                    chunk,
                    reco_prefixes,
                    reco_branches,
                    truth_branches,
                    dr_max,
                    reco_iso_dr,
                    truth_iso_dr,
                    reco_pt_min,
                    truth_pt_min,
                    pt_min,
                    extra_vars_by_prefix,
                    reco_extra_branches,
                )

            for reco_prefix in reco_prefixes:
                for name, values in results[reco_prefix].items():
                    acc[reco_prefix][name].append(values)

            event_ids.append(np.arange(event_offset, event_offset + n_events_chunk, dtype=np.int64))
            file_weights.append(np.asarray(ak.to_numpy(chunk["weight"]), dtype=np.float64))
            event_rhos.append(
                np.asarray(
                    ak.to_numpy(ak.fill_none(ak.pad_none(chunk["gFEX_rho"], 3, axis=1, clip=True), 0.)),
                    dtype=np.float64,
                )
            )
            event_offset += n_events_chunk

            del chunk, results
            gc.collect()

        file_weights = np.concatenate(file_weights) if file_weights else np.zeros(0, dtype=np.float64)
        total_weight = np.sum(file_weights)
        event_weights.append(file_weights * weight / total_weight)

    for i, f in enumerate(files):
        process_file(f, weights[i])

    event_ids = np.concatenate(event_ids)
    event_weights = np.concatenate(event_weights)
    event_rhos = np.concatenate(event_rhos, axis=0)

    output = {}
    for reco_prefix in reco_prefixes:
        counts = np.concatenate(acc[reco_prefix].pop("counts"))
        fields = {
            "event": ak.Array(event_ids),
            "weight": ak.Array(event_weights),
            "rho": ak.Array(event_rhos),
        }
        for name in fields_by_prefix[reco_prefix]:
            # concatenate one field at a time and drop the chunk list right away to limit peak memory
            fields[name] = ak.unflatten(np.concatenate(acc[reco_prefix].pop(name)), counts)
        output[reco_prefix] = ak.zip(fields, depth_limit=1)
    return output


def kth_sort_order(arr, sorton):
    """
    Per-event descending argsort of ``arr[sorton]`` (NaN / missing sort as 0).
    Compute this once per (array, sort field) and reuse it for several k / fields.
    """
    sort_key = ak.nan_to_num(ak.fill_none(arr[sorton], 0.0), nan=0.0)
    return ak.argsort(sort_key, axis=1, ascending=False)


def kth_from_order(arr, field, order, k):
    """
    k-th (1-indexed) value of ``arr[field]`` following a precomputed per-event ``order``.
    Events with fewer than k entries (or NaN values) give 0. Returns a float64 numpy array.
    """
    # only gather the single k-th index per event instead of reordering the whole field
    kth = ak.firsts(arr[field][order[:, k-1:k]], axis=1)
    return np.nan_to_num(np.asarray(ak.to_numpy(ak.fill_none(kth, 0.0)), dtype=np.float64), nan=0.)


def select_kth(arr, field, sorton, k):
    """
    k-th (1-indexed) value of ``field`` when the event is sorted by descending ``sorton``.
    """
    return kth_from_order(arr, field, kth_sort_order(arr, sorton), k)


def select_kths(arr, fields, sorton, k):
    """
    Same as :func:`select_kth` for several fields with a single sort.

    Returns
    -------
    list : numpy arrays, one per field
    """
    order = kth_sort_order(arr, sorton)
    return [kth_from_order(arr, field, order, k) for field in fields]


def _resolve_selector(selector, pairs, nobj):
    """Accept a selector callable ``f(pairs, nobj)`` or a precomputed boolean event mask."""
    if selector is None:
        return np.ones(len(pairs), dtype=bool)
    if callable(selector):
        return np.asarray(selector(pairs, nobj), dtype=bool)
    return np.asarray(selector, dtype=bool)


def _event_weights(pairs, event_weights=None):
    if event_weights is None:
        return np.asarray(ak.to_numpy(pairs["weight"]), dtype=np.float64)
    return np.asarray(event_weights, dtype=np.float64)


def weighted_percentile(data, q, weights):
    """
    Compute weighted percentiles for 1-D data.

    Parameters
    ----------
    data : 1-D array-like
        Values.
    q : float or sequence of floats in [0, 100]
        Percentile or sequence of percentiles to compute.
    weights : 1-D array-like or None
        Non-negative weights same length as data. If None, unweighted percentiles are computed.

    Returns
    -------
    percentiles : ndarray
        If q is scalar, returns a scalar ndarray (0-d) containing the percentile.
        If q is sequence, returns array of same length as q.
    """
    data = np.asarray(data)
    if data.size == 0:
        return np.array([])

    q_arr = np.atleast_1d(q).astype(float)
    if np.any((q_arr < 0) | (q_arr > 100)):
        raise ValueError("q must be in [0, 100]")

    # If no weights, use numpy's percentile on the flattened array
    if weights is None:
        return np.percentile(data, q_arr)

    w = np.asarray(weights, dtype=float)
    if w.shape != data.shape:
        raise ValueError("weights must have the same shape as data")

    # Mask out NaNs in data or non-finite weights
    mask = np.isfinite(data) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return np.full(q_arr.shape, np.nan)

    data = data[mask]
    w = w[mask]

    # sort by data
    order = np.argsort(data)
    x = data[order]
    w = w[order]

    # normalized cumulative weights in [0,1]
    cumw = np.cumsum(w)
    total = cumw[-1]
    if total <= 0:
        return np.full(q_arr.shape, np.nan)
    cumw = cumw / total

    # percentiles expressed as fractions between 0 and 1
    probs = q_arr / 100.0

    # np.interp requires xp to be increasing and returns linear interp;
    # extend xp with 0 at left using the first x for correct behavior at 0.
    xp = np.concatenate(([0.0], cumw))
    fp = np.concatenate(([x[0]], x))

    vals = np.interp(probs, xp, fp)

    # return scalar if q was scalar
    if np.isscalar(q):
        return np.array(vals.item())
    return vals

def compute_pt_threshold(bkg_pairs, target_eff, nobj, correctors=None, selector=None,
                         reco_pt_kth=None, event_weights=None):
    """
    Compute a reco-pt threshold that yields the requested background efficiency.

    The function finds threshold T such that fraction(background with reco_pt > T)
    is approximately `target_eff`. Returns (threshold, actual_eff).

    Parameters
    ----------
    bkg_pairs : awkward array of matched pairs (see match_reco_truth).
    target_eff : float
        Target background efficiency in (0,1). Example: 0.01 for 1%.
    selector : callable or bool array
        Applied to the background before the threshold scan (rate selector).
    reco_pt_kth : np.ndarray, optional
        Precomputed nobj-th leading reco pt per event (from select_kth) to avoid re-sorting.
    event_weights : np.ndarray, optional
        Precomputed per-event weights.

    Returns
    -------
    threshold : float
        Reco-pt threshold. If bkg_pairs is empty, returns np.inf.
    actual_eff : float
        Actual fraction of background events with reco_pt > threshold.
    """
    if not (0.0 < target_eff < 1.0):
        raise ValueError("target_eff must be between 0 and 1 (exclusive).")

    if bkg_pairs is None or len(bkg_pairs) == 0:
        return np.inf, 0.0

    if correctors is not None:
        for corrector in correctors:
            if callable(corrector):
                corrector(bkg_pairs)

    reco_pt = select_kth(bkg_pairs, "reco_pt", "reco_pt", nobj) if reco_pt_kth is None else np.asarray(reco_pt_kth)
    w = _event_weights(bkg_pairs, event_weights)
    sel = _resolve_selector(selector, bkg_pairs, nobj)
    reco_pt_selected = reco_pt[sel]
    w_selected = w[sel]

    if reco_pt_selected.size == 0 or np.sum(w_selected) <= 0:
        return np.inf, 0.0

    # We want threshold T so that fraction with (reco_pt > T and selector) == target_eff.
    # That means T is the (1-target_eff) quantile of the reco_pt distribution.
    q = 100.0 * (1.0 - target_eff * np.sum(w) / np.sum(w_selected))
    if q < 0.0:
        print(f"Warning: target efficiency {target_eff:.3g} not reachable with selector (max {np.sum(w_selected)/np.sum(w):.3g}); using lowest threshold")
        q = 0.0

    threshold = weighted_percentile(reco_pt_selected, q, w_selected)

    # Compute actual achieved efficiency (strictly greater than threshold)
    # If you prefer >=, change '>' to '>='.
    actual_eff = np.sum(w[(reco_pt > threshold) & sel]) / np.sum(w)

    return float(threshold), float(actual_eff)

def compute_rate(bkg_pairs, threshold, nobj, correctors=None, full_rate=31_000., selector=None,
                 reco_pt_kth=None, event_weights=None):
    """
    Compute a rate from the given threshold and nobj.

    Returns (rate in kHz, actual_eff).

    Parameters
    ----------
    bkg_pairs : awkward array of matched pairs (see match_reco_truth).
    threshold : float
        Reco-pt threshold (>= passes).
    reco_pt_kth, event_weights : optional precomputed per-event arrays (see compute_pt_threshold).

    Returns
    -------
    rate : float
        Rate in kHz. If bkg_pairs is empty, returns np.inf.
    actual_eff : float
        Actual fraction of background events with reco_pt >= threshold.
    """
    if threshold < 0.:
        raise ValueError("threshold must be non-negative.")

    if bkg_pairs is None or len(bkg_pairs) == 0:
        return np.inf, 0.0

    if correctors is not None:
        for corrector in correctors:
            if callable(corrector):
                corrector(bkg_pairs)

    reco_pt = select_kth(bkg_pairs, "reco_pt", "reco_pt", nobj) if reco_pt_kth is None else np.asarray(reco_pt_kth)
    w = _event_weights(bkg_pairs, event_weights)
    sel = _resolve_selector(selector, bkg_pairs, nobj)

    # Compute actual achieved efficiency
    actual_eff = np.sum(w[(reco_pt >= threshold) & sel]) / np.sum(w)

    return float(full_rate*actual_eff), float(actual_eff)

from scipy.stats import beta

def _safe_ratio(num, den):
    out = np.zeros(np.broadcast(num, den).shape, dtype=float)
    np.divide(num, den, out=out, where=den > 0.)
    return out

def teff(num, den, sumw2_num=None, sumw2_den=None, alpha=0.682689492137086):
    """
    Efficiency and (central) Clopper-Pearson-style interval.

    Unweighted (default):
      num, den are integer counts, uses standard Beta intervals.

    Weighted:
      pass sumw2_num and sumw2_den (sum of squared weights in numerator/denominator).
      Form an effective trial count Neff = (sumw_den)^2 / sumw2_den and an
      effective success count keff = eff * Neff, then use Beta intervals on
      (keff, Neff). This is an approximation but behaves well for 0<=eff<=1.
    """
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)

    eff = _safe_ratio(num, den)
    eff = np.clip(eff, 0.0, 1.0)

    # Unweighted case: integer-count Clopper-Pearson
    if sumw2_den is None or sumw2_num is None:
        low = np.where(
            num <= 0,
            0.0,
            beta.ppf((1 - alpha) / 2, num, den - num + 1),
        )
        high = np.where(
            num >= den,
            1.0,
            beta.ppf(1 - (1 - alpha) / 2, num + 1, den - num),
        )
        low = np.clip(low, 0.0, 1.0)
        high = np.clip(high, 0.0, 1.0)
        return eff, eff - low, high - eff

    else:

        # Weighted case: use effective counts
        sumw2_num = np.asarray(sumw2_num, dtype=float)
        sumw2_den = np.asarray(sumw2_den, dtype=float)

        # Neff = (sumw)^2 / sumw2 (guard against zeros)
        neff = _safe_ratio(den * den, sumw2_den)
        neff = np.maximum(neff, 0.0)

        keff = eff * neff
        keff = np.clip(keff, 0.0, neff)

        # Beta interval on (keff, neff-keff)
        low = np.where(
            (keff <= 0.) | (neff <= 0.),
            0.0,
            beta.ppf((1 - alpha) / 2, keff, neff - keff + 1),
        )
        high = np.where(
            (keff >= neff) & (neff > 0.),
            1.0,
            np.where(
                neff <= 0.,
                1.0,
                beta.ppf(1 - (1 - alpha) / 2, keff + 1, neff - keff),
            ),
        )

        low = np.clip(low, 0.0, 1.0)
        high = np.clip(high, 0.0, 1.0)
        return eff, eff - low, high - eff

def _hist_with_flow(values, bins, weights_arr=None):
    """Histogram with under/overflow absorbed in first/last bins."""
    nbins = len(bins) - 1
    indices = np.searchsorted(bins, values, side="right") - 1
    indices = np.clip(indices, 0, nbins - 1)
    return np.bincount(indices, weights=weights_arr, minlength=nbins)

def compute_signal_efficiency(
    sig_pairs,
    threshold,
    turnon_bins,
    nobj,
    selector,
    numerator_selector=None,
    turnon_values=None,
    weights=False,
    inclusive=True,
    correctors=None,
    reco_pt_kth=None,
    event_weights=None,
):
    """
    Compute signal efficiency vs a turn-on variable for a given reco-pt threshold.

    Parameters
    ----------
    sig_pairs : awkward array of matched pairs (see match_reco_truth).
    threshold : float
        Reco-pt threshold. An event "passes" if reco_pt >= threshold (inclusive=True)
        or reco_pt > threshold (inclusive=False).
    turnon_bins : array_like
        Bin edges for the turn-on variable (e.g. np.linspace(0,2000,41)).
    selector : callable or bool array
        Selection applied to both numerator and denominator.
    numerator_selector : callable or bool array
        Additional selection applied only to the numerator together with the pt threshold.
    turnon_values : np.ndarray, optional
        Per-event turn-on variable (default: nobj-th leading truth pt). Events with a
        non-finite value (e.g. no dijet pair) are excluded from numerator and denominator.
    weights : bool
        Use weights.
    inclusive : bool
        If True use reco_pt >= threshold (default), else > threshold.
    reco_pt_kth, event_weights : optional precomputed per-event arrays.

    Returns
    -------
    centers : np.ndarray
        Bin centers for turn-on bins.
    efficiency : np.ndarray
        Efficiency in each bin (length = len(turnon_bins)-1).
    total_counts : np.ndarray
        Total counts (or sum of weights) in each bin.
    passed_counts : np.ndarray
        Counts (or sum of weights) in each bin that pass the threshold.
    err : np.ndarray
        (2, nbins) lower/upper 1-sigma uncertainty on the efficiency in each bin.
    """
    if sig_pairs is None or len(sig_pairs) == 0:
        nbins = len(turnon_bins) - 1
        return (
            0.5 * (turnon_bins[:-1] + turnon_bins[1:]),
            np.zeros(nbins, dtype=float),
            np.zeros(nbins, dtype=float),
            np.zeros(nbins, dtype=float),
            np.zeros((2,nbins), dtype=float),
        )

    if correctors is not None:
        for corrector in correctors:
            if callable(corrector):
                corrector(sig_pairs)

    reco_pt = select_kth(sig_pairs, "reco_pt", "reco_pt", nobj) if reco_pt_kth is None else np.asarray(reco_pt_kth)
    if turnon_values is None:
        turnon_values = select_kth(sig_pairs, "truth_pt", "truth_pt", nobj)
    turnon_values = np.asarray(ak.to_numpy(turnon_values) if isinstance(turnon_values, ak.Array) else turnon_values, dtype=float)

    denominator_sel = _resolve_selector(selector, sig_pairs, nobj) & np.isfinite(turnon_values)
    numerator_sel = _resolve_selector(numerator_selector, sig_pairs, nobj)

    reco_pt = reco_pt[denominator_sel]
    turnon_values = turnon_values[denominator_sel]
    numerator_sel = numerator_sel[denominator_sel]

    if inclusive:
        passed_mask = (reco_pt >= threshold) & numerator_sel
    else:
        passed_mask = (reco_pt > threshold) & numerator_sel

    if not weights:
        # Unweighted histograms (with first/last bins as under/overflow)
        total_counts = _hist_with_flow(turnon_values, turnon_bins)
        passed_counts = _hist_with_flow(turnon_values[passed_mask], turnon_bins)

        efficiency, errlo, errhi = teff(passed_counts, total_counts)

    else:
        # Weighted
        w = _event_weights(sig_pairs, event_weights)[denominator_sel]
        w_pass = w[passed_mask]

        # sum of weights
        total_w = _hist_with_flow(turnon_values, turnon_bins, weights_arr=w)
        passed_w = _hist_with_flow(turnon_values[passed_mask], turnon_bins, weights_arr=w_pass)

        # sum of squared weights (for effective counts)
        total_w2 = _hist_with_flow(turnon_values, turnon_bins, weights_arr=w * w)
        passed_w2 = _hist_with_flow(turnon_values[passed_mask], turnon_bins, weights_arr=w_pass * w_pass)

        efficiency, errlo, errhi = teff(
            passed_w,
            total_w,
            sumw2_num=passed_w2,
            sumw2_den=total_w2,
        )

        # convert to float arrays
        total_counts = total_w.astype(float)
        passed_counts = passed_w.astype(float)

    edges = np.asarray(turnon_bins, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, efficiency, total_counts, passed_counts, np.stack([errlo, errhi])

def compute_full_efficiency(
    pairs,
    pt_bins,
    nobj,
    selector,
    numerator_selector=None,
    weights=False,
    correctors=None,
    reco_pt_kth=None,
    event_weights=None,
):
    """
    Compute efficiency vs reco-pt threshold (fraction of selected events whose
    nobj-th leading reco pt is >= each threshold in ``pt_bins``).

    Parameters
    ----------
    pairs : awkward array of matched pairs (see match_reco_truth).
    pt_bins : array_like
        Thresholds to scan.
    selector : callable or bool array
        Selection applied to numerator and denominator.
    numerator_selector : callable or bool array
        Additional selection applied only to the numerator.
    weights : bool
        Use weights.
    reco_pt_kth, event_weights : optional precomputed per-event arrays.

    Returns
    -------
    efficiency : np.ndarray
        Efficiency for each pt threshold (length = len(pt_bins)).
    err : np.ndarray
        (2, len(pt_bins)) lower/upper 1-sigma uncertainty on the efficiency.
    """

    if pairs is None or len(pairs) == 0:
        nbins = len(pt_bins)
        return (
            np.zeros(nbins, dtype=float),
            np.zeros((2,nbins), dtype=float)
        )

    if correctors is not None:
        for corrector in correctors:
            if callable(corrector):
                corrector(pairs)

    reco_pt = select_kth(pairs, "reco_pt", "reco_pt", nobj) if reco_pt_kth is None else np.asarray(reco_pt_kth)

    denominator_sel = _resolve_selector(selector, pairs, nobj)
    numerator_sel = _resolve_selector(numerator_selector, pairs, nobj)[denominator_sel]
    reco_pt = reco_pt[denominator_sel]

    thresholds = np.asarray(pt_bins, dtype=float)

    # Sort the numerator candidates once; "count of reco_pt >= thr" is then a searchsorted lookup.
    cand_pt = reco_pt[numerator_sel]
    order = np.argsort(cand_pt, kind="stable")
    cand_sorted = cand_pt[order]
    first_ge = np.searchsorted(cand_sorted, thresholds, side="left")

    if not weights:
        total_counts = np.full(len(thresholds), reco_pt.shape[0], dtype=float)
        passed_counts = (cand_sorted.shape[0] - first_ge).astype(float)

        efficiency, errlo, errhi = teff(passed_counts, total_counts)

    else:
        # Weighted: suffix sums of (sorted) weights give sum of weights above each threshold
        w = _event_weights(pairs, event_weights)[denominator_sel]
        if w.shape[0] != reco_pt.shape[0]:
            raise ValueError("weights must have same length as pairs")

        w2 = w * w
        w_sorted = w[numerator_sel][order]
        w2_sorted = w2[numerator_sel][order]
        suffix_w = np.concatenate([np.cumsum(w_sorted[::-1])[::-1], [0.0]])
        suffix_w2 = np.concatenate([np.cumsum(w2_sorted[::-1])[::-1], [0.0]])

        total_w = np.full(len(thresholds), np.sum(w), dtype=float)
        total_w2 = np.full(len(thresholds), np.sum(w2), dtype=float)
        passed_w = suffix_w[first_ge]
        passed_w2 = suffix_w2[first_ge]

        efficiency, errlo, errhi = teff(
            passed_w,
            total_w,
            sumw2_num=passed_w2,
            sumw2_den=total_w2,
        )

    return efficiency, np.stack([errlo, errhi])

import scipy.optimize as opt

# Define the gaussian function
def gaussian_function(x, amplitude, mean, stddev, baseline):
    """
    A standard Gaussian function with an optional baseline offset.
    amplitude: peak height of the Gaussian
    mean: center of the Gaussian peak (x0)
    stddev: standard deviation (controls the width)
    baseline: constant vertical offset
    """
    return baseline + amplitude * np.exp(-((x - mean) / (2 * stddev))**2)

def fit_gaussian(data, truncate=0.05, trim=5):
    tlo, thi = np.quantile(data, [truncate, 1.-truncate])
    y_data, bins = np.histogram(data[(data>tlo) & (data<thi)], bins=50)
    y_data = y_data/np.sum(y_data)
    if trim>0:
        y_data = y_data[trim:-trim]
    x_data = 0.5*(bins[trim:-(1+trim)]+bins[trim+1:-trim])
    # Fit the curve to the data
    try:
        params, _ = opt.curve_fit(gaussian_function, x_data, y_data, p0=[1., 0.5, 0.2, 0.]) # p0 provides initial guesses for parameters
    except:
        params = [1.,1.,1.,0.]
    # Extract the fitted parameters
    return params[1], np.abs(params[2])

def energy(pt,eta):
    return pt*np.cosh(eta)

def compute_response(pairs, pt_bins, eta_bins, min_pt=None, respcorrs=None, debug=None, dofit=False):
    if respcorrs is not None:
        for respcorr in respcorrs:
            if callable(respcorr):
                respcorr(pairs)

    if min_pt is None:
        min_pt = pt_bins[0]

    # Apply cuts in Awkward before materializing NumPy (reduces N if lots are cut)
    reco_pt_ak   = ak.flatten(pairs["reco_pt"])
    reco_eta_ak  = ak.flatten(pairs["reco_eta"])
    truth_pt_ak  = ak.flatten(pairs["truth_pt"])
    truth_eta_ak = ak.flatten(pairs["truth_eta"])

    mask = (truth_pt_ak > min_pt) & (reco_pt_ak > 0.)

    reco_pt   = ak.to_numpy(reco_pt_ak[mask]).astype(np.float32, copy=False)
    reco_eta  = ak.to_numpy(reco_eta_ak[mask]).astype(np.float32, copy=False)
    truth_pt  = ak.to_numpy(truth_pt_ak[mask]).astype(np.float32, copy=False)
    truth_eta = ak.to_numpy(truth_eta_ak[mask]).astype(np.float32, copy=False)

    # response in a single buffer
    response = energy(reco_pt, reco_eta).astype(np.float32, copy=False)
    response /= energy(truth_pt, truth_eta).astype(np.float32, copy=False)

    n_pt = len(pt_bins) - 1
    n_eta = len(eta_bins) - 1
    nb = n_pt * n_eta

    pt_idx  = np.clip(np.digitize(truth_pt, pt_bins) - 1, 0, n_pt - 1).astype(np.int32, copy=False)
    eta_idx = np.clip(np.digitize(truth_eta, eta_bins) - 1, 0, n_eta - 1).astype(np.int32, copy=False)
    bin_id = (eta_idx * n_pt + pt_idx).astype(np.int32, copy=False)

    order = np.argsort(bin_id, kind="stable")
    bin_id_s = bin_id[order]
    resp_s = response[order]

    # boundaries
    edges = np.searchsorted(bin_id_s, np.arange(nb + 1), side="left")
    counts = np.diff(edges).astype(np.int32)

    response_centers = np.full(nb, np.nan, dtype=np.float32)
    response_uncs    = np.full(nb, np.nan, dtype=np.float32)

    for i in range(nb):
        a, b = edges[i], edges[i+1]
        if a == b:
            continue
        r = np.array(resp_s[a:b], copy=True) # needed to avoid numpy read-only error
        if dofit:
            response_centers[i], response_uncs[i] = fit_gaussian(r)
        else:
            response_centers[i] = np.median(r)
            p16, p84 = np.percentile(r, [16, 84])
            response_uncs[i] = 0.5 * (p84 - p16)
        
    # --- Debug Plotting ---
    if debug is not None:
        for ie in range(n_eta):
            etamask = (ie == eta_idx)
            if np.sum(etamask) == 0: continue

            bin_slice = slice(ie * n_pt, (ie + 1) * n_pt)
            x_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
            y_centers = response_centers[bin_slice]
            
            # 2D Hist: Truth pT vs Response
            plt.hist2d(truth_pt[etamask], response[etamask], 
                       bins=[pt_bins, np.linspace(0., np.maximum(np.max(y_centers[np.isfinite(y_centers)]) if np.sum(np.isfinite(y_centers))>0 else 3.,3.), 100)], cmap="viridis", rasterized=True)
            
            # Overlay the response
            plt.errorbar(x_centers, response_centers[bin_slice], 
                         yerr=response_uncs[bin_slice], 
                         fmt='o-', color='r', alpha=marker_alpha(0.25), markersize=DEFAULT_MARKER_SIZE, label='Fit' if dofit else 'Median')
            
            plt.axhline(1.0, color='k', linestyle='--')
            plt.ylabel(r"Response (Reco / Truth)")
            plt.xlabel(r"Truth $p_T$ [GeV]")
            plt.title(f"Eta: {eta_bins[ie]:.1f} - {eta_bins[ie+1]:.1f}")
            place_legend_above()
            
            # Ensure plotdir exists or handle path
            plot_name = build_debug_plot_path(debug, 'debug_%s_eta_%.1f_%.1f.pdf'%(debug,eta_bins[ie],eta_bins[ie+1]))
            plt.savefig(plot_name, bbox_inches='tight')
            plt.close()
            
            # ------
            # 2D Hist: Reco pT vs Response
            plt.hist2d(reco_pt[etamask], response[etamask], 
                       bins=[pt_bins, np.linspace(0., np.maximum(np.max(y_centers[np.isfinite(y_centers)]) if np.sum(np.isfinite(y_centers))>0 else 3.,3.), 100)], cmap="viridis", rasterized=True)
            
            # Overlay the response
            bin_slice = slice(ie * n_pt, (ie + 1) * n_pt)
            x_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:]) * response_centers[bin_slice]
            #plt.errorbar(x_centers, response_centers[bin_slice], 
            #             yerr=response_uncs[bin_slice], 
            #             fmt='o-', color='r', alpha=marker_alpha(0.25), markersize=DEFAULT_MARKER_SIZE, label='Fit' if dofit else 'Median')
            
            plt.axhline(1.0, color='k', linestyle='--')
            plt.ylabel(r"Response (Reco / Truth)")
            plt.xlabel(r"Reco $p_T$ [GeV]")
            plt.title(f"Eta: {eta_bins[ie]:.1f} - {eta_bins[ie+1]:.1f}")
            place_legend_above()
            
            # Ensure plotdir exists or handle path
            plot_name = build_debug_plot_path(debug, 'debug_%s_reco_eta_%.1f_%.1f.pdf'%(debug,eta_bins[ie],eta_bins[ie+1]))
            plt.savefig(plot_name, bbox_inches='tight')
            plt.close()
            
        if response.size > 0:
            plt.hist(response-1., bins=np.linspace(-1., np.max(response)-1., 91), color='r', histtype='step')
        plt.axvline(0., color='k', linestyle='--')
        plt.xlabel(r"Reco - Truth / Truth")
        plt.ylabel(r"Number of objects")

        # Ensure plotdir exists or handle path
        plot_name = build_debug_plot_path(debug, 'debug_%s_all.pdf'%(debug))
        plt.savefig(plot_name, bbox_inches='tight')
        plt.close()
            
        
    return response_centers, response_uncs, counts

class AreaSubtractor:
    """
    A callable class that computes and applies a subtraction based on rho
    Memory-optimized version with explicit cleanup
    """
    def __init__(self, pairs, eta_bins, min_pt=20., debug=None):
        self.eta_bins = eta_bins
        self.slopes = []
        self.intercepts = []
        self.debug = debug is not None
        self.debug_name = debug if self.debug else ""
        
        # --- Data Preparation ---
        reco_pt = ak.to_numpy(ak.flatten(pairs["reco_pt"]))
        reco_eta = ak.to_numpy(ak.flatten(pairs["reco_eta"]))
        truth_pt = ak.to_numpy(ak.flatten(pairs["truth_pt"]))
    
        #Element 0 for -2.5 < eta < 0.0
        #Element 1 for 0.0 < eta < 2.5
        #Element 2 for |eta| > 2.5
        rho_per_event = ak.to_numpy(pairs["rho"])
        
        # build index based on value of eta
        eta_idx = ak.where(
            (pairs["reco_eta"] > -2.5) & (pairs["reco_eta"] <= 0), 0,
            ak.where(
                (pairs["reco_eta"] > 0) & (pairs["reco_eta"] < 2.5), 1,
                2
            )
        )

        # select from numpy array row-wise using the computed index
        # broadcast row indices to match awkward structure
        row_idx = ak.broadcast_arrays(
            ak.local_index(pairs["reco_eta"], axis=0), eta_idx
        )[0]
        
        rho_per_obj = rho_per_event[ak.to_numpy(ak.flatten(ak.fill_none(row_idx,0))), ak.to_numpy(ak.flatten(ak.fill_none(eta_idx,0)))]
        del eta_idx, row_idx

        # Broadcast rho to match the flattened obj arrays
        #rho_broadcast = ak.broadcast_arrays(rho_per_event, pairs["reco_pt"])[0]
        #rho_per_obj = ak.to_numpy(ak.flatten(rho_broadcast))
        #del rho_broadcast  # Explicit cleanup
        
        if min_pt is None:
            min_pt = 0.0
        
        # Apply basic kinematic cuts
        ptmask = ((truth_pt > min_pt) & (reco_pt > 0.))
        reco_pt = reco_pt[ptmask]
        reco_eta = reco_eta[ptmask]
        truth_pt = truth_pt[ptmask]
        rho_per_obj = rho_per_obj[ptmask]
        del ptmask  # Explicit cleanup
        
        n_eta = len(eta_bins) - 1
        
        # Create single figure for all debug plots if needed
        if self.debug:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Fit slope of (reco_pt - truth_pt) vs rho in each eta bin
        for ie in range(n_eta):
            eta_min = eta_bins[ie]
            eta_max = eta_bins[ie + 1]
            
            # Select objects in this eta bin
            mask = (reco_eta >= eta_min) & (reco_eta < eta_max)
            
            if np.sum(mask) < 2:
                if self.debug:
                    print(f"Warning: Not enough objects in eta bin [{eta_min:.2f}, {eta_max:.2f}]")
                self.slopes.append(0.0)
                self.intercepts.append(0.0)
                continue
            
            # Calculate pt difference
            pt_diff = reco_pt[mask] - truth_pt[mask]
            diff_lo, diff_hi = np.percentile(pt_diff,[16,84])
            diffmask = (pt_diff >= diff_lo) & (pt_diff < diff_hi)
            rho_bin = rho_per_obj[mask]
            
            pt_diff = pt_diff[diffmask]
            rho_bin = rho_bin[diffmask]
            
            # Fit linear relationship: pt_diff = slope * rho + intercept
            coeffs = np.polyfit(rho_bin, pt_diff, deg=1)
            slope = coeffs[0]
            intercept = coeffs[1]
            
            # Check if slope is sensible
            if slope < 0:
                if self.debug:
                    print(f"Warning: Negative slope {slope:.4f} in eta bin [{eta_min:.2f}, {eta_max:.2f}]. Setting to 0.")
                slope = 0.0
            
            self.slopes.append(slope)
            self.intercepts.append(intercept)
            
            if self.debug:
                ax.clear()  # Clear axes, not figure
                ax.hist2d(rho_bin, pt_diff, bins=50, cmap="viridis", rasterized=True)
                ax.set_xlabel(r"$\rho$")
                ax.set_ylabel(r"$\Delta p_{T}$ (Reco-Truth)")
                rho_min = np.min(rho_bin)
                rho_max = np.max(rho_bin)
                ax.plot([rho_min, rho_max], 
                       [rho_min*slope+intercept, rho_max*slope+intercept],
                       color='r', linestyle='dashed')
                plt.savefig(build_debug_plot_path(debug, 'debug_%srhofit_eta_%.2f_%.2f.pdf'%(debug,eta_min,eta_max)), 
                           bbox_inches='tight')
            
            # Clean up per-iteration arrays
            del pt_diff, rho_bin, mask
        
        # Clean up figure and large arrays
        if self.debug:
            plt.close(fig)
        del reco_pt, reco_eta, truth_pt, rho_per_obj, rho_per_event
    
    def __call__(self, pairs, use_fit = False):
        """
        Apply rho-based subtraction to correct object pt in pairs structure IN-PLACE
        
        Parameters:
        -----------
        pairs : dict
            Dictionary containing "reco_pt", "reco_eta", and "rho" fields
            Modified in-place
        """
        # Extract data from pairs
        reco_pt = pairs["reco_pt"]
        reco_eta = pairs["reco_eta"]
        rho_per_event = ak.to_numpy(pairs["rho"])
        
        # Flatten for processing
        pt_flat = ak.flatten(reco_pt)
        eta_flat = ak.flatten(reco_eta)
        
        # Broadcast rho to per-obj
        
        # build index based on value of eta
        eta_idx = ak.where(
            (reco_eta > -2.5) & (reco_eta <= 0), 0,
            ak.where(
                (reco_eta > 0) & (reco_eta < 2.5), 1,
                2
            )
        )

        # select from numpy array row-wise using the computed index
        # broadcast row indices to match awkward structure
        row_idx = ak.broadcast_arrays(
            ak.local_index(reco_eta, axis=0), eta_idx
        )[0]

        rho_per_obj = rho_per_event[ak.to_numpy(ak.flatten(ak.fill_none(row_idx,0))), ak.to_numpy(ak.flatten(ak.fill_none(eta_idx,0)))]
        del eta_idx, row_idx

        #rho_broadcast = ak.broadcast_arrays(rho_per_event, reco_pt)[0]
        #rho_per_obj = ak.flatten(rho_broadcast)
        #del rho_broadcast  # Explicit cleanup
        
        # Convert to numpy for correction (computed in float64, stored back as float32)
        pt_np = np.asarray(ak.to_numpy(pt_flat), dtype=np.float64)
        eta_np = np.asarray(ak.to_numpy(eta_flat), dtype=np.float64)
        rho_np = np.asarray(ak.to_numpy(rho_per_obj), dtype=np.float64)
        
        # Clean up awkward intermediates
        del pt_flat, eta_flat, rho_per_obj
        
        corrected_pt_flat = pt_np.copy()
        
        # Apply correction in each eta bin
        for ie in range(len(self.eta_bins) - 1):
            eta_min = self.eta_bins[ie]
            eta_max = self.eta_bins[ie + 1]
            
            mask = (eta_np >= eta_min) & (eta_np < eta_max) & (pt_np > 0.)
            
            if np.any(mask) and use_fit:
                # Subtract: corrected_pt = reco_pt - intercept - slope * rho
                correction = self.intercepts[ie] + (self.slopes[ie] * rho_np[mask])
                corrected_pt_flat[mask] -= correction  # In-place operation
                del correction
                
            else: # assuming R=0.4 circles for all jet areas (and converting to GeV)
                correction = rho_np[mask]*(0.16*3.14159)/1000.
                corrected_pt_flat[mask] -= correction  # In-place operation
                del correction
        
        # Clean up numpy intermediates before unflatten
        del pt_np, eta_np, rho_np, rho_per_event
        
        # Unflatten back to original structure
        corrected_pt = ak.unflatten(corrected_pt_flat.astype(np.float32), ak.num(reco_pt))
        del corrected_pt_flat  # Clean up before reassignment
        
        # Modify pairs in-place
        pairs["reco_pt"] = corrected_pt


from scipy.interpolate import make_smoothing_spline

def identity_response(x): 
    return np.ones_like(x)
        
class ResponseInterpolator:
    """
    A callable class that interpolates the Response (Reco/Truth)
    Memory-optimized version with explicit cleanup
    """
    def __init__(self, response_centers, response_errors, pt_bins, eta_bins, debug=None, spline_lambda=1e-5):
        self.eta_bins = eta_bins
        self.interpolators = []
        self.interp_ranges = []
        self.endpoint_values = []
        self.debug = debug
        
        # Reshape the 1D response array into (n_eta_bins, n_pt_bins)
        n_pt = len(pt_bins) - 1
        n_eta = len(eta_bins) - 1
        resp_2d = response_centers.reshape(n_eta, n_pt)
        
        # Calculate Truth pT centers
        truth_centers = 0.5 * (pt_bins[1:] + pt_bins[:-1])
        reco_centers = truth_centers[np.newaxis, :] * resp_2d
        
        plt.clf()
        for i_eta in range(n_eta):
            R = resp_2d[i_eta]
            
            # Filter out invalid points
            valid = (R > 0) & (~np.isnan(R))
            
            if np.sum(valid) < 5:
                # Fallback: return identity (scipy's make_smoothing_spline requires >=5 points)
                self.interpolators.append(None)
                self.interp_ranges.append(None)
                self.endpoint_values.append(None)
            else:
                # Create smoothing spline for R(log(Reco_pT))
                x = np.log(reco_centers[i_eta][valid])
                y = R[valid]
                order = np.argsort(x)
                x = x[order]
                y = y[order]

                lam = spline_lambda * len(x)

                spline = make_smoothing_spline(x, y, lam=lam)

                # Clamp to endpoint values outside fitted domain.
                x0, x1 = x[0], x[-1]
                y_lo = y[0]
                y_hi = y[-1]

                self.interpolators.append(spline)
                self.interp_ranges.append((x0, x1))
                self.endpoint_values.append((y_lo, y_hi))
                x_eval = np.clip(x, x0, x1)
                y_eval = spline(x_eval)
                y_plot = np.where(x < x0, y_lo, np.where(x > x1, y_hi, y_eval))

                plt.plot(x,
                         y_plot,
                         color='C%i'%i_eta, 
                         label=r"$%.2f<\eta<%.2f$"%(eta_bins[i_eta],eta_bins[i_eta+1])
                        )
                plt.scatter(x, y, color='C%i'%i_eta, alpha=marker_alpha(0.5), s=DEFAULT_SCATTER_SIZE)
                del x, y, order, spline, y_lo, y_hi, lam, x0, x1, x_eval, y_eval, y_plot
            
            # Clean up per-iteration arrays
            del R, valid
        plt.xlabel(r'$log(p_{T})$')
        plt.ylabel('R (reco/truth)')
        place_legend_above()
        plot_name = build_debug_plot_path(debug, 'debug_%s.pdf'%(debug))
        plt.savefig(plot_name, bbox_inches='tight')
        
        # Clean up after initialization
        del resp_2d, reco_centers, truth_centers

    def __call__(self, pairs):
        """
        Apply the interpolation IN-PLACE to correct reco_pt
        """
        # Extract data from pairs
        reco_pt = pairs["reco_pt"]
        reco_eta = pairs["reco_eta"]
        
        # evaluate the correction in float64 (storage is float32)
        pt = np.asarray(ak.to_numpy(ak.flatten(reco_pt)), dtype=np.float64)
        eta = np.asarray(ak.to_numpy(ak.flatten(reco_eta)), dtype=np.float64)
        
        out_resp_flat = np.ones_like(pt)
        
        # Identify which eta bin each object belongs to
        eta_indices = np.digitize(eta, self.eta_bins) - 1
        eta_indices = np.clip(eta_indices, 0, len(self.interpolators) - 1)
        
        for ie in range(len(self.eta_bins)-1):
            mask = (eta_indices == ie)
            if not np.any(mask):
                continue

            spline = self.interpolators[ie]
            if spline is None:
                out_resp_flat[mask] = 1.0
                continue

            log_pt = np.log(pt[mask])
            x0, x1 = self.interp_ranges[ie]
            y0, y1 = self.endpoint_values[ie]
            log_pt_eval = np.clip(log_pt, x0, x1)
            y_eval = spline(log_pt_eval)
            out_resp_flat[mask] = np.where(log_pt < x0, y0, np.where(log_pt > x1, y1, y_eval))
        
        # Corrected pt, stored back as float32 to keep the pairs memory footprint
        corrected = (pt / out_resp_flat).astype(np.float32)
        del out_resp_flat, pt, eta, eta_indices
        
        # Modify pairs in-place
        pairs["reco_pt"] = ak.unflatten(corrected, ak.num(reco_pt))

import pickle

def save_corrs(corr_obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(corr_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_corrs(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def _debug_kinematic_plots(config, sig_pairs, bkg_pairs, nobj, prefix, tag):
    """Debug histograms of the nobj-th leading reco/truth pt/eta for every collection."""
    for var in ["reco_pt","reco_eta","truth_pt","truth_eta"]:
        figb, axb = plt.subplots()
        figs, axs = plt.subplots()
        fig, ax = plt.subplots()
        for r,reco_prefix in enumerate(config.reco_prefixes):
            if r>0 and "truth" in var:
                continue
            if 'pt' in var:
                bins = config.truth_pt_bins
            else:
                bins = config.truth_eta_bins
            bkgv = select_kth(bkg_pairs[reco_prefix], var, "reco_pt" if "reco" in var else "truth_pt", nobj)
            bkgw = bkg_pairs[reco_prefix]["weight"]
            sigv = select_kth(sig_pairs[reco_prefix], var, "reco_pt" if "reco" in var else "truth_pt", nobj)
            axb.hist(bkgv,bins=bins,histtype='step',label='Truth' if 'truth' in var else reco_prefix,weights=bkgw)
            axs.hist(sigv,bins=bins,histtype='step',label='Truth' if 'truth' in var else reco_prefix)
            ax.hist(bkgv,bins=bins,histtype='step',label='bkg : '+('Truth' if 'truth' in var else reco_prefix),weights=bkgw,density=True)
            ax.hist(sigv,bins=bins,histtype='step',label='sig : '+('Truth' if 'truth' in var else reco_prefix),density=True)
        place_legend_above(axb)
        place_legend_above(axs)
        place_legend_above(ax)
        if 'pt' in var:
            axb.set_yscale('log')
            axs.set_yscale('log')
            ax.set_yscale('log')
        figb.savefig(build_debug_plot_path(config.name, 'debug_bkg_%s%s%s_%s_%i.pdf'%(prefix,var,tag,config.name,nobj)), bbox_inches='tight')
        figs.savefig(build_debug_plot_path(config.name, 'debug_sig_%s%s%s_%s_%i.pdf'%(prefix,var,tag,config.name,nobj)), bbox_inches='tight')
        fig.savefig(build_debug_plot_path(config.name, 'debug_%s%s%s_%s_%i.pdf'%(prefix,var,tag,config.name,nobj)), bbox_inches='tight')
        plt.close(figb)
        plt.close(figs)
        plt.close(fig)


def _debug_multiplicity_plots(config, sig_pairs, bkg_pairs, prefix, tag):
    """Debug histograms of the number of reco/truth objects above 50 GeV for every collection."""
    for var in ["num_truth","num_reco"]:
        figb, axb = plt.subplots()
        figs, axs = plt.subplots()
        fig, ax = plt.subplots()
        bins = np.arange(-0.5,11.5,1.)
        for r,reco_prefix in enumerate(config.reco_prefixes):
            if r>0 and "truth" in var:
                continue
            bkgv = ak.sum(bkg_pairs[reco_prefix]['truth_pt' if 'truth' in var else 'reco_pt']>50.,axis=1)
            bkgw = bkg_pairs[reco_prefix]["weight"]
            sigv = ak.sum(sig_pairs[reco_prefix]['truth_pt' if 'truth' in var else 'reco_pt']>50.,axis=1)
            axb.hist(bkgv,bins=bins,histtype='step',label='Truth' if 'truth' in var else reco_prefix,weights=bkgw)
            axs.hist(sigv,bins=bins,histtype='step',label='Truth' if 'truth' in var else reco_prefix)
            ax.hist(bkgv,bins=bins,histtype='step',label='bkg : '+('Truth' if 'truth' in var else reco_prefix),weights=bkgw,density=True)
            ax.hist(sigv,bins=bins,histtype='step',label='sig : '+('Truth' if 'truth' in var else reco_prefix),density=True)
        place_legend_above(axb)
        place_legend_above(axs)
        place_legend_above(ax)
        figb.savefig(build_debug_plot_path(config.name, 'debug_bkg_%s%s%s_%s.pdf'%(prefix,var,tag,config.name)), bbox_inches='tight')
        figs.savefig(build_debug_plot_path(config.name, 'debug_sig_%s%s%s_%s.pdf'%(prefix,var,tag,config.name)), bbox_inches='tight')
        fig.savefig(build_debug_plot_path(config.name, 'debug_%s%s%s_%s.pdf'%(prefix,var,tag,config.name)), bbox_inches='tight')
        plt.close(figb)
        plt.close(figs)
        plt.close(fig)


def process_run(config: RunConfig, debug=True, prefix="", corr_cache=""):
    sig_pairs = match_reco_truth(
        files=config.signal_files,
        weights=[1.]*len(config.signal_files),
        reco_prefixes=config.reco_prefixes,
        truth_prefix=config.truth_prefix,
        truth_suffix=config.truth_suffix,
        dr_max=config.dr_max,
        tree_name=config.tree,
        extra_vars=config.extra_vars,
        reco_sources=config.reco_sources,
        extra_var_branches=config.extra_var_branches,
        pt_min=config.pt_min,
        reco_pt_min=config.reco_pt_min,
        truth_pt_min=config.truth_pt_min,
        reco_iso_dr=config.reco_iso_dr,
        truth_iso_dr=config.truth_iso_dr,
        **config.match_dict
    )

    bkg_pairs = match_reco_truth(
        files=config.background_files,
        weights=config.background_weights,
        reco_prefixes=config.reco_prefixes,
        truth_prefix=config.truth_prefix,
        truth_suffix=config.truth_suffix,
        dr_max=config.dr_max,
        tree_name=config.tree,
        extra_vars=config.extra_vars,
        reco_sources=config.reco_sources,
        extra_var_branches=config.extra_var_branches,
        pt_min=config.pt_min,
        reco_pt_min=config.reco_pt_min,
        truth_pt_min=config.truth_pt_min,
        reco_iso_dr=config.reco_iso_dr,
        truth_iso_dr=config.truth_iso_dr,
        **config.match_dict
    )

    if debug:
        print(config.name)
        print('sig')
        print(sig_pairs)
        print('bkg')
        print(bkg_pairs)

    results = []

    if debug:
        for nobj in config.nobjs:
            _debug_kinematic_plots(config, sig_pairs, bkg_pairs, nobj, prefix, "_nocorr")
        _debug_multiplicity_plots(config, sig_pairs, bkg_pairs, prefix, "_nocorr")

    for reco_prefix, reco_label in zip(config.reco_prefixes, config.reco_labels):
        sp = sig_pairs[reco_prefix]
        bp = bkg_pairs[reco_prefix]

        # ---- response and corrections (computed once per collection, before any nobj loop) ----
        response_uncorr, resol_uncorr, _ = compute_response(
            bp,
            config.truth_pt_bins,
            config.truth_eta_bins,
            config.pt_min,
            debug=f"{config.name}_{reco_prefix}" if corr_cache=="" else None
        )

        # handle corrections (compute or load)
        corr_filename = corr_cache
        if corr_filename == "":
            # Default filename if computing
            corr_filename = f"{prefix}{config.name}_corrs_{reco_prefix}.pkl"

        if corr_cache != "":
            # ----  Corrections ----
            print(f"Loading corrections from {corr_filename}_{reco_prefix}")
            rhosub,corrector = load_corrs(f"{corr_filename}_{reco_prefix}.pkl")[reco_prefix]

            if config.do_rho_sub:
                rhosub(bp)
                rhosub(sp)
            corrector(bp)
            corrector(sp)

        else:
            # ---- COMPUTE Corrections ----
            corrs = {}

            print(f"Starting {reco_prefix}...")
            # compute area subtraction (if requested)
            if config.do_rho_sub:
                rhosub = AreaSubtractor(
                    bp,
                    config.truth_eta_bins,
                    debug=prefix+reco_prefix+"_"
                )

                rhosub(bp)
                rhosub(sp)

                # recompute response corrections for debug
                response_acorr, resol_acorr, _ = compute_response(
                    bp,
                    config.truth_pt_bins,
                    config.truth_eta_bins,
                    config.pt_min,
                    debug=None
                )
            else:
                rhosub = None

            # compute response correction
            corrector = ResponseInterpolator(
                response_acorr if config.do_rho_sub else response_uncorr,
                resol_acorr if config.do_rho_sub else resol_uncorr,
                config.truth_pt_bins,
                config.truth_eta_bins,
                debug=f"{config.name}_{reco_prefix}_responseinterp",
                spline_lambda=config.spline_lambdas.get(reco_prefix, 1e-5),
            )

            #apply corrections
            corrector(bp)
            corrector(sp)

            corrs[reco_prefix] = (rhosub,corrector)
            # Save once
            print(f"Saving corrections to {corr_filename}")
            save_corrs(corrs, corr_filename)

        # recompute response corrections
        response_corr, resol_corr, _ = compute_response(
            bp,
            config.truth_pt_bins,
            config.truth_eta_bins,
            config.pt_min,
            debug=f"{config.name}_{reco_prefix}_corr" if corr_cache=="" else None
        )
        print(f"Finished {reco_prefix} response...")

        del corrector
        del rhosub
        gc.collect()

        # ---- per-collection caches on the corrected pairs ----
        sig_w = np.asarray(ak.to_numpy(sp["weight"]), dtype=np.float64)
        bkg_w = np.asarray(ak.to_numpy(bp["weight"]), dtype=np.float64)
        sig_reco_order = kth_sort_order(sp, "reco_pt")
        bkg_reco_order = kth_sort_order(bp, "reco_pt")

        for n, nobj in enumerate(config.nobjs):
            # nobj-th leading reco pt and event selections, computed once per (collection, nobj)
            sig_reco_k = kth_from_order(sp, "reco_pt", sig_reco_order, nobj)
            bkg_reco_k = kth_from_order(bp, "reco_pt", bkg_reco_order, nobj)
            sig_sel_masks = [_resolve_selector(sel, sp, nobj) for sel in config.sels]
            bkg_sel_masks = [_resolve_selector(sel, bp, nobj) for sel in config.sels]
            sig_rate_masks = [_resolve_selector(sel, sp, nobj) for sel in config.rate_sels]
            bkg_rate_masks = [_resolve_selector(sel, bp, nobj) for sel in config.rate_sels]
            turnon_cache = {}
            full_eff_cache = {}

            def event_selections(s, dijet_threshold):
                """Denominator selections for selector s (plus the truth multiplicity cut for m_jj turn-ons)."""
                sig_mask = sig_sel_masks[s]
                bkg_mask = bkg_sel_masks[s]
                if dijet_threshold is not None:
                    sig_mask = sig_mask & truth_multiplicity_selector(sp, nobj, pt_min=dijet_threshold)
                    bkg_mask = bkg_mask & truth_multiplicity_selector(bp, nobj, pt_min=dijet_threshold)
                return sig_mask, bkg_mask

            def full_efficiencies(r, s, dijet_threshold):
                """Full efficiency scans; they only depend on (rate selector, selector, m_jj threshold)."""
                key = (r, s, dijet_threshold)
                if key not in full_eff_cache:
                    sig_mask, bkg_mask = event_selections(s, dijet_threshold)
                    full_eff_cache[key] = (
                        compute_full_efficiency(
                            sp, config.truth_pt_bins, nobj, sig_mask,
                            numerator_selector=sig_rate_masks[r], weights=False,
                            reco_pt_kth=sig_reco_k, event_weights=sig_w,
                        ),
                        compute_full_efficiency(
                            bp, config.truth_pt_bins, nobj, bkg_mask,
                            numerator_selector=bkg_rate_masks[r], weights=True,
                            reco_pt_kth=bkg_reco_k, event_weights=bkg_w,
                        ),
                    )
                return full_eff_cache[key]

            for r in range(len(config.rate_sels)):
                # Compute threshold for fixed background efficiency
                rate_eff = config.rates[n]/31_000.
                threshold,actual_eff = compute_pt_threshold(
                    bp,
                    rate_eff,
                    nobj,
                    selector=bkg_rate_masks[r],
                    reco_pt_kth=bkg_reco_k,
                    event_weights=bkg_w,
                ) #in kHz
                print(f"For {reco_prefix}, n={nobj}, target rate efficiency of {rate_eff:.6f}, threshold of {threshold:.6f} gives actual rate efficiency of {actual_eff:.6f}")

                # fixed-rate threshold first, then the fixed trigger thresholds
                thresholds = [(True, threshold, config.rates[n])]
                for trig_threshold in config.triggers[n]:
                    rate, actual_eff = compute_rate(
                        bp,
                        trig_threshold,
                        nobj,
                        selector=bkg_rate_masks[r],
                        reco_pt_kth=bkg_reco_k,
                        event_weights=bkg_w,
                    )
                    print(f"For {reco_prefix}, n={nobj}, trigger threshold of {trig_threshold:.1f} gives actual rate efficiency of {actual_eff:.6f} (rate {rate:.6f} kHz)")
                    thresholds.append((False, trig_threshold, rate))

                for fixrate, thr, rate in thresholds:
                    for t_idx, (turnon_var, turnon_fn, turnon_label, turnon_bins) in enumerate(zip(
                        config.turnon_vars,
                        config.turnon_fns,
                        config.turnon_var_labels,
                        config.turnon_bins,
                    )):
                        is_dijet = turnon_var == "_dijet_mass"
                        if is_dijet:
                            turnon_values = dijet_mass_turnon_var(sp, nobj, pt_min=thr)
                        else:
                            if t_idx not in turnon_cache:
                                turnon_cache[t_idx] = turnon_fn(sp, nobj)
                            turnon_values = turnon_cache[t_idx]
                        dijet_threshold = thr if is_dijet else None

                        for s in range(len(config.sels)):
                            sig_mask, _ = event_selections(s, dijet_threshold)

                            # Signal efficiency vs turn-on variable
                            centers, eff, _,_, err = compute_signal_efficiency(
                                sp,
                                thr,
                                turnon_bins,
                                nobj,
                                sig_mask,
                                numerator_selector=sig_rate_masks[r],
                                turnon_values=turnon_values,
                                reco_pt_kth=sig_reco_k,
                                event_weights=sig_w,
                            )

                            (full_sig_eff, full_sig_err), (full_bkg_eff, full_bkg_err) = full_efficiencies(r, s, dijet_threshold)

                            results.append(
                                RunResult(
                                    name=config.name,
                                    sel_label=config.sel_labels[s],
                                    rate_sel_label=config.rate_sel_labels[r],
                                    reco=reco_prefix,
                                    reco_label=reco_label,
                                    nobj=nobj,
                                    fixrate=fixrate,
                                    threshold=thr,
                                    rate=rate,
                                    truth_pt_bins=config.truth_pt_bins,
                                    truth_eta_bins=config.truth_eta_bins,
                                    signal_efficiency=eff,
                                    signal_efficiency_error=err,
                                    full_sig_efficiency=full_sig_eff,
                                    full_sig_efficiency_error=full_sig_err,
                                    full_bkg_efficiency=full_bkg_eff,
                                    full_bkg_efficiency_error=full_bkg_err,
                                    response_uncorr=response_uncorr,
                                    response_corr=response_corr,
                                    resol_uncorr=resol_uncorr,
                                    resol_corr=resol_corr,
                                    turnon_var=turnon_var,
                                    turnon_label=turnon_label,
                                    turnon_bins=turnon_bins,
                                )
                            )

        del sig_reco_order, bkg_reco_order
        gc.collect()

    if debug:
        for nobj in config.nobjs:
            _debug_kinematic_plots(config, sig_pairs, bkg_pairs, nobj, prefix, "")
        _debug_multiplicity_plots(config, sig_pairs, bkg_pairs, prefix, "")

    return results

def save_run_result(result: RunResult, path):
    np.savez(
        path,
        name=result.name,
        reco=result.reco,
        reco_label=result.reco_label,
        nobj=result.nobj,
        sel_label=result.sel_label,
        rate_sel_label=result.rate_sel_label,
        fixrate=result.fixrate,
        threshold=result.threshold,
        rate=result.rate,
        truth_pt_bins=result.truth_pt_bins,
        truth_eta_bins=result.truth_eta_bins,
        signal_efficiency=result.signal_efficiency,
        signal_efficiency_error=result.signal_efficiency_error,
        full_sig_efficiency=result.full_sig_efficiency,
        full_sig_efficiency_error=result.full_sig_efficiency_error,
        full_bkg_efficiency=result.full_bkg_efficiency,
        full_bkg_efficiency_error=result.full_bkg_efficiency_error,
        response_uncorr=np.array(result.response_uncorr, dtype=object),
        response_corr=np.array(result.response_corr, dtype=object),
        resol_uncorr=np.array(result.resol_uncorr, dtype=object),
        resol_corr=np.array(result.resol_corr, dtype=object),
        turnon_var=result.turnon_var,
        turnon_label=result.turnon_label,
        turnon_bins=result.turnon_bins,
    )

def load_run_result(path):
    data = np.load(path, allow_pickle=True)
    turnon_var = data["turnon_var"].item() if "turnon_var" in data else "truth_pt"
    turnon_label = data["turnon_label"].item() if "turnon_label" in data else "Truth p_{T}"
    turnon_bins = data["turnon_bins"] if "turnon_bins" in data else data["truth_pt_bins"]
    return RunResult(
        name=data["name"].item(),
        reco=data["reco"].item(),
        reco_label=data["reco_label"].item() if "reco_label" in data else data["reco"].item(),
        nobj=data["nobj"].item(),
        sel_label=data["sel_label"].item(),
        rate_sel_label=data["rate_sel_label"].item(),
        fixrate=data["fixrate"].item(),
        threshold=data["threshold"].item(),
        rate=data["rate"].item(),
        truth_pt_bins=data["truth_pt_bins"],
        truth_eta_bins=data["truth_eta_bins"],
        signal_efficiency=data["signal_efficiency"],
        signal_efficiency_error=data["signal_efficiency_error"],
        full_sig_efficiency=data["full_sig_efficiency"],
        full_sig_efficiency_error=data["full_sig_efficiency_error"],
        full_bkg_efficiency=data["full_bkg_efficiency"],
        full_bkg_efficiency_error=data["full_bkg_efficiency_error"],
        response_uncorr=data["response_uncorr"],
        response_corr=data["response_corr"],
        resol_uncorr=data["resol_uncorr"],
        resol_corr=data["resol_corr"],
        turnon_var=turnon_var,
        turnon_label=turnon_label,
        turnon_bins=turnon_bins,
    )

import scipy.optimize as opt

# Define the logistic function (e.g., a four-parameter sigmoid)
def logistic_function(x, A, B, C, D):
    """
    A: lower asymptote
    B: steepness
    C: inflection point
    D: upper asymptote
    """
    return A + (D - A) / (1 + np.exp(-B * (x - C)))

def fit_logistic(x_data, y_data, y_data_err=None):
    # Fit the curve to the data
    try:
        params, covariance = opt.curve_fit(logistic_function, x_data, y_data, sigma=y_data_err ,p0=[0, 0.05, 50, 1]) # p0 provides initial guesses for parameters
    except:
        params, covariance = [-1,-1,-1,-1], None
    # Extract the fitted parameters
    return params, covariance

def overlay_efficiency(results, suffix="", titletxt="", nobj=1, xmax=-1., noerr=False):
    plt.clf()

    numtext = {
        0:"",
        1:"Leading ",
        2:"Subleading ",
        3:"Third Leading ",
        4:"Fourth Leading "
    }   
    
    params = {}
    for i,r in enumerate(results):
        marker = get_efficiency_marker(i)
        turnon_bins = getattr(r, "turnon_bins", r.truth_pt_bins)
        turnon_label = getattr(r, "turnon_label", "Truth $p_T$ [GeV]")
        turnon_centers = 0.5*(turnon_bins[:-1]+turnon_bins[1:])
        xmask = (turnon_centers < xmax) if xmax>0. else np.ones(len(turnon_centers),dtype=bool)
        params[i],_ = fit_logistic(turnon_centers[xmask], r.signal_efficiency[xmask], np.mean(r.signal_efficiency_error,axis=0)[xmask])
        label_full = result_reco_label(r)+r' [$p_T$>'+('%.1f'%r.threshold)+'] ($\\sigma$='+('%.2f'%(1./params[i][1]))+', $p_T^{98\\%}$='+('%.2f'%(params[i][2]+np.log(49)/params[i][1]))+')'
        plt.errorbar(turnon_centers[xmask], r.signal_efficiency[xmask], None if noerr else r.signal_efficiency_error[:,xmask], marker=marker, label=label_full, color='C%i'%i, capsize=3, capthick=2, linestyle='none', mfc='none', alpha=marker_alpha(0.5), markersize=DEFAULT_MARKER_SIZE)
        if params[i][0]==-1:
            plt.plot(turnon_centers[xmask], logistic_function(turnon_centers[xmask],*params[i]), color='C%i'%i, linestyle='dashed')

    plt.xlabel(r"%s%s"%(numtext[nobj], turnon_label))
    plt.ylabel("Signal efficiency")
    plt.ylim(0,1.1)
    place_legend_above(title=titletxt)
    plt.grid(True)
    plt.savefig(build_plot_path('efficiency%s.pdf'%suffix), bbox_inches='tight')

    plt.clf()

    for i,r in enumerate(results):
        marker = get_efficiency_marker(i)
        turnon_bins = getattr(r, "turnon_bins", r.truth_pt_bins)
        turnon_centers = 0.5*(turnon_bins[:-1]+turnon_bins[1:])
        shiftmask = turnon_centers < (params[i][2]+3.*np.log(49)/params[i][1]) # 2x the shift from 50 to 98 to make sure its visible
        params_adj,_ = fit_logistic(turnon_centers[shiftmask]-(params[i][2]+np.log(49)/params[i][1]), r.signal_efficiency[shiftmask], np.mean(r.signal_efficiency_error,axis=0)[shiftmask])
        label_full = result_reco_label(r)+r' [$p_T$>'+('%.1f'%r.threshold)+'] ($\\hat{\\sigma}$='+('%.2f'%(1./params_adj[1]))+', $\\hat{p}_T^{98\\%}$='+('%.2f'%(params_adj[2]+np.log(49)/params_adj[1]))+')'
        plt.errorbar(turnon_centers[shiftmask]-(params[i][2]+np.log(49)/params[i][1]), r.signal_efficiency[shiftmask], None if noerr else r.signal_efficiency_error[:,shiftmask], marker=marker, label=label_full, color='C%i'%i, capsize=3, capthick=2, linestyle='none', mfc='none', alpha=marker_alpha(0.5), markersize=DEFAULT_MARKER_SIZE)
        if params[i][0]==-1:
            plt.plot(turnon_centers[shiftmask]-(params[i][2]+np.log(49)/params[i][1]), logistic_function(turnon_centers[shiftmask]-(params[i][2]+np.log(49)/params[i][1]),*params_adj), color='C%i'%i, linestyle='dashed')

    plt.xlabel(r"%s$\Delta$ %s"%(numtext[nobj],turnon_label))
    plt.ylabel("Signal efficiency")
    plt.ylim(0,1.1)
    place_legend_above(title=titletxt)
    plt.grid(True)
    plt.savefig(build_plot_path('efficiency%s_corr.pdf'%suffix), bbox_inches='tight')

def overlay_resp_resol(results, corr=False, prefix=""):

    nres = len(results)
    for ie in range(len(results[0].truth_eta_bins)-1):
        plt.clf()
        for i,r in enumerate(results):
            pt_centers = 0.5*(r.truth_pt_bins[:-1]+r.truth_pt_bins[1:])
            pt_widths = 0.5*(pt_centers-r.truth_pt_bins[:-1])
            plt.errorbar(pt_centers+(pt_widths)*((0.5*nres)-float(i)), 
                         (r.response_corr if corr else r.response_uncorr)[ie*len(pt_centers):(ie+1)*len(pt_centers)], 
                         (r.resol_corr if corr else r.resol_uncorr)[ie*len(pt_centers):(ie+1)*len(pt_centers)],
                         marker='o', color='C%i'%i, capsize=3, capthick=2, linestyle='none', alpha=marker_alpha(0.5), markersize=DEFAULT_MARKER_SIZE,
                         label=result_reco_label(r)+", "+r.name)

        plt.ylabel(r"%sResponse (Reco $p_T$ / Truth $p_T$)"%("Corrected " if corr else ""))
        plt.xlabel(r"Truth $p_T$ [GeV]")
        plt.title(r"$%.2f<\eta<%.2f$"%(r.truth_eta_bins[ie],r.truth_eta_bins[ie+1]))
        place_legend_above()
        plt.grid(True)
        plt.savefig(build_plot_path(prefix+'response'+("_corr" if corr else "")+'_eta_%.2f_%.2f.pdf'%(r.truth_eta_bins[ie],r.truth_eta_bins[ie+1])), bbox_inches='tight')

        plt.clf()

        for i,r in enumerate(results):
            pt_centers = 0.5*(r.truth_pt_bins[:-1]+r.truth_pt_bins[1:])
            plt.plot(pt_centers+(pt_widths)*((0.5*nres)-float(i)), 
                         (r.resol_corr if corr else r.resol_uncorr)[ie*len(pt_centers):(ie+1)*len(pt_centers)], 
                         marker='o', color='C%i'%i, linestyle='none', alpha=marker_alpha(0.5), markersize=DEFAULT_MARKER_SIZE,
                         label=result_reco_label(r)+", "+r.name)

        plt.ylabel(r"%sResolution"%("Corrected " if corr else ""))
        plt.xlabel(r"Truth $p_T$ [GeV]")
        plt.title(r"$%.2f<\eta<%.2f$"%(r.truth_eta_bins[ie],r.truth_eta_bins[ie+1]))
        place_legend_above()
        plt.grid(True)
        plt.savefig(build_plot_path(prefix+'resol'+("_corr" if corr else "")+'_eta_%.2f_%.2f.pdf'%(r.truth_eta_bins[ie],r.truth_eta_bins[ie+1])), bbox_inches='tight')

def overlay_full_effs(results, suffix="", nobj=1, xmax=-1.):
    plt.clf()

    numtext = {
        1:"Leading",
        2:"Subleading",
        3:"Third Leading",
        4:"Fourth Leading"
    }   
    
    for i,r in enumerate(results):
        xmask = (r.truth_pt_bins<xmax) if xmax>0. else np.ones(len(r.truth_pt_bins),dtype=bool)
        marker = get_efficiency_marker(i)
        plt.errorbar(r.truth_pt_bins[xmask],
                     r.full_sig_efficiency[xmask],
                     yerr=r.full_sig_efficiency_error[:,xmask],
                     marker=marker, color='C%i'%i, linestyle='none', alpha=marker_alpha(0.5), markersize=DEFAULT_MARKER_SIZE,
                     label=result_reco_label(r))
    plt.ylabel(r"Signal efficiency")
    plt.xlabel(r"%s $p_T$ [GeV]"%numtext[nobj])
    place_legend_above()
    plt.grid(True)
    plt.savefig(build_plot_path('efficiency_full_signal%s.pdf'%(suffix)), bbox_inches='tight')
    
    plt.clf()
    
    for i,r in enumerate(results):
        xmask = (r.truth_pt_bins<xmax) if xmax>0. else np.ones(len(r.truth_pt_bins),dtype=bool)
        marker = get_efficiency_marker(i)
        plt.errorbar(r.truth_pt_bins[xmask],
                     r.full_bkg_efficiency[xmask]*31_000.,
                     yerr=r.full_bkg_efficiency_error[:,xmask]*31_000.,
                     marker=marker, color='C%i'%i, linestyle='none', alpha=marker_alpha(0.5), markersize=DEFAULT_MARKER_SIZE,
                     label=result_reco_label(r))
    plt.ylabel(r"Background rate [kHz]")
    plt.ylim(1,9e4)
    plt.yscale('log')
    plt.xlabel(r"%s $p_T$ [GeV]"%numtext[nobj])
    place_legend_above()
    plt.grid(True)
    plt.savefig(build_plot_path('efficiency_full_background%s.pdf'%(suffix)), bbox_inches='tight')
    
    plt.clf()
    
    for i,r in enumerate(results):
        xmask = (r.truth_pt_bins<xmax) if xmax>0. else np.ones(len(r.truth_pt_bins),dtype=bool)
        marker = get_efficiency_marker(i)
        plt.errorbar(r.full_sig_efficiency,
                     r.full_bkg_efficiency*31_000.,
                     xerr=r.full_sig_efficiency_error,
                     yerr=r.full_bkg_efficiency_error*31_000.,
                     marker=marker, color='C%i'%i, linestyle='none', alpha=marker_alpha(0.5), markersize=DEFAULT_MARKER_SIZE,
                     label=result_reco_label(r))
    plt.ylabel(r"Background rate [kHz]")
    plt.ylim(1,9e4)
    plt.yscale('log')
    plt.xlabel(r"Signal efficiency")
    plt.title(numtext[nobj]+" "+results[0].name)
    place_legend_above()
    plt.grid(True)
    plt.savefig(build_plot_path('efficiency_full_combined%s.pdf'%(suffix)), bbox_inches='tight')

# selectors
def null_selector(pairs, nobj):
    #print('Null selector fraction: 1.')
    return np.ones(len(pairs),dtype=bool)

def truth_multiplicity_selector(pairs, nobj, pt_min=0.0, coll="truth"):
    """
    Select events containing at least ``nobj`` objects in ``coll`` above
    ``pt_min``.

    This is useful for turn-on variables such as VBF ``m_{jj}``, where the
    denominator should match the truth-level multiplicity implied by the
    trigger being studied (for example, a 3-jet trigger should only consider
    events with at least three truth jets above that trigger threshold).
    """
    pt = pairs[f"{coll}_pt"]
    return ak.to_numpy(ak.sum(pt >= pt_min, axis=1) >= nobj)

def barrel_selector(pairs, nobj, maxeta=1.4, coll="truth"):
    """
    Select events where all of the leading nobj objects satisfy
    abs(eta) < maxeta.

    Objects are ranked by descending {coll}_pt in each event.
    """
    eta_field = f"{coll}_eta"
    pt_field = f"{coll}_pt"

    has_nobj = ak.to_numpy(ak.num(pairs[eta_field], axis=1) >= nobj)
    event_sel = has_nobj.copy()

    order = kth_sort_order(pairs, pt_field)
    for n in range(1, nobj + 1):
        eta_n = kth_from_order(pairs, eta_field, order, n)
        event_sel &= np.abs(eta_n) < maxeta

    return event_sel

def boosted_truth_selector(pairs, nobj, dr_threshold=0.7, truth_pt_threshold=40.0, debug=0, chunk_size=10000):
    """
    Memory-optimized version that:
    1. Uses chunking to limit memory usage
    2. Explicitly deletes intermediates
    3. Avoids creating unnecessary pair structures
    
    Parameters
    ----------
    pairs : dict
        Dictionary with "truth_pt", "truth_eta", "truth_phi" fields
    dr_threshold : float
        Delta R threshold for considering jets "close"
    truth_pt_threshold : float
        Minimum truth-jet pT threshold for jets considered in close-pair checks
    debug : int
        Number of events to print debug info for
    chunk_size : int
        Number of events to process at once (tune based on available memory)
        
    Returns
    -------
    event_sel : np.ndarray
        Boolean array indicating which events have close jet pairs
    """
    truth_pt = pairs["truth_pt"]
    truth_eta = pairs["truth_eta"]
    truth_phi = pairs["truth_phi"]
    
    n_events = len(truth_eta)
    event_sel = np.zeros(n_events, dtype=bool)
    
    # Process in chunks to limit memory usage
    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)
        
        # Extract chunk and keep only truth jets above the configured pT threshold
        pt_chunk = truth_pt[chunk_start:chunk_end]
        eta_chunk = truth_eta[chunk_start:chunk_end]
        phi_chunk = truth_phi[chunk_start:chunk_end]
        truth_pt_mask = pt_chunk >= truth_pt_threshold
        eta_chunk = eta_chunk[truth_pt_mask]
        phi_chunk = phi_chunk[truth_pt_mask]
        
        # Create combinations for this chunk only
        eta_pairs = ak.combinations(eta_chunk, 2, axis=1)
        phi_pairs = ak.combinations(phi_chunk, 2, axis=1)
        
        # Extract components
        eta_1 = eta_pairs['0']
        eta_2 = eta_pairs['1']
        phi_1 = phi_pairs['0']
        phi_2 = phi_pairs['1']
        
        # Clean up pair structures immediately
        del eta_pairs, phi_pairs
        
        # Calculate Delta Eta
        delta_eta = eta_1 - eta_2
        del eta_1, eta_2  # Cleanup
        
        # Calculate Delta Phi with wrapping
        delta_phi = phi_1 - phi_2
        del phi_1, phi_2  # Cleanup
        delta_phi = np.remainder(delta_phi + np.pi, 2 * np.pi) - np.pi
        
        # Calculate Delta R
        delta_r_squared = delta_eta**2 + delta_phi**2
        del delta_eta, delta_phi  # Cleanup
        delta_r = np.sqrt(delta_r_squared)
        del delta_r_squared  # Cleanup
        
        # Check for close pairs
        is_close_pair = delta_r < dr_threshold
        
        # Debug output for this chunk (if requested)
        if debug > 0:
            debug_end = min(debug, chunk_end - chunk_start)
            for i in range(debug_end):
                global_idx = chunk_start + i
                print(f'debug: {global_idx}')
                print(f'\ttruth_pt_threshold: {truth_pt_threshold}')
                print(f'\teta: {eta_chunk[i]}')
                print(f'\tphi: {phi_chunk[i]}')
                print(f'\tdelta_r: {delta_r[i]}')
                print(f'\tmin delta_r: {ak.min(delta_r[i])}')
            debug -= debug_end  # Reduce remaining debug count
        
        # Check if any pair is close
        events_with_close_pair = ak.any(is_close_pair, axis=1)
        del is_close_pair  # Cleanup
        
        # Store results for this chunk
        event_sel[chunk_start:chunk_end] = ak.to_numpy(events_with_close_pair)
        
        # Clean up chunk data
        del pt_chunk, truth_pt_mask, eta_chunk, phi_chunk, delta_r, events_with_close_pair
    
    return event_sel

def hh_mass_window_selector(pairs, nobj, m_min=75., m_max=175., coll='reco', debug=0, chunk_size=10000):
    """
    Returns an event mask that is True if the event contains TWO DISJOINT PAIRS
    of objects, chosen from the top `top_n` objects (by input order), such that
    both pairs have invariant mass in [m_min, m_max]. Otherwise returns False.

    The only disjoint pairings (perfect matchings) are:
      (0,1)&(2,3), (0,2)&(1,3), (0,3)&(1,2)

    Assumes objects are approximately massless (m^2 = 2 pT1 pT2 (cosh(deta) - cos(dphi))).

    Parameters
    ----------
    pairs : dict
        Dictionary with at least "truth_pt", "truth_eta", "truth_phi" (awkward arrays).
        Arrays are expected to be jagged: shape (n_events, n_objects_per_event).
        The "top" objects are taken as the first `top_n` entries in each event.
    m_min, m_max : float
        Invariant mass window bounds (same units as pT).
    debug : int
        Number of events to print debug info for.
    chunk_size : int
        Number of events to process at once.

    Returns
    -------
    event_sel : np.ndarray
        Boolean array of length n_events.
    """

    pt  = pairs[f"{coll}_pt"]
    eta = pairs[f"{coll}_eta"]
    phi = pairs[f"{coll}_phi"]

    n_events = len(pt)
    event_sel = np.zeros(n_events, dtype=bool)


    def m_massless(pt1, eta1, phi1, pt2, eta2, phi2):
        deta = eta1 - eta2
        dphi_nowrap = phi1 - phi2
        dphi = np.remainder(dphi_nowrap + np.pi, 2 * np.pi) - np.pi
        m2 = 2.0 * pt1 * pt2 * (np.cosh(deta) - np.cos(dphi))
        # protect against tiny negative values from numerical precision
        m2 = ak.where(m2 > 0, m2, 0.0)
        return np.sqrt(m2)

    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)

        pt_chunk  = pt[chunk_start:chunk_end]
        eta_chunk = eta[chunk_start:chunk_end]
        phi_chunk = phi[chunk_start:chunk_end]

        # Require at least 4 objects in the event
        has4 = ak.num(pt_chunk, axis=1) >= 4
        
        # ---- NEW: sort objects by descending pt per event ----
        order = ak.argsort(
            ak.where(
                ak.is_none(pt_chunk) | np.isnan(pt_chunk),
                0,
                pt_chunk
            ), axis=1, ascending=False
        )

        pt_sorted  = pt_chunk[order]
        eta_sorted = eta_chunk[order]
        phi_sorted = phi_chunk[order]

        del order  # cleanup early

        # Pad to length 4, then take the top 4
        pt4  = ak.fill_none(ak.pad_none(pt_sorted,  4, axis=1)[:, :4],  0.0)
        eta4 = ak.fill_none(ak.pad_none(eta_sorted, 4, axis=1)[:, :4], 0.0)
        phi4 = ak.fill_none(ak.pad_none(phi_sorted, 4, axis=1)[:, :4], 0.0)

        # Extract the 4 leading objects
        pt0, pt1, pt2, pt3 = pt4[:, 0], pt4[:, 1], pt4[:, 2], pt4[:, 3]
        e0,  e1,  e2,  e3  = eta4[:, 0], eta4[:, 1], eta4[:, 2], eta4[:, 3]
        p0,  p1_, p2_, p3_ = phi4[:, 0], phi4[:, 1], phi4[:, 2], phi4[:, 3]

        # Compute masses for the 3 possible disjoint pairings
        m01 = m_massless(pt0, e0, p0,  pt1, e1, p1_)
        m23 = m_massless(pt2, e2, p2_, pt3, e3, p3_)

        m02 = m_massless(pt0, e0, p0,  pt2, e2, p2_)
        m13 = m_massless(pt1, e1, p1_, pt3, e3, p3_)

        m03 = m_massless(pt0, e0, p0,  pt3, e3, p3_)
        m12 = m_massless(pt1, e1, p1_, pt2, e2, p2_)

        inwin01 = (m01 >= m_min) & (m01 <= m_max)
        inwin23 = (m23 >= m_min) & (m23 <= m_max)
        inwin02 = (m02 >= m_min) & (m02 <= m_max)
        inwin13 = (m13 >= m_min) & (m13 <= m_max)
        inwin03 = (m03 >= m_min) & (m03 <= m_max)
        inwin12 = (m12 >= m_min) & (m12 <= m_max)

        # Event passes if ANY matching has both pairs in window, and the event truly had >=4 objects
        pass_evt = has4 & (
            (inwin01 & inwin23) |
            (inwin02 & inwin13) |
            (inwin03 & inwin12)
        )

        # Optional debug printing
        if debug > 0:
            debug_end = min(debug, chunk_end - chunk_start)
            for i in range(debug_end):
                global_idx = chunk_start + i
                print(f"debug: {global_idx}")
                print(f"\tN obj: {len(pt_sorted[i])}")
                print(f"\tpt[:4]:  {pt_sorted[i][:4]}")
                print(f"\teta[:4]: {eta_sorted[i][:4]}")
                print(f"\tphi[:4]: {phi_sorted[i][:4]}")
                print(f"\tm01,m23: {m01[i]}, {m23[i]}")
                print(f"\tm02,m13: {m02[i]}, {m13[i]}")
                print(f"\tm03,m12: {m03[i]}, {m12[i]}")
                print(f"\tpass: {pass_evt[i]}")
            debug -= debug_end

        event_sel[chunk_start:chunk_end] = ak.to_numpy(pass_evt)

        # Cleanup
        del pt_chunk, eta_chunk, phi_chunk
        del pt4, eta4, phi4
        del pt0, pt1, pt2, pt3, e0, e1, e2, e3, p0, p1_, p2_, p3_
        del m01, m23, m02, m13, m03, m12
        del inwin01, inwin23, inwin02, inwin13, inwin03, inwin12
        del has4, pass_evt

    return event_sel


def eratio_selector(pairs, nobj, threshold=0.65):
    """
    Select events where each of the ``nobj`` leading (in reco pt) reco objects
    has Eratio above ``threshold``.

    If ``reco_Eratio`` is not available for the current collection, this
    selector is a no-op and all events pass.
    """
    reco_eratio = np.ones(len(pairs), dtype=bool)

    if "reco_Eratio" in ak.fields(pairs):
        order = kth_sort_order(pairs, "reco_pt")
        for n in range(1,nobj+1):
            reco_eratio = reco_eratio & (kth_from_order(pairs, "reco_Eratio", order, n) > threshold)

    return reco_eratio

# turn-on variable functions
def truth_pt_turnon_var(pairs, nobj):
    return select_kth(pairs, "truth_pt", "truth_pt", nobj)

def dijet_mass_turnon_var(pairs, nobj, pt_min=0.0, coll="truth"):
    pt = pairs[f"{coll}_pt"]
    eta = pairs[f"{coll}_eta"]
    phi = pairs[f"{coll}_phi"]

    if pt_min is None:
        pt_min = 0.0

    valid = pt > pt_min
    pt = pt[valid]
    eta = eta[valid]
    phi = phi[valid]

    combos = ak.combinations(pt, 2, axis=1)
    eta_pairs = ak.combinations(eta, 2, axis=1)
    phi_pairs = ak.combinations(phi, 2, axis=1)

    pt1 = combos["0"]
    pt2 = combos["1"]
    eta1 = eta_pairs["0"]
    eta2 = eta_pairs["1"]
    phi1 = phi_pairs["0"]
    phi2 = phi_pairs["1"]

    deta = eta1 - eta2
    dphi = ak.where(phi1 - phi2 > np.pi, phi1 - phi2 - 2 * np.pi, phi1 - phi2)
    dphi = ak.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    m2 = 2.0 * pt1 * pt2 * (np.cosh(deta) - np.cos(dphi))
    mass = np.sqrt(m2)

    abs_deta = np.abs(deta)
    max_idx = ak.argmax(abs_deta, axis=1, keepdims=True)
    mass_max = ak.flatten(mass[max_idx], axis=1)

    return ak.fill_none(mass_max, np.nan)

base_dir = '/eos/home-d/drankin/GEPEnc/'
