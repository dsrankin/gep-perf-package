import uproot
import awkward as ak
import numpy as np
import vector
from tqdm import tqdm
import math
import gc
import matplotlib.pyplot as plt

vector.register_awkward()

from dataclasses import dataclass, replace
from typing import Callable, Optional

import os

DEFAULT_PLOTDIR = 'perf_plots'
DEFAULT_RESDIR = 'perf_results'

plotdir = os.environ.get('GEP_PERF_PLOTDIR', DEFAULT_PLOTDIR)
resdir = os.environ.get('GEP_PERF_RESDIR', DEFAULT_RESDIR)
os.makedirs(plotdir, exist_ok=True)
os.makedirs(resdir, exist_ok=True)



import os

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
    truth_pt_bins: np.ndarray
    truth_eta_bins: np.ndarray
    do_rho_sub: bool
    rates: list[float]
    triggers :list[list[float]]
    extra_vars: dict[str, list[str]]
    turnon_vars: list[Callable[[ak.Array, int], ak.Array]]
    turnon_var_labels: list[str]
    turnon_bins: list[np.ndarray]
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
        if len(self.nobjs)!=len(self.triggers):
            raise ValueError(f"Number of objects ({len(self.nobjs)}) and triggers ({len(self.triggers)}) must be the same length")
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
    turnon_label: str
    turnon_bins: np.ndarray


def result_reco_label(result: RunResult) -> str:
    return getattr(result, "reco_label", result.reco)

def delta_phi(phi1,phi2):
    dphi = phi1-phi2
    dphi = dphi+2*np.pi*(dphi<=-np.pi).astype(np.float32)
    dphi = dphi-2*np.pi*(dphi>np.pi).astype(np.float32)
    return dphi

def match_chunk_vectorized(chunk, reco_prefixes, reco_branches, truth_branches, dr_max,
                          reco_iso_dr=0.4, truth_iso_dr=0.4,
                          reco_pt_min=None, truth_pt_min=None, pt_min=None,
                          extra_vars_by_prefix=None):
    """
    Vectorized matching for an entire chunk of events using awkward arrays.
    Non-greedy: each truth object matches to its closest reco object within dr_max.
    Returns dictionaries of awkward arrays indexed by reco_prefix.
    """
    # Extract truth arrays once (shared across all reco prefixes)
    t_pt = chunk[truth_branches[0]]
    t_eta = chunk[truth_branches[1]]
    t_phi = chunk[truth_branches[2]]
    
    if pt_min is None:
        pt_min = -1.

    # Apply truth pT cuts if specified
    if truth_pt_min is not None:
        t_mask = t_pt > truth_pt_min
        t_pt = t_pt[t_mask]
        t_eta = t_eta[t_mask]
        t_phi = t_phi[t_mask]
    
    # Apply truth isolation if specified
    t_isolated = compute_isolation_awkward(t_eta, t_phi, truth_iso_dr)
#         t_pt = t_pt[t_isolated]
#         t_eta = t_eta[t_isolated]
#         t_phi = t_phi[t_isolated]
    
    if extra_vars_by_prefix is None:
        extra_vars_by_prefix = {}

    results = {
        'reco_pt': {}, 'reco_eta': {}, 'reco_phi': {},
        'truth_pt': {}, 'truth_eta': {}, 'truth_phi': {},
        'reco_extra': {},
    }
    
    for reco_prefix in reco_prefixes:
        r_pt = chunk[reco_branches[reco_prefix][0]]
        r_eta = chunk[reco_branches[reco_prefix][1]]
        r_phi = chunk[reco_branches[reco_prefix][2]]
        r_extra = {}
        for extra_name in extra_vars_by_prefix.get(reco_prefix, []):
            r_extra[extra_name] = chunk[f"{reco_prefix}_{extra_name}"]

        # Apply pT cuts if specified
        r_mask = r_pt > pt_min
        r_pt = r_pt[r_mask]
        r_eta = r_eta[r_mask]
        r_phi = r_phi[r_mask]
        for extra_name in r_extra:
            r_extra[extra_name] = r_extra[extra_name][r_mask]
        
        # Apply reco isolation if specified
        r_isolated = compute_isolation_awkward(r_eta, r_phi, reco_iso_dr)
#             r_pt = r_pt[r_isolated]
#             r_eta = r_eta[r_isolated]
#             r_phi = r_phi[r_isolated]

        if truth_pt_min is not None:
            r_isolated = r_isolated & (r_pt > reco_pt_min)
        
        # Full vectorized matching
        matched = match_awkward_full(
            r_pt,
            r_eta,
            r_phi,
            t_pt,
            t_eta,
            t_phi,
            dr_max,
            r_isolated,
            t_isolated,
            pt_min,
            extra_vars=r_extra,
        )
        
        results['reco_pt'][reco_prefix] = matched['reco_pt']
        results['reco_eta'][reco_prefix] = matched['reco_eta']
        results['reco_phi'][reco_prefix] = matched['reco_phi']
        results['truth_pt'][reco_prefix] = matched['truth_pt']
        results['truth_eta'][reco_prefix] = matched['truth_eta']
        results['truth_phi'][reco_prefix] = matched['truth_phi']
        results['reco_extra'][reco_prefix] = matched.get('reco_extra', {})
    
    return results


def delta_phi_awkward(phi1, phi2):
    """Awkward array version of delta_phi"""
    dphi = phi1 - phi2
    return ak.where(dphi > np.pi, dphi - 2*np.pi, 
                    ak.where(dphi < -np.pi, dphi + 2*np.pi, dphi))


def compute_isolation_awkward(eta, phi, iso_dr):
    """
    Compute isolation mask for awkward arrays.
    Returns True if object is isolated (no other object within iso_dr).
    Shape: (n_events, n_objects_per_event)
    """
    # Compute pairwise distances within each event
    # Broadcasting: (events, objects, 1) vs (events, 1, objects)
    d_eta = eta[:, :, np.newaxis] - eta[:, np.newaxis, :]
    d_phi = delta_phi_awkward(phi[:, :, np.newaxis], phi[:, np.newaxis, :])
    dr2 = d_eta**2 + d_phi**2
    
    # Check if any OTHER object is too close (dr2 > 0 excludes self)
    has_nearby = ak.any((dr2 < iso_dr**2) & (dr2 > 0), axis=2)
    
    return ~has_nearby

def match_awkward_full(r_pt, r_eta, r_phi, t_pt, t_eta, t_phi, dr_max, riso, tiso, ptmin, k=4, extra_vars=None):
    """
    Full vectorized matching using awkward arrays.
    Non-greedy: each truth object matches to closest reco within dr_max.
    
    Parameters:
    -----------
    riso : awkward array, same shape as r_pt
        If True, the reco jet is marked as isolated and will not be matched
    tiso : awkward array, same shape as t_pt
        If True, the truth jet is marked as isolated and will not be matched
    """
    # ---- pairwise distances (ragged broadcasting) ----
    # shapes: (n_events, n_truth, n_reco)
    d_eta = t_eta[:, :, np.newaxis] - r_eta[:, np.newaxis, :]
    d_phi = delta_phi_awkward(t_phi[:, :, np.newaxis], r_phi[:, np.newaxis, :])
    dr2 = d_eta**2 + d_phi**2

    # ---- apply isolation masks ----
    # riso: (n_events, n_reco) -> (n_events, 1, n_reco)
    # tiso: (n_events, n_truth) -> (n_events, n_truth, 1)
    riso_b = riso[:, np.newaxis, :]
    tiso_b = tiso[:, :, np.newaxis]
    isolated_pair = riso_b & tiso_b   # True where both jets are isolated
    valid = (dr2 < (dr_max**2)) & isolated_pair

    # Mask dr2 for argmin; invalid entries -> +inf
    dr2_masked = ak.where(valid, dr2, np.inf)

    # Closest reco index per truth (per-event ragged indexing)
    closest_reco_idx = ak.argmin(dr2_masked, axis=2)   # shape (n_events, n_truth)
    # has_match True where dr2_masked < inf
    has_match = ak.min(dr2_masked, axis=2) < np.inf

    # ---- extract matched reco values per truth ----
    matched_pt  = r_pt[closest_reco_idx]
    matched_eta = r_eta[closest_reco_idx]
    matched_phi = r_phi[closest_reco_idx]
    
    has_match = has_match & (matched_pt>ptmin)

    # Replace non-matches with NaN
    matched_eta = ak.where(has_match, matched_eta, np.nan)
    matched_phi = ak.where(has_match, matched_phi, np.nan)
    matched_pt = ak.where(has_match, matched_pt, np.nan)

    # ---- identify reco objects used in any match ----
    # matched_indices: sentinel -1 where truth had no match
    matched_indices = ak.where(has_match, closest_reco_idx, -1)

    reco_local_idx = ak.local_index(r_pt, axis=1)   # (n_events, n_reco)
    # For each reco, check if equals any matched index in its event (broadcast compare)
    reco_was_matched = ak.any(reco_local_idx[:, :, np.newaxis] == matched_indices[:, np.newaxis, :], axis=2) # ak.any((n_events, n_reco, n_truth), axis=2) -> (n_events, n_reco)

    # ---- unmatched reco arrays ----
    unmatched_reco_eta = r_eta[(~reco_was_matched) & (r_pt>ptmin)]
    unmatched_reco_phi = r_phi[(~reco_was_matched) & (r_pt>ptmin)]
    unmatched_reco_pt = r_pt[(~reco_was_matched) & (r_pt>ptmin)]
    
    # ---- sort by reco pt ----
    order = ak.argsort(unmatched_reco_pt, axis=1, ascending=False)
    unmatched_reco_eta = unmatched_reco_eta[order]
    unmatched_reco_phi = unmatched_reco_phi[order]
    unmatched_reco_pt = unmatched_reco_pt[order]
    
    # ---- keep top k ----
    unmatched_reco_eta = unmatched_reco_eta[:,:k]
    unmatched_reco_phi = unmatched_reco_phi[:,:k]
    unmatched_reco_pt = unmatched_reco_pt[:,:k]

    # corresponding NaN truth placeholders for unmatched reco
    unmatched_truth_pt = ak.broadcast_arrays(np.nan, unmatched_reco_pt)[0]
    unmatched_truth_eta = ak.broadcast_arrays(np.nan, unmatched_reco_eta)[0]
    unmatched_truth_phi = ak.broadcast_arrays(np.nan, unmatched_reco_phi)[0]

    # ---- concatenate matched truths + unmatched reco-rows ----
    reco_pt_out = ak.concatenate([matched_pt, unmatched_reco_pt], axis=1)
    reco_eta_out = ak.concatenate([matched_eta, unmatched_reco_eta], axis=1)
    reco_phi_out = ak.concatenate([matched_phi, unmatched_reco_phi], axis=1)

    truth_pt_out = ak.concatenate([t_pt, unmatched_truth_pt], axis=1)
    truth_eta_out = ak.concatenate([t_eta, unmatched_truth_eta], axis=1)
    truth_phi_out = ak.concatenate([t_phi, unmatched_truth_phi], axis=1)
    
    #print('n_total',np.histogram(ak.to_numpy(ak.count(reco_pt_out, axis=-1))))
    #print('n_reco',np.histogram(ak.to_numpy(ak.count_nonzero(reco_pt_out>0, axis=-1))))
    #print('n_truth',np.histogram(ak.to_numpy(ak.count_nonzero(truth_pt_out>0, axis=-1))))

    output = {
        'reco_pt': reco_pt_out,
        'reco_eta': reco_eta_out,
        'reco_phi': reco_phi_out,
        'truth_pt': truth_pt_out,
        'truth_eta': truth_eta_out,
        'truth_phi': truth_phi_out,
    }

    if extra_vars:
        reco_extra_out = {}
        for extra_name, r_extra in extra_vars.items():
            matched_extra = r_extra[closest_reco_idx]
            matched_extra = ak.where(has_match, matched_extra, np.nan)
            unmatched_extra = r_extra[(~reco_was_matched) & (r_pt > ptmin)]
            unmatched_extra = unmatched_extra[order]
            unmatched_extra = unmatched_extra[:,:k]
            reco_extra_out[extra_name] = ak.concatenate([matched_extra, unmatched_extra], axis=1)
        output['reco_extra'] = reco_extra_out

    return output


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
    step_size=10000,
):

    if pt_reco_names is None:
        pt_reco_names=["pt"]*len(reco_prefixes)
    
    extra_vars_by_prefix = _normalize_extra_vars(reco_prefixes, extra_vars)

    reco_branches = {}
    for i, reco_prefix in enumerate(reco_prefixes):
        reco_branches[reco_prefix] = [
            f"{reco_prefix}_{pt_reco_names[i]}",
            f"{reco_prefix}_{eta_name}",
            f"{reco_prefix}_{phi_name}",
        ] + [f"{reco_prefix}_{extra_var}" for extra_var in extra_vars_by_prefix[reco_prefix]]

    truth_branches = [
        f"{truth_prefix}_{pt_truth_name}{truth_suffix}",
        f"{truth_prefix}_{eta_name}{truth_suffix}",
        f"{truth_prefix}_{phi_name}{truth_suffix}",
    ]
    branches = truth_branches
    for reco_prefix in reco_prefixes:
        branches = branches + reco_branches[reco_prefix]
    branches = branches + ["weight","gFEX_rho"]

    # Accumulators for awkward arrays
    event_ids = []
    event_weights = []
    event_rhos = []
    reco_pts = {reco_prefix:[] for reco_prefix in reco_prefixes}
    reco_etas = {reco_prefix:[] for reco_prefix in reco_prefixes}
    reco_phis = {reco_prefix:[] for reco_prefix in reco_prefixes}
    truth_pts = {reco_prefix:[] for reco_prefix in reco_prefixes}
    truth_etas = {reco_prefix:[] for reco_prefix in reco_prefixes}
    truth_phis = {reco_prefix:[] for reco_prefix in reco_prefixes}
    reco_extras = {
        reco_prefix: {extra_name: [] for extra_name in extra_vars_by_prefix[reco_prefix]}
        for reco_prefix in reco_prefixes
    }

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
        file_rhos = []

        for chunk in tqdm(it, total=total_chunks, desc=f"{filename}"):
            for name in [reco_branches[r][0] for r in reco_branches] + truth_branches[:1]:
                chunk = ak.with_field(chunk, ak.values_astype(chunk[name] / 1000.0, np.float32), name) # convert to GeV
            for name in [reco_branches[r][1] for r in reco_branches] + truth_branches[1:2]:
                chunk = ak.with_field(chunk, ak.values_astype(chunk[name], np.float32), name)
            for name in [reco_branches[r][2] for r in reco_branches] + truth_branches[2:]:
                chunk = ak.with_field(chunk, ak.values_astype(chunk[name], np.float32), name)
            for name in [
                extra_branch
                for reco_prefix in reco_prefixes
                for extra_branch in reco_branches[reco_prefix][3:]
            ]:
                chunk = ak.with_field(chunk, ak.values_astype(chunk[name], np.float32), name)
            n_events_chunk = len(chunk[reco_branches[reco_prefixes[0]][0]])
            # Process entire chunk at once
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
            )

            # Append results (now as awkward arrays)
            event_ids.extend(range(event_offset, event_offset + n_events_chunk))
            file_weights.extend(chunk["weight"])
            file_rhos.extend(chunk["gFEX_rho"])

            for reco_prefix in reco_prefixes:
                #reco_pts[reco_prefix].extend(ak.to_list(results['reco_pt'][reco_prefix])) # old implementation
                reco_pts[reco_prefix].append(results['reco_pt'][reco_prefix])
                reco_etas[reco_prefix].append(results['reco_eta'][reco_prefix])
                reco_phis[reco_prefix].append(results['reco_phi'][reco_prefix])
                truth_pts[reco_prefix].append(results['truth_pt'][reco_prefix])
                truth_etas[reco_prefix].append(results['truth_eta'][reco_prefix])
                truth_phis[reco_prefix].append(results['truth_phi'][reco_prefix])
                for extra_name, extra_values in results['reco_extra'][reco_prefix].items():
                    reco_extras[reco_prefix][extra_name].append(extra_values)

            event_offset += n_events_chunk
            
            #break

            # Force garbage collection 
            del chunk
            gc.collect()
        

        total_weight = np.sum(file_weights)
        file_weights = [f*weight/total_weight for f in file_weights]
        event_weights.extend(file_weights)
        
        event_rhos.extend(file_rhos)

    for i,f in enumerate(files):
        process_file(f,weights[i])
        
    # Convert everything to an awkward record-of-lists
    return {
        reco_prefix: ak.zip(
            {
                "event": ak.Array(event_ids),
                "weight": ak.Array(event_weights),
                "rho": ak.fill_none(ak.pad_none(event_rhos, 3, axis=-1, clip=True), 0., axis=-1), # TODO: why is this necessary for zprime?
                #"reco_pt": ak.Array(reco_pts[reco_prefix]), # old implementation
                "reco_pt": ak.concatenate(reco_pts[reco_prefix]),
                "reco_eta": ak.concatenate(reco_etas[reco_prefix]),
                "reco_phi": ak.concatenate(reco_phis[reco_prefix]),
                "truth_pt": ak.concatenate(truth_pts[reco_prefix]),
                "truth_eta": ak.concatenate(truth_etas[reco_prefix]),
                "truth_phi": ak.concatenate(truth_phis[reco_prefix]),
                **{
                    f"reco_{extra_name}": ak.concatenate(reco_extras[reco_prefix][extra_name])
                    for extra_name in extra_vars_by_prefix[reco_prefix]
                },
            },
            depth_limit=1,
        )
        for reco_prefix in reco_prefixes
    }


def select_kth(arr, field, sorton, k):
    """
    Memory-optimized single-field version with explicit cleanup.
    """
    # Convert NaN in the sort key to 0
    sort_key = ak.where(
        ak.is_none(arr[sorton]) | np.isnan(arr[sorton]),
        0,
        arr[sorton]
    )
    
    # argsort using the cleaned sort key
    order = ak.argsort(sort_key, axis=1, ascending=False)
    del sort_key  # Immediate cleanup
    
    # reorder the variable of interest
    sorted_var = arr[field][order]
    del order  # Don't need this anymore
    
    # get kth element (slice k-1:k so firsts works)
    kth = ak.firsts(sorted_var[:, k-1:k], axis=1)
    del sorted_var  # Cleanup
    
    # fill missing with 0 and convert to numpy
    result = np.nan_to_num(ak.to_numpy(ak.fill_none(kth, 0)), nan=0.)
    del kth  # Cleanup before return
    
    return result

def select_kths(arr, fields, sorton, k):
    """
    Memory-optimized version that extracts multiple fields in one pass.
    
    Parameters
    ----------
    arr : awkward array
        Input array with nested structure
    fields : list of str
        Field names to extract (e.g., ["reco_pt", "reco_eta", "truth_pt"])
    sorton : str
        Field name to sort on
    k : int
        Which element to select (1-indexed)
    
    Returns
    -------
    dict : {field_name: numpy array}
        Dictionary mapping field names to their kth values
    """
    # Convert NaN in the sort key to 0
    sort_key = ak.where(
        ak.is_none(arr[sorton]) | np.isnan(arr[sorton]),
        0,
        arr[sorton]
    )
    
    # Get sort order ONCE
    order = ak.argsort(sort_key, axis=1, ascending=False)
    del sort_key  # Immediate cleanup
    
    # Extract all requested fields using the same sort order
    results = []
    for field in fields:
        sorted_var = arr[field][order]
        kth = ak.firsts(sorted_var[:, k-1:k], axis=1)
        results.append(
            np.nan_to_num(
                ak.to_numpy(ak.fill_none(kth, 0)), 
                nan=0.
            )
        )
        # Clean up intermediates
        del sorted_var, kth
    
    del order
    return results

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

def compute_pt_threshold(bkg_pairs, target_eff, nobj, correctors=None):
    """
    Compute a reco-pt threshold that yields the requested background efficiency.

    The function finds threshold T such that fraction(background with reco_pt > T)
    is approximately `target_eff`. Returns (threshold, actual_eff).

    Parameters
    ----------
    bkg_pairs : np.ndarray
        Array shape (N, >=1) where column 0 is reco_pt (as in the pipeline).
    target_eff : float
        Target background efficiency in (0,1). Example: 0.01 for 1%.

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
                
    reco_pt,reco_eta = select_kths(bkg_pairs, ["reco_pt","reco_eta"], "reco_pt", nobj)
    w = ak.to_numpy(bkg_pairs["weight"])

    # We want threshold T so that fraction with reco_pt > T == target_eff.
    # That means T is the (1-target_eff) quantile of the reco_pt distribution.
    q = 100.0 * (1.0 - target_eff)

    # Use numpy.percentile which handles small arrays gracefully.
    #threshold = np.percentile(reco_pt, q, weights=w, method="inverted_cdf")
    threshold = weighted_percentile(reco_pt, q, w)

    # Compute actual achieved efficiency (strictly greater than threshold)
    # If you prefer >=, change '>' to '>='.
    actual_eff = np.sum(w[reco_pt > threshold]) / np.sum(w)

    return float(threshold), float(actual_eff)

def compute_rate(bkg_pairs, threshold, nobj, correctors=None, full_rate=31_000.):
    """
    Compute a rate from the given threshold and nobj.

    Returns (rate in kHz, actual_eff).

    Parameters
    ----------
    bkg_pairs : np.ndarray
        Array shape (N, >=1) where column 0 is reco_pt (as in the pipeline).
    threshold : float
        Target background efficiency in (0,1). Example: 0.01 for 1%.

    Returns
    -------
    rate : float
        Rate in kHz. If bkg_pairs is empty, returns np.inf.
    actual_eff : float
        Actual fraction of background events with reco_pt > threshold.
    """
    if threshold < 0.:
        raise ValueError("threshold must be non-negative.")

    if bkg_pairs is None or len(bkg_pairs) == 0:
        return np.inf, 0.0
    
    if correctors is not None:
        for corrector in correctors:
            if callable(corrector):
                corrector(bkg_pairs)
                
    reco_pt,reco_eta = select_kths(bkg_pairs, ["reco_pt","reco_eta"], "reco_pt", nobj)
    w = ak.to_numpy(bkg_pairs["weight"])

    # Compute actual achieved efficiency (strictly greater than threshold)
    # If you prefer >=, change '>' to '>='.
    actual_eff = np.sum(w[reco_pt > threshold]) / np.sum(w)

    return float(full_rate*actual_eff), float(actual_eff)

from scipy.stats import beta

def teff(num, den, weights=None, alpha=0.682689492137086):
    if weights is not None: #not fully implemented
        w = np.asarray(weights)
        num = num * w
        den = den * w
        
    eff = np.where(den == 0, 0.0, num / den)
    low = np.where(
        num == 0, 
        0,
        beta.ppf((1 - alpha) / 2, num, den - num + 1)
    )
    high = np.where(
        num == den,
        1,
        beta.ppf(1 - (1 - alpha) / 2, num + 1, den - num)
    )
    return eff, eff - low, high - eff

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

    eff = np.where(den > 0., num / den, 0.0)
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
        neff = np.where(sumw2_den > 0., (den * den) / sumw2_den, 0.0)
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

def compute_signal_efficiency(
    sig_pairs,
    threshold,
    turnon_bins,
    nobj,
    selector,
    turnon_values=None,
    weights=False,
    inclusive=False,
    correctors=None,
):
    """
    Compute signal efficiency vs truth-pt for a given reco-pt threshold.

    Parameters
    ----------
    sig_pairs : np.ndarray
        Array-like with shape (N, >=2) where:
          column 0 = reco_pt
          column 1 = truth_pt
    threshold : float
        Reco-pt threshold. An event "passes" if reco_pt > threshold by default.
        Use inclusive=True to treat reco_pt >= threshold as passing.
    turnon_bins : array_like
        Bin edges for the turn-on variable (e.g. np.linspace(0,2000,41)).
    weights : bool
        Use weights.
    inclusive : bool
        If True use reco_pt >= threshold (default False uses > threshold).

    Returns
    -------
    centers : np.ndarray
        Bin centers for truth-pt bins.
    efficiency : np.ndarray
        Efficiency in each truth-pt bin (length = len(truth_pt_bins)-1).
    total_counts : np.ndarray
        Total counts (or sum of weights) in each truth-pt bin.
    passed_counts : np.ndarray
        Counts (or sum of weights) in each bin that pass the threshold.
    err : np.ndarray
        Approximate 1-sigma uncertainty on the efficiency in each bin.
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
                

    reco_pt, reco_eta = select_kths(sig_pairs, ["reco_pt","reco_eta"], "reco_pt", nobj)
    if turnon_values is None:
        turnon_values = select_kth(sig_pairs, "truth_pt", "truth_pt", nobj)


    esel = selector(sig_pairs)
    reco_pt = reco_pt[esel]
    reco_eta = reco_eta[esel]
    turnon_values = turnon_values[esel]
    w = ak.to_numpy(sig_pairs["weight"])[esel]

    if not weights:
        # Unweighted histograms
        total_counts, edges = np.histogram(turnon_values, bins=turnon_bins)
        if inclusive:
            passed_mask = (reco_pt >= threshold)
        else:
            passed_mask = (reco_pt > threshold)
        passed_counts, _ = np.histogram(turnon_values[passed_mask], bins=turnon_bins)

        efficiency, errlo, errhi = teff(passed_counts, total_counts)

    else:
        # Weighted
        w_pass = w[passed_mask]

        # sum of weights
        total_w, _ = np.histogram(turnon_values, bins=turnon_bins, weights=w)
        passed_w, _ = np.histogram(turnon_values[passed_mask], bins=turnon_bins, weights=w_pass)

        # sum of squared weights (for effective counts)
        total_w2, _ = np.histogram(turnon_values, bins=turnon_bins, weights=w * w)
        passed_w2, _ = np.histogram(turnon_values[passed_mask], bins=turnon_bins, weights=w_pass * w_pass)

        efficiency, errlo, errhi = teff(
            passed_w,
            total_w,
            sumw2_num=passed_w2,
            sumw2_den=total_w2,
        )

        # convert to float arrays
        total_counts = total_w.astype(float)
        passed_counts = passed_w.astype(float)

    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, efficiency, total_counts, passed_counts, np.stack([errlo, errhi])

def compute_full_efficiency(
    pairs,
    pt_bins,
    nobj,
    selector,
    weights=False,
    correctors=None,
):
    """
    Compute efficiency vs reco-pt threshold.

    Parameters
    ----------
    pairs : np.ndarray
        Array-like with shape (N, >=2) where:
          column 0 = reco_pt
          column 1 = truth_pt
    pt_bins : array_like
        Bin edges for pt (e.g. np.linspace(0,2000,41)).
    weights : bool
        Use weights.

    Returns
    -------
    efficiency : np.ndarray
        Efficiency for each pt threshold (length = len(pt_bins)).
    err : np.ndarray
        Approximate 1-sigma uncertainty on the efficiency in each bin.
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
                
    reco_pt, reco_eta = select_kths(pairs, ["reco_pt","reco_eta"], "reco_pt", nobj)
    truth_pt = select_kth(pairs, "truth_pt", "truth_pt", nobj)

    esel = selector(pairs)
    reco_pt = reco_pt[esel]
    reco_eta = reco_eta[esel]
    truth_pt = truth_pt[esel]
    w = ak.to_numpy(pairs["weight"])[esel]
    
    if not weights:
        total_counts, _ = np.histogram(reco_pt, bins=np.concatenate([pt_bins,[1e9]])) # add overflow bin
        all_count = np.sum(total_counts)
        passed_counts = np.array([np.sum(total_counts[i:]) for i in range(len(pt_bins))])
        for i in range(len(pt_bins)):
            total_counts[i] = all_count

        efficiency, errlo, errhi = teff(passed_counts, total_counts)

    else:
        # Weighted histograms
        if w.shape[0] != reco_pt.shape[0]:
            raise ValueError("weights must have same length as pairs")
            
        # Weighted histograms
        total_w, _ = np.histogram(reco_pt, bins=np.concatenate([pt_bins,[1e9]]), weights=w) # add overflow bin
        total_w2, _ = np.histogram(reco_pt, bins=np.concatenate([pt_bins,[1e9]]), weights=w*w)
        all_w = np.sum(total_w)
        all_w2 = np.sum(total_w2)
        passed_w = np.array([np.sum(total_w[i:]) for i in range(len(pt_bins))])
        passed_w2 = np.array([np.sum(total_w2[i:]) for i in range(len(pt_bins))])
        for i in range(len(pt_bins)):
            total_w[i] = np.sum(all_w)
            total_w2[i] = np.sum(all_w2)

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
    
    #print(debug)
    #print('\t',reco_pt[:10])
    #print('\t',truth_pt[:10])
    #print('\t',response[:10])

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
            
            # 2D Hist: Truth pT vs Response
            plt.hist2d(truth_pt[etamask], response[etamask], 
                       bins=[pt_bins, np.linspace(0., 3., 100)], cmap="viridis")
            
            # Overlay the response
            bin_slice = slice(ie * n_pt, (ie + 1) * n_pt)
            x_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
            plt.errorbar(x_centers, response_centers[bin_slice], 
                         yerr=response_uncs[bin_slice], 
                         fmt='o-', color='r', alpha=0.25, label='Fit' if dofit else 'Median')
            
            plt.axhline(1.0, color='k', linestyle='--')
            plt.ylabel(r"Response (Reco / Truth)")
            plt.xlabel(r"Truth $p_T$ [GeV]")
            plt.title(f"Eta: {eta_bins[ie]:.1f} - {eta_bins[ie+1]:.1f}")
            plt.legend()
            
            # Ensure plotdir exists or handle path
            plot_name = f"{plotdir}/debug_{debug}_eta_{eta_bins[ie]:.1f}_{eta_bins[ie+1]:.1f}.pdf"
            plt.savefig(plot_name, bbox_inches='tight')
            plt.close()
            
            # ------
            # 2D Hist: Reco pT vs Response
            plt.hist2d(reco_pt[etamask], response[etamask], 
                       bins=[pt_bins, np.linspace(0., 3., 100)], cmap="viridis")
            
            # Overlay the response
            bin_slice = slice(ie * n_pt, (ie + 1) * n_pt)
            x_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:]) * response_centers[bin_slice]
            #plt.errorbar(x_centers, response_centers[bin_slice], 
            #             yerr=response_uncs[bin_slice], 
            #             fmt='o-', color='r', alpha=0.25, label='Fit' if dofit else 'Median')
            
            plt.axhline(1.0, color='k', linestyle='--')
            plt.ylabel(r"Response (Reco / Truth)")
            plt.xlabel(r"Reco $p_T$ [GeV]")
            plt.title(f"Eta: {eta_bins[ie]:.1f} - {eta_bins[ie+1]:.1f}")
            plt.legend()
            
            # Ensure plotdir exists or handle path
            plot_name = f"{plotdir}/debug_{debug}_reco_eta_{eta_bins[ie]:.1f}_{eta_bins[ie+1]:.1f}.pdf"
            plt.savefig(plot_name, bbox_inches='tight')
            plt.close()
            
        plt.hist(response-1., bins=np.linspace(-1.5, 1.5, 81), color='r', histtype='step')
        plt.axvline(0., color='k', linestyle='--')
        plt.xlabel(r"Reco - Truth / Truth")
        plt.ylabel(r"Number of objects")

        # Ensure plotdir exists or handle path
        plot_name = f"{plotdir}/debug_{debug}_all.pdf"
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
                ax.hist2d(rho_bin, pt_diff, bins=50, cmap="viridis")
                ax.set_xlabel(r"$\rho$")
                ax.set_ylabel(r"$\Delta p_{T}$ (Reco-Truth)")
                rho_min = np.min(rho_bin)
                rho_max = np.max(rho_bin)
                ax.plot([rho_min, rho_max], 
                       [rho_min*slope+intercept, rho_max*slope+intercept],
                       color='r', linestyle='dashed')
                plt.savefig(f'{plotdir}/debug_{debug}rhofit_eta_{eta_min:.2f}_{eta_max:.2f}.pdf', 
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
        
        # Convert to numpy for correction
        pt_np = ak.to_numpy(pt_flat)
        eta_np = ak.to_numpy(eta_flat)
        rho_np = ak.to_numpy(rho_per_obj)
        
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
        corrected_pt = ak.unflatten(corrected_pt_flat, ak.num(reco_pt))
        del corrected_pt_flat  # Clean up before reassignment
        
        # Modify pairs in-place
        pairs["reco_pt"] = corrected_pt


from scipy.interpolate import interp1d, make_smoothing_spline

def identity_response(x): 
    return np.ones_like(x)
        
class ResponseInterpolator:
    """
    A callable class that interpolates the Response (Reco/Truth)
    Memory-optimized version with explicit cleanup
    """
    def __init__(self, response_centers, response_errors, pt_bins, eta_bins, debug=None):
        self.eta_bins = eta_bins
        self.interpolators = []
        self.debug = debug
        
        # Reshape the 1D response array into (n_eta_bins, n_pt_bins)
        n_pt = len(pt_bins) - 1
        n_eta = len(eta_bins) - 1
        resp_2d = response_centers.reshape(n_eta, n_pt)
        
        # Calculate Truth pT centers
        truth_centers = 0.5 * (pt_bins[1:] + pt_bins[:-1])
        reco_centers = truth_centers[np.newaxis, :] * resp_2d
        
        pt_reco = []
        
        plt.clf()
        for i_eta in range(n_eta):
            R = resp_2d[i_eta]
            
            # Filter out invalid points
            valid = (R > 0) & (~np.isnan(R))
            
            if np.sum(valid) < 3:
                # Fallback: return identity
                self.interpolators.append(identity_response)
            else:
                # Create smoothing spline for R(log(Reco_pT))
                x = np.log(reco_centers[i_eta][valid])
                y = R[valid]
                order = np.argsort(x)
                x = x[order]
                y = y[order]
                s = 1e-4 * len(x)

                f = make_smoothing_spline(x, y, lam=s)
                flin = interp1d(
                    np.log(reco_centers[i_eta][valid]), 
                    R[valid], 
                    kind='linear', 
                    bounds_error=False, 
                    fill_value=(R[valid][0], R[valid][-1])
                )
                self.interpolators.append(f)
                plt.plot(x,
                         f(x),
                         color='C%i'%i_eta, 
                         label=r"$%.2f<\eta<%.2f$"%(eta_bins[i_eta],eta_bins[i_eta+1])
                        )
                plt.scatter(x,y,color='C%i'%i_eta, alpha=0.5)
            
            # Clean up per-iteration arrays
            del R, valid
        plt.xlabel(r'$log(p_{T})$')
        plt.ylabel('R (reco/truth)')
        plt.legend()
        plot_name = f"{plotdir}/debug_{debug}.pdf"
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
        
        pt = ak.to_numpy(ak.flatten(reco_pt))
        eta = ak.to_numpy(ak.flatten(reco_eta))
        
        out_resp_flat = np.ones_like(pt)
        
        # Identify which eta bin each object belongs to
        eta_indices = np.digitize(eta, self.eta_bins) - 1
        eta_indices = np.clip(eta_indices, 0, len(self.interpolators) - 1)
        
        for ie in range(len(self.eta_bins)-1):
            mask = (eta_indices == ie)
            if np.any(mask):
                out_resp_flat[mask] = self.interpolators[ie](np.log(np.array(pt[mask])))
        
        # Clean up numpy intermediates
        del pt, eta, eta_indices
        
        # Unflatten back to original structure
        out_resp = ak.unflatten(out_resp_flat, ak.num(reco_pt))
        del out_resp_flat  # Clean up before division
        
        # Modify pairs in-place
        pairs["reco_pt"] = reco_pt / out_resp
        del out_resp  # Clean up after modification

def apply_dict(func, d, opts):
    outs = [{} for _ in d]
    for k, v in d.items():
        r = func(v, **opts)
        for i,x in enumerate(r):
            outs[i] = x
    return outs

import pickle

def save_corrs(corr_obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(corr_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_corrs(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

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
    
    
        
    for n in range(len(config.nobjs)):
        if debug: 
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
                    bkgv = select_kth(bkg_pairs[reco_prefix], var, "reco_pt" if "reco" in var else "truth_pt", config.nobjs[n])
                    bkgw = bkg_pairs[reco_prefix]["weight"]
                    axb.hist(bkgv,bins=bins,histtype='step',label='Truth' if 'truth' in var else reco_prefix,weights=bkgw)
                    sigv = select_kth(sig_pairs[reco_prefix], var, "reco_pt" if "reco" in var else "truth_pt", config.nobjs[n])
                    axs.hist(sigv,bins=bins,histtype='step',label='Truth' if 'truth' in var else reco_prefix)
                    ax.hist(bkgv,bins=bins,histtype='step',label='bkg : '+('Truth' if 'truth' in var else reco_prefix),weights=bkgw,density=True)
                    ax.hist(sigv,bins=bins,histtype='step',label='sig : '+('Truth' if 'truth' in var else reco_prefix),density=True)
                axb.legend(bbox_to_anchor=(1.05, 1))
                axs.legend(bbox_to_anchor=(1.05, 1))
                ax.legend(bbox_to_anchor=(1.05, 1))
                if 'pt' in var:
                    axb.set_yscale('log')
                    axs.set_yscale('log')
                    ax.set_yscale('log')
                figb.savefig(plotdir+'/debug_bkg_%s%s_nocorr_%s_%i.pdf'%(prefix,var,config.name,config.nobjs[n]), bbox_inches='tight')
                figs.savefig(plotdir+'/debug_sig_%s%s_nocorr_%s_%i.pdf'%(prefix,var,config.name,config.nobjs[n]), bbox_inches='tight')
                fig.savefig(plotdir+'/debug_%s%s_nocorr_%s_%i.pdf'%(prefix,var,config.name,config.nobjs[n]), bbox_inches='tight')
                plt.close(figb)
                plt.close(figs)
                plt.close(fig)
                
            if n==0:
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
                    axb.legend(bbox_to_anchor=(1.05, 1))
                    axs.legend(bbox_to_anchor=(1.05, 1))
                    ax.legend(bbox_to_anchor=(1.05, 1))
                    figb.savefig(plotdir+'/debug_bkg_%s%s_nocorr_%s.pdf'%(prefix,var,config.name), bbox_inches='tight')
                    figs.savefig(plotdir+'/debug_sig_%s%s_nocorr_%s.pdf'%(prefix,var,config.name), bbox_inches='tight')
                    fig.savefig(plotdir+'/debug_%s%s_nocorr_%s.pdf'%(prefix,var,config.name), bbox_inches='tight')
                    plt.close(figb)
                    plt.close(figs)
                    plt.close(fig)
        
    for n in range(len(config.nobjs)):
        for reco_prefix, reco_label in zip(config.reco_prefixes, config.reco_labels):

            if n==0:
                
                # compute uncorrected response
                response_uncorr, resol_uncorr, _ = compute_response(
                    bkg_pairs[reco_prefix], 
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
                        rhosub(bkg_pairs[reco_prefix])
                        rhosub(sig_pairs[reco_prefix])
                    corrector(bkg_pairs[reco_prefix])
                    corrector(sig_pairs[reco_prefix])

                else:
                    # ---- COMPUTE Corrections ----
                    corrs = {}
                    
                    print(f"Starting {reco_prefix}...")
                    # compute area subtraction (if requested)
                    if config.do_rho_sub:
                        rhosub = AreaSubtractor(
                            bkg_pairs[reco_prefix], 
                            config.truth_eta_bins,
                            debug=prefix+reco_prefix+"_"
                        )
                        
                        rhosub(bkg_pairs[reco_prefix])
                        rhosub(sig_pairs[reco_prefix])
                        
                        # recompute response corrections for debug
                        response_acorr, resol_acorr, _ = compute_response(
                            bkg_pairs[reco_prefix], 
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
                        debug=f"{config.name}_{reco_prefix}_responseinterp"
                    )
                                    
                    #apply corrections
                    corrector(bkg_pairs[reco_prefix])
                    corrector(sig_pairs[reco_prefix])
                    
                    corrs[reco_prefix] = (rhosub,corrector)
                    # Save once
                    print(f"Saving corrections to {corr_filename}")
                    save_corrs(corrs, corr_filename)
                
                # recompute response corrections
                response_corr, resol_corr, _ = compute_response(
                    bkg_pairs[reco_prefix], 
                    config.truth_pt_bins, 
                    config.truth_eta_bins,
                    config.pt_min,
                    debug=f"{config.name}_{reco_prefix}_corr" if corr_cache=="" else None
                )
                print(f"Finished {reco_prefix} response...")
                
                del corrector
                del rhosub
                gc.collect()
                
            # Compute threshold for fixed background efficiency
            rate_eff = config.rates[n]/31_000.
            threshold,actual_eff = compute_pt_threshold(bkg_pairs[reco_prefix], rate_eff, config.nobjs[n]) #in kHz
            print(f"For {reco_prefix}, n={config.nobjs[n]}, target rate efficiency of {rate_eff:.6f}, threshold of {threshold:.6f} gives actual rate efficiency of {actual_eff:.6f}")

            for turnon_fn, turnon_label, turnon_bins in zip(
                config.turnon_vars,
                config.turnon_var_labels,
                config.turnon_bins,
            ):
                turnon_values = turnon_fn(sig_pairs[reco_prefix], config.nobjs[n])
                for s in range(len(config.sels)):
                    # Signal efficiency vs turn-on variable
                    centers, eff, _,_, err = compute_signal_efficiency(
                        sig_pairs[reco_prefix],
                        threshold,
                        turnon_bins,
                        config.nobjs[n],
                        config.sels[s],
                        turnon_values=turnon_values,
                    )
                    
                    full_sig_eff, full_sig_err = compute_full_efficiency(sig_pairs[reco_prefix], config.truth_pt_bins, config.nobjs[n], config.sels[s], weights=False)
                    full_bkg_eff, full_bkg_err = compute_full_efficiency(bkg_pairs[reco_prefix], config.truth_pt_bins, config.nobjs[n], config.sels[s], weights=True)

                    results.append(
                        RunResult(
                            name=config.name,
                            sel_label=config.sel_labels[s],
                            reco=reco_prefix,
                            reco_label=reco_label,
                            nobj=config.nobjs[n],
                            fixrate=True,
                            threshold=threshold,
                            rate=config.rates[n],
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
                            turnon_label=turnon_label,
                            turnon_bins=turnon_bins,
                        )
                    )
                
                
            for threshold in config.triggers[n]:
                rate, actual_eff = compute_rate(bkg_pairs[reco_prefix], threshold, config.nobjs[n])
                print(f"For {reco_prefix}, n={config.nobjs[n]}, trigger threshold of {threshold:.1f} gives actual rate efficiency of {actual_eff:.6f} (rate {rate:.6f} kHz)")
                for turnon_fn, turnon_label, turnon_bins in zip(
                    config.turnon_vars,
                    config.turnon_var_labels,
                    config.turnon_bins,
                ):
                    turnon_values = turnon_fn(sig_pairs[reco_prefix], config.nobjs[n])
                    for s in range(len(config.sels)):
                        # Signal efficiency vs turn-on variable
                        centers, eff, _,_, err = compute_signal_efficiency(
                            sig_pairs[reco_prefix],
                            threshold,
                            turnon_bins,
                            config.nobjs[n],
                            config.sels[s],
                            turnon_values=turnon_values,
                        )

                        results.append(
                            RunResult(
                                name=config.name,
                                sel_label=config.sel_labels[s],
                                reco=reco_prefix,
                                reco_label=reco_label,
                                nobj=config.nobjs[n],
                                fixrate=False,
                                threshold=threshold,
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
                                turnon_label=turnon_label,
                                turnon_bins=turnon_bins,
                            )
                        )
                
        if debug: 
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
                    bkgv = select_kth(bkg_pairs[reco_prefix], var, "reco_pt" if "reco" in var else "truth_pt", config.nobjs[n])
                    bkgw = bkg_pairs[reco_prefix]["weight"]
                    sigv = select_kth(sig_pairs[reco_prefix], var, "reco_pt" if "reco" in var else "truth_pt", config.nobjs[n])
                    axb.hist(bkgv,bins=bins,histtype='step',label='Truth' if 'truth' in var else reco_prefix,weights=bkgw)
                    axs.hist(sigv,bins=bins,histtype='step',label='Truth' if 'truth' in var else reco_prefix)
                    ax.hist(bkgv,bins=bins,histtype='step',label='bkg : '+('Truth' if 'truth' in var else reco_prefix),weights=bkgw,density=True)
                    ax.hist(sigv,bins=bins,histtype='step',label='sig : '+('Truth' if 'truth' in var else reco_prefix),density=True)
                axb.legend(bbox_to_anchor=(1.05, 1))
                axs.legend(bbox_to_anchor=(1.05, 1))
                ax.legend(bbox_to_anchor=(1.05, 1))
                if 'pt' in var:
                    axb.set_yscale('log')
                    axs.set_yscale('log')
                    ax.set_yscale('log')
                figb.savefig(plotdir+'/debug_bkg_%s%s_%s_%i.pdf'%(prefix,var,config.name,config.nobjs[n]), bbox_inches='tight')
                figs.savefig(plotdir+'/debug_sig_%s%s_%s_%i.pdf'%(prefix,var,config.name,config.nobjs[n]), bbox_inches='tight')
                fig.savefig(plotdir+'/debug_%s%s_%s_%i.pdf'%(prefix,var,config.name,config.nobjs[n]), bbox_inches='tight')
                plt.close(figb)
                plt.close(figs)
                plt.close(fig)
            
            if n==0:
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
                    axb.legend(bbox_to_anchor=(1.05, 1))
                    axs.legend(bbox_to_anchor=(1.05, 1))
                    ax.legend(bbox_to_anchor=(1.05, 1))
                    figb.savefig(plotdir+'/debug_bkg_%s%s_%s.pdf'%(prefix,var,config.name), bbox_inches='tight')
                    figs.savefig(plotdir+'/debug_sig_%s%s_%s.pdf'%(prefix,var,config.name), bbox_inches='tight')
                    fig.savefig(plotdir+'/debug_%s%s_%s.pdf'%(prefix,var,config.name), bbox_inches='tight')
                    plt.close(figb)
                    plt.close(figs)
                    plt.close(fig)

    
    return results

def save_run_result(result: RunResult, path):
    np.savez(
        path,
        name=result.name,
        reco=result.reco,
        reco_label=result.reco_label,
        nobj=result.nobj,
        sel_label=result.sel_label,
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
        turnon_label=result.turnon_label,
        turnon_bins=result.turnon_bins,
    )

def load_run_result(path):
    data = np.load(path, allow_pickle=True)
    turnon_label = data["turnon_label"].item() if "turnon_label" in data else "truth_pt"
    turnon_bins = data["turnon_bins"] if "turnon_bins" in data else data["truth_pt_bins"]
    return RunResult(
        name=data["name"].item(),
        reco=data["reco"].item(),
        reco_label=data["reco_label"].item() if "reco_label" in data else data["reco"].item(),
        nobj=data["nobj"].item(),
        sel_label=data["sel_label"].item(),
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
        1:"Leading",
        2:"Subleading",
        3:"Third Leading",
        4:"Fourth Leading"
    }   
    
    params = {}
    for i,r in enumerate(results):
        turnon_bins = getattr(r, "turnon_bins", r.truth_pt_bins)
        turnon_label = getattr(r, "turnon_label", "Truth $p_T$ [GeV]")
        turnon_centers = 0.5*(turnon_bins[:-1]+turnon_bins[1:])
        xmask = (turnon_centers < xmax) if xmax>0. else np.ones(len(turnon_centers))
        params[i],_ = fit_logistic(turnon_centers[xmask], r.signal_efficiency[xmask], np.mean(r.signal_efficiency_error,axis=0)[xmask])
        label_full = result_reco_label(r)+", "+r.name+r' [$p_T$>'+('%.1f'%r.threshold)+'] ($\\sigma$='+('%.2f'%(1./params[i][1]))+', $p_T^{98\\%}$='+('%.2f'%(params[i][2]+np.log(49)/params[i][1]))+')'
        plt.errorbar(turnon_centers[xmask], r.signal_efficiency[xmask], None if noerr else r.signal_efficiency_error[:,xmask], marker='s', label=label_full, color='C%i'%i, capsize=3, capthick=2, linestyle='none', mfc='none', alpha=0.5, markersize=4)
        plt.plot(turnon_centers[xmask], logistic_function(turnon_centers[xmask],*params[i]), color='C%i'%i, linestyle='dashed')

    plt.xlabel(r"%s %s"%(numtext[nobj], turnon_label))
    plt.ylabel("Signal efficiency")
    plt.legend(bbox_to_anchor=(1.05, 1),title=titletxt)
    plt.grid(True)
    plt.savefig(plotdir+'/efficiency%s.pdf'%suffix, bbox_inches='tight')

    plt.clf()

    for i,r in enumerate(results):
        turnon_bins = getattr(r, "turnon_bins", r.truth_pt_bins)
        turnon_centers = 0.5*(turnon_bins[:-1]+turnon_bins[1:])
        shiftmask = turnon_centers < (params[i][2]+3.*np.log(49)/params[i][1]) # 2x the shift from 50 to 98 to make sure its visible
        params_adj,_ = fit_logistic(turnon_centers[shiftmask]-(params[i][2]+np.log(49)/params[i][1]), r.signal_efficiency[shiftmask], np.mean(r.signal_efficiency_error,axis=0)[shiftmask])
        label_full = result_reco_label(r)+", "+r.name+r' [$p_T$>'+('%.1f'%r.threshold)+'] ($\\hat{\\sigma}$='+('%.2f'%(1./params_adj[1]))+', $\\hat{p}_T^{98\\%}$='+('%.2f'%(params_adj[2]+np.log(49)/params_adj[1]))+')'
        plt.errorbar(turnon_centers[shiftmask]-(params[i][2]+np.log(49)/params[i][1]), r.signal_efficiency[shiftmask], None if noerr else r.signal_efficiency_error[:,shiftmask], marker='s', label=label_full, color='C%i'%i, capsize=3, capthick=2, linestyle='none', mfc='none', alpha=0.5, markersize=4)
        plt.plot(turnon_centers[shiftmask]-(params[i][2]+np.log(49)/params[i][1]), logistic_function(turnon_centers[shiftmask]-(params[i][2]+np.log(49)/params[i][1]),*params_adj), color='C%i'%i, linestyle='dashed')

    plt.xlabel(r"%s Truth $\Delta p_T$ [GeV]"%numtext[nobj])
    plt.ylabel("Signal efficiency")
    plt.legend(bbox_to_anchor=(1.05, 1),title=titletxt)
    plt.grid(True)
    plt.savefig(plotdir+'/efficiency%s_corr.pdf'%suffix, bbox_inches='tight')

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
                         marker='o', color='C%i'%i, capsize=3, capthick=2, linestyle='none', alpha=0.5, markersize=4,
                         label=result_reco_label(r)+", "+r.name)

        plt.ylabel(r"%sResponse (Truth $p_T$ / Reco $p_T$)"%("Corrected " if corr else ""))
        plt.xlabel(r"Truth $p_T$ [GeV]")
        plt.title(r"$%.2f<\eta<%.2f$"%(r.truth_eta_bins[ie],r.truth_eta_bins[ie+1]))
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        plt.savefig(plotdir+'/'+prefix+'response'+("_corr" if corr else "")+'_eta_%.2f_%.2f.pdf'%(r.truth_eta_bins[ie],r.truth_eta_bins[ie+1]), bbox_inches='tight')

        plt.clf()

        for i,r in enumerate(results):
            pt_centers = 0.5*(r.truth_pt_bins[:-1]+r.truth_pt_bins[1:])
            plt.plot(pt_centers+(pt_widths)*((0.5*nres)-float(i)), 
                         (r.resol_corr if corr else r.resol_uncorr)[ie*len(pt_centers):(ie+1)*len(pt_centers)], 
                         marker='o', color='C%i'%i, linestyle='none', alpha=0.5, markersize=4,
                         label=result_reco_label(r)+", "+r.name)

        plt.ylabel(r"%sResolution"%("Corrected " if corr else ""))
        plt.xlabel(r"Truth $p_T$ [GeV]")
        plt.title(r"$%.2f<\eta<%.2f$"%(r.truth_eta_bins[ie],r.truth_eta_bins[ie+1]))
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.grid(True)
        plt.savefig(plotdir+'/'+prefix+'resol'+("_corr" if corr else "")+'_eta_%.2f_%.2f.pdf'%(r.truth_eta_bins[ie],r.truth_eta_bins[ie+1]), bbox_inches='tight')

def overlay_full_effs(results, suffix="", nobj=1, xmax=300.):
    plt.clf()

    numtext = {
        1:"Leading",
        2:"Subleading",
        3:"Third Leading",
        4:"Fourth Leading"
    }   
    
    for i,r in enumerate(results):
        xmask = r.truth_pt_bins<xmax
        plt.errorbar(r.truth_pt_bins[xmask],
                     r.full_sig_efficiency[xmask],
                     yerr=r.full_sig_efficiency_error[:,xmask],
                     marker='o', color='C%i'%i, linestyle='none', alpha=0.5, markersize=4,
                     label=result_reco_label(r))
    plt.ylabel(r"Signal efficiency")
    plt.xlabel(r"%s $p_T$ [GeV]"%numtext[nobj])
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.savefig(plotdir+'/efficiency_full_signal%s.pdf'%(suffix), bbox_inches='tight')
    
    plt.clf()
    
    for i,r in enumerate(results):
        plt.errorbar(r.truth_pt_bins[xmask],
                     r.full_bkg_efficiency[xmask]*31_000.,
                     yerr=r.full_bkg_efficiency_error[:,xmask]*31_000.,
                     marker='o', color='C%i'%i, linestyle='none', alpha=0.5, markersize=4,
                     label=result_reco_label(r))
    plt.ylabel(r"Background rate [kHz]")
    plt.yscale('log')
    plt.xlabel(r"%s $p_T$ [GeV]"%numtext[nobj])
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.savefig(plotdir+'/efficiency_full_background%s.pdf'%(suffix), bbox_inches='tight')
    
    plt.clf()
    
    for i,r in enumerate(results):
        plt.errorbar(r.full_sig_efficiency,
                     r.full_bkg_efficiency*31_000.,
                     xerr=r.full_sig_efficiency_error,
                     yerr=r.full_bkg_efficiency_error*31_000.,
                     marker='o', color='C%i'%i, linestyle='none', alpha=0.5, markersize=4,
                     label=result_reco_label(r))
    plt.ylabel(r"Background rate [kHz]")
    plt.yscale('log')
    plt.xlabel(r"Signal efficiency")
    plt.title(numtext[nobj]+" "+results[0].name)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True)
    plt.savefig(plotdir+'/efficiency_full_combined%s.pdf'%(suffix), bbox_inches='tight')

# selectors
def null_selector(pairs):
    #print('Null selector fraction: 1.')
    return np.ones(len(pairs),dtype=bool)

def boosted_truth_selector(pairs, dr_threshold=0.7, debug=0, chunk_size=10000):
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
    debug : int
        Number of events to print debug info for
    chunk_size : int
        Number of events to process at once (tune based on available memory)
        
    Returns
    -------
    event_sel : np.ndarray
        Boolean array indicating which events have close jet pairs
    """
    truth_eta = pairs["truth_eta"]
    truth_phi = pairs["truth_phi"]
    
    n_events = len(truth_eta)
    event_sel = np.zeros(n_events, dtype=bool)
    
    # Process in chunks to limit memory usage
    for chunk_start in range(0, n_events, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_events)
        
        # Extract chunk
        eta_chunk = truth_eta[chunk_start:chunk_end]
        phi_chunk = truth_phi[chunk_start:chunk_end]
        
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
        del eta_chunk, phi_chunk, delta_r, events_with_close_pair
    
    return event_sel

def hh_mass_window_selector(pairs, m_min=75., m_max=175., coll='reco', debug=0, chunk_size=10000):
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
    mass = ak.sqrt(m2)

    abs_deta = ak.abs(deta)
    max_idx = ak.argmax(abs_deta, axis=1, keepdims=True)
    mass_max = ak.flatten(mass[max_idx], axis=1)

    return ak.fill_none(mass_max, np.nan)

base_dir = '/eos/home-d/drankin/GEPEnc/'
