# gep-perf

This package is designed to allow for simple, configurable, studies of different triggers. 

## Install (editable)

```bash
pip install -e .
```

### lxplus (no install)

On lxplus, or any EL9 machine with CVMFS, nothing needs installing. Source the
setup script from the repository root in each new shell:

```bash
source setup_lxplus.sh
```

It sets up the LCG view (default `LCG_110a`, `x86_64-el9-gcc14-opt`), which
provides every dependency including the XRootD bindings for `root://` files,
and puts this checkout's `gep-perf` command on your `PATH`. To use a different
view, set `LCG_VIEW` to its `setup.sh` before sourcing.

## Run

```bash
gep-perf run configs/jet_example.yaml --plotdir perf_plots --resdir perf_results
gep-perf run configs/tau_example.yaml --plotdir perf_plots --resdir perf_results
gep-perf run configs/ele_example.yaml --plotdir perf_plots --resdir perf_results
gep-perf run configs/met_example.yaml --plotdir perf_plots --resdir perf_results
# Restrict a run to only SK/EtaSK/other collections from the same config:
gep-perf run configs/jet_example.yaml --collection-sets sk
gep-perf run configs/jet_example.yaml --collection-sets etask
gep-perf run configs/jet_example.yaml --collection-sets other
```

Outputs:
- `.npz` result files in `--resdir`
- plots in `--plotdir`

The checked-in example configurations produce both correction modes. After
running them, `./make_plots.sh <object-type>` reads both result-file variants
and writes the comparison plots beneath `perf_plots/corrected/` and
`perf_plots/uncorrected/`. `./runplots.sh <object-type> [...]` provides the
same behavior and also creates the plot archives. Response and resolution
plots are mode-independent and are therefore produced only in the corrected
plot tree.

## YAML configuration

Top-level keys correspond to the original `RunConfig` dataclass fields, except selections, which are expressed as `selectors`. You can also specify a single `rate_selector` that is used for background rejection/rate and for the signal-efficiency numerator only (not the denominator).

Example:

```yaml
name: Jet
signal_files: [".../outputGEPNtuple.root"]
background_files: [".../outputGEPNtuple.root", "..."]
background_weights: [76.66, 3.66, 0, 0, 0, 0]
reco_prefixes: ["AntiKt4GEPCellsTowerAlgJets", "..."]
reco_labels: ["AK4 GEP CellTower", "..."]  # optional display labels for plots
truth_prefix: AntiKt4TruthJets
truth_suffix: ""
match_dict:
  pt_reco_names: ["pt","pt","pt","pt","et","pt","pt","pt"]

nobjs: [1,2,3,4]

selectors:
  - name: null_selector
    label: ""
  - name: boosted_truth_selector
    label: "_boosted"
    kwargs: {dr_threshold: 0.7}
  - name: hh_mass_window_selector
    label: "_hhmass"

# optional selector used for background rejection/rate and signal numerator only
rate_selector:
  name: eratio_selector
  label: "_eratio"
  kwargs: {threshold: 0.65}

truth_pt_bins: [20, 22, 24, ...]
truth_eta_bins: [-4.9, -3.2, ...]
do_rho_sub: true
# Produce both calibrated and raw-pt/energy performance in the same run.
# The default is [corrected] for compatibility with existing configurations.
correction_modes: [corrected, uncorrected]
rates: [50, 50, 75, 100]
triggers: [[60,100],[50,60],[50,90],[40,50]]

# optional overrides
tree: ntuple
dr_max: 0.2
reco_pt_min: 5.0
truth_pt_min: 20.0
pt_min: 5.0
reco_iso_dr: 0.4
truth_iso_dr: 0.6
extra_vars:
  AntiKt4GEPCellsTowerAlgJets: ["em_frac", "timing"]
  L1_jFexSRJetRoISim: ["quality"]

# optional per-collection smoothing-spline lambda (default is 1e-5)
spline_lambdas:
  AntiKt4GEPCellsTowerAlgJets: 2.0e-5
  L1_jFexSRJetRoISim: 5.0e-6
```

### Input samples

Each entry in `signal_files` and `background_files` is one sample: either a
single file, or a list of files that together make up the sample (e.g. the
files of one dataset). Each background sample has one entry in
`background_weights`. Event weights are normalized per sample, so all events of
a sample sum to its weight however many files it is split across.

```yaml
background_files:
- /path/to/jz0/outputGEPNtuple.root              # one-file sample
- [/path/to/jz1/file1.root, /path/to/jz1/file2.root]  # multi-file sample
background_weights: [76.66, 3.66]
```

Files can be local paths or remote XRootD URLs (`root://host//path/file.root`).
This needs the XRootD Python bindings (`fsspec_xrootd`, `xrootd`). On lxplus,
`source setup_lxplus.sh` provides them (see below); elsewhere install them with
`pip install fsspec-xrootd xrootd`. Grid storage also needs a valid proxy
(`voms-proxy-init -voms atlas`).

### Rucio datasets

A sample can also be a Rucio dataset or container, written
`rucio://<scope>:<name>`. When the config is loaded, each such entry is
replaced by the `root://` URLs of all its files and treated as one sample, so
its weight applies to the whole dataset. It can also appear inside a
multi-file sample list.

```yaml
signal_files:
- rucio://user.drankin:user.drankin.tgp_base_zvvhbb_sep25_EXT0
background_files:
- rucio://user.drankin:user.drankin.tgp_base_jz0_sep25_EXT0
- rucio://user.drankin:user.drankin.tgp_base_jz1_sep25_EXT0
background_weights: [76.66, 3.66]
# optional: sites to read from, in order of preference
rucio_rses: [CERN-PROD_DATADISK]
```

Without `rucio_rses`, each file is read from its best-priority disk replica
(tape replicas are never used). The lookup talks to the ATLAS Rucio servers
directly with your grid proxy, so it needs no Rucio client or `lsetup rucio`;
it works in the `setup_lxplus.sh` environment. It needs:

- a grid proxy: `voms-proxy-init -voms atlas` (or set `X509_USER_PROXY`)
- `RUCIO_ACCOUNT`, only if your certificate maps to more than one Rucio account
- `RUCIO_HOST` / `RUCIO_AUTH_HOST`, only to use servers other than ATLAS's

### Corrected and uncorrected results

Set `correction_modes` to any ordered combination of `corrected` and
`uncorrected`. Corrections are fitted or loaded once per reconstruction
collection, while the efficiency, threshold, and rate calculations are run
independently against corrected and original object pt/energy. Corrected files
retain the existing filename; uncorrected files add `_uncorrected` before the
`.npz` extension, so both sets can be produced without overwriting each other.
The correction mode is also stored in each result file as `correction_mode`.

### Supported selector names

For MET studies, enable `match_dict.met_mode: true`. In this mode the code builds truth MET from vector truth neutrinos (`truth_neu_pt/eta/phi`) and treats each reco MET collection as a single object per event (using `<prefix>_et`, `<prefix>_ex`, and `<prefix>_ey`). See `configs/met_example.yaml` for a complete example with both fixed-rate and fixed-threshold trigger definitions.


- `null_selector`
- `boosted_truth_selector` (kwargs: `dr_threshold`, `truth_pt_threshold`, `debug`, `chunk_size`; default `truth_pt_threshold: 40.0`)
- `hh_mass_window_selector`
- `eratio_selector` (kwargs: `threshold`)

To add more, extend `gep_perf.config.SELECTORS`.

When `extra_vars` contains multiple variants of the same variable for one reco prefix (for example `eRatio_variantA`, `eRatio_variantB`), the loader automatically expands this into multiple logical reco collections (`<prefix>_variantA`, `<prefix>_variantB`, etc.). The original collection is also kept without that split extra variable. Each expanded collection gets a single logical extra variable name (`eRatio`) and points to the corresponding source branch. This is for a production that reconstructs several variants of the same object under one branch naming scheme (each variant's extra variable suffixed accordingly); none of the current configs need it, since each reco collection is now just one branch.

If `spline_lambdas` is provided, keys are per reco collection from `reco_prefixes`; missing entries use the default `1e-5`. For auto-expanded reco collections from `extra_vars`, the configured lambda on the source reco prefix is inherited.
