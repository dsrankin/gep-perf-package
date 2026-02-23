# gep-perf

This package is designed to allow for simple, configurable, studies of different triggers. 

## Install (editable)

```bash
pip install -e .
```

## Run

```bash
gep-perf run configs/jet_example.yaml --plotdir perf_plots --resdir perf_results
```

Outputs:
- `.npz` result files in `--resdir`
- plots in `--plotdir`

## YAML configuration

Top-level keys correspond to the original `RunConfig` dataclass fields, except selections, which are expressed as `selectors`. You can also specify a single `rate_selector` that is used for background rejection/rate and for the signal-efficiency numerator only (not the denominator).

Example:

```yaml
name: Jet
signal_files: [".../outputGEPNtuple.root"]
background_files: [".../outputGEPNtuple.root", "..."]
background_weights: [76.66, 3.66, 0, 0, 0, 0]
reco_prefixes: ["AntiKt4GepCellsE6LSB40G4TowerAlgJets", "..."]
reco_labels: ["GEP E6 LSB40", "..."]  # optional display labels for plots
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
  AntiKt4GepCellsE6LSB40G4TowerAlgJets: ["em_frac", "timing"]
  L1_jFexSRJetRoI: ["quality"]
```

### Supported selector names

- `null_selector`
- `boosted_truth_selector` (kwargs: `dr_threshold`, `debug`, `chunk_size`)
- `hh_mass_window_selector`
- `eratio_selector` (kwargs: `threshold`)

To add more, extend `gep_perf.config.SELECTORS`.

When `extra_vars` contains multiple variants of the same variable for one reco prefix (for example `eRatio_LSB40SIG2`, `eRatio_LSB80SIG2`), the loader automatically expands this into multiple logical reco collections (`<prefix>_LSB40SIG2`, `<prefix>_LSB80SIG2`, etc.). The original collection is also kept without that split extra variable. Each expanded collection gets a single logical extra variable name (`eRatio`) and points to the corresponding source branch.
