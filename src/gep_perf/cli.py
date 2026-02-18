
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_run_config
from . import core

def run_from_yaml(config_path: str | Path, plotdir: str | None = None, resdir: str | None = None,
                 prefix: str = "", debug: bool = False, corr_cache: str = ""):
    if plotdir is not None:
        core.plotdir = plotdir
        os.makedirs(core.plotdir, exist_ok=True)
    if resdir is not None:
        core.resdir = resdir
        os.makedirs(core.resdir, exist_ok=True)

    cfg = load_run_config(config_path)
    results = core.process_run(cfg, debug=debug, prefix=prefix, corr_cache=corr_cache)

    # Save one file per (reco, nobj, selection) result
    for r in results:
        suffix = "rate"+str(int(r.rate))  if r.fixrate else "pt"+str(int(r.threshold))
        out = Path(core.resdir) / f"{r.name}{r.turnon_var}_{r.reco}_n{r.nobj}{r.sel_label}_{suffix}.npz"
        core.save_run_result(r, str(out))

    return results


def load_from_yaml(res_paths: list[str], plotdir: str | None = None, name: str = "", plot_label: str = "", plot_text: str = "", 
        nobj: int = 1, xmax: float = 200., noerr: bool = False, do_full_eff: bool = False, do_resp_resol: bool = False):

    if plotdir is not None:
        core.plotdir = plotdir
        os.makedirs(core.plotdir, exist_ok=True)

    results = []
    for res_path in res_paths:
        results.append(core.load_run_result(res_path))
               
    # Convenience overlays
    core.overlay_efficiency(results, f"{plot_label}_{name}_n{nobj}", plot_text, nobj, xmax=xmax, noerr=noerr)
    if do_full_eff:
        core.overlay_full_effs(results, suffix=f"_{name}{nobj}", nobj=nobj, xmax=xmax)
    if do_resp_resol:
        core.overlay_resp_resol(results, prefix=f"{name}_")
        core.overlay_resp_resol(results, corr=True, prefix=f"{name}_")

    return results


def main(argv=None):
    p = argparse.ArgumentParser(prog="gep-perf", description="Run GEP performance processing from a YAML config.")
    sub = p.add_subparsers(dest="cmd", required=True)

    prun = sub.add_parser("run", help="Run a single configuration YAML.")
    prun.add_argument("config", help="Path to YAML config.")
    prun.add_argument("--plotdir", default=None, help="Directory for plots (default: perf_plots).")
    prun.add_argument("--resdir", default=None, help="Directory for results (default: perf_results).")
    prun.add_argument("--prefix", default="", help="Prefix for output plot filenames.")
    prun.add_argument("--corr-cache", default="", help="Directory for cached correction fits.")
    prun.add_argument("--only-plot", action="store_true", help="Skip results computation, just load and plot.")
    prun.add_argument("--debug", action="store_true", help="Enable verbose debugging / extra plots.")

    pplot = sub.add_parser("plot", help="Make final efficiency and rate comparison plots.")
    pplot.add_argument("respaths", nargs='+', help="List of results to combine.")
    pplot.add_argument("--plotdir", default=None, help="Directory for plots (default: perf_plots).")
    pplot.add_argument("--name", default=None, help="Additional name for output plot files (default: jet).")
    pplot.add_argument("--plotlabel", default="", help="Plot label.")
    pplot.add_argument("--plottext", default="", help="Plot text.")
    pplot.add_argument("--nobj", default=1, type=int, help="Number of objects.")
    pplot.add_argument("--xmax", default=200., type=float, help="Maximum x value for plot..")
    pplot.add_argument("--noerr", action="store_true", help="Do not plot errors on efficiency plots.")
    pplot.add_argument("--dofulleff", action="store_true", help="Make full efficiency plots.")
    pplot.add_argument("--dorespresol", action="store_true", help="Make response and resolution plots.")

    args = p.parse_args(argv)

    if args.cmd == "run":
        run_from_yaml(args.config, plotdir=args.plotdir, resdir=args.resdir, prefix=args.prefix,
                      debug=args.debug, corr_cache=args.corr_cache)

    if args.cmd == "plot":
        load_from_yaml(args.respaths, plotdir=args.plotdir, name=args.name, plot_label=args.plotlabel, plot_text=args.plottext, 
                nobj=args.nobj, xmax=args.xmax, noerr=args.noerr, do_full_eff=args.dofulleff, do_resp_resol=args.dorespresol)
                            #cfg.sel_labels[s]+("_fixrate" if tr==0 else "_pt%i"%cfg.triggers[n][tr-1])+"_jet%i"%(n+1),
                            #("Fixed Rate: %i kHz"%cfg.rates[n]) if tr==0 else ("%iJ%i"%(cfg.nobjs[n],cfg.triggers[n][tr-1])),


if __name__ == "__main__":
    main()
