#!/bin/bash
#
# Set up gep-perf on lxplus (or any EL9 machine with CVMFS) without
# installing anything: the LCG view provides every dependency, including the
# XRootD bindings (xrootd, fsspec_xrootd) needed to read root:// files, and
# this checkout is used directly from source.
#
# Usage (source it, don't execute it):
#   source setup_lxplus.sh
# Override the view with LCG_VIEW=/cvmfs/.../setup.sh source setup_lxplus.sh

LCG_VIEW="${LCG_VIEW:-/cvmfs/sft.cern.ch/lcg/views/LCG_110a/x86_64-el9-gcc14-opt/setup.sh}"

if [[ ! -f "$LCG_VIEW" ]]; then
    echo "setup_lxplus.sh: LCG view not found: $LCG_VIEW" >&2
    echo "  (needs CVMFS; set LCG_VIEW to another view's setup.sh)" >&2
    return 1 2>/dev/null || exit 1
fi
source "$LCG_VIEW"

GEP_PERF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${GEP_PERF_DIR}/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="${GEP_PERF_DIR}/bin:${PATH}"

echo "gep-perf: using $(python3 --version) from ${LCG_VIEW%/setup.sh}"
