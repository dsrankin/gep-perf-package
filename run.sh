#!/bin/bash

OBJECT="$1"
shift

for TYPE in "$@"; do
  echo "$OBJECT $TYPE"

  [ "$OBJECT" = "jet" ] || [ "$OBJECT" = "all" ] && {
    gep-perf run configs/jet_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
    gep-perf run configs/jet_tthad_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
  }

  [ "$OBJECT" = "fatjet" ] || [ "$OBJECT" = "all" ] && {
    gep-perf run configs/jet_larger_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
  }

  [ "$OBJECT" = "tau" ] || [ "$OBJECT" = "all" ] && {
    gep-perf run configs/tau_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
  }

  [ "$OBJECT" = "pho" ] || [ "$OBJECT" = "all" ] && {
    gep-perf run configs/pho_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
  }

  [ "$OBJECT" = "ele" ] || [ "$OBJECT" = "all" ] && {
    gep-perf run configs/ele_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
  }

  [ "$OBJECT" = "met" ] || [ "$OBJECT" = "all" ] && {
    gep-perf run configs/met_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
  }
done
