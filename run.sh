#!/bin/bash

OBJECT="$1"
shift

for TYPE in "$@"; do
  echo "$OBJECT $TYPE"

  # GEPBase note: the old encoding-parameter-scan sig3/sig4 config variants
  # are gone (no equivalent collections exist any more); jet_zprime_example
  # is new. Also note "sk" is no longer a valid --collection-sets value for
  # any GEPBase config -- the plain (non-EtaSK) "SK" collection variant does
  # not exist in this production, so requesting it alone raises an error.
  # Use "etask", "other", or "all" instead.

  [ "$OBJECT" = "jet" ] || [ "$OBJECT" = "all" ] && {
    gep-perf run configs/jet_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
    gep-perf run configs/jet_tthad_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
    gep-perf run configs/jet_zprime_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
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
