#!/bin/bash

OBJECT="$1"
shift

for TYPE in "$@"; do
  echo "$OBJECT $TYPE"

  # GEPBase note: the old encoding-parameter-scan sig3/sig4 config variants
  # are gone (no equivalent collections exist any more). Also note "sk" is
  # not a useful --collection-sets value for jet/jet_tthad/tau/ele/pho/met
  # (the plain, non-EtaSK "SK" collection variant does not exist in this
  # production, so requesting it alone raises an error there) -- use
  # "etask", "other", or "all" instead. jet_larger_example is the one
  # exception: AntiKt10UFOCSSKJets' name happens to end in "...CSSK", which
  # contains "SK" as a substring, so it gets misclassified into the "sk"
  # bucket even though it has no EtaSK counterpart. Run jet_larger_example
  # with "all" (not just "etask"/"other") or its results go missing.

  # The plotting script's jet overlays consume both the VBF HH and ttbar AK4
  # results.  Produce both configurations here; previously this branch ran
  # only the AK10 configuration, leaving all AK4 (including uncorrected)
  # inputs absent at plotting time.
  if [ "$OBJECT" = "jet" ] || [ "$OBJECT" = "all" ]; then
    gep-perf run configs/jet_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
    gep-perf run configs/jet_tthad_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
  fi

  if [ "$OBJECT" = "fatjet" ] || [ "$OBJECT" = "all" ]; then
    gep-perf run configs/jet_larger_example.yaml --plotdir perf_plots --resdir perf_results --collection-sets "$TYPE"
  fi

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
