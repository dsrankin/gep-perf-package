#!/bin/bash
#
# GEPBase note: there is no more GEP encoding-parameter scan to loop over
# (the old script called make_plots.sh once per E6LSB50/LSB50G2/SIG2/SIG3/SIG4
# variant), so this is now a thin wrapper that just runs make_plots.sh once
# per requested object type and tars up the results.

for TYPE in "$@"; do
  echo $TYPE
  ./make_plots.sh $TYPE true
done
