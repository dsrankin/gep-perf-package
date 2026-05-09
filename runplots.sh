#!/bin/bash

for TYPE in "$@"; do
  echo $TYPE
  if [[ "$TYPE" == *"pu"* ]]; then
    ./make_plots.sh $TYPE "E6LSB50G4SIG2" _e6lsb50g4sig2 false
    ./make_plots.sh $TYPE "E6LSB50G4SIG3" _e6lsb50g4sig3 false
    ./make_plots.sh $TYPE "E6LSB50G4SIG4" _e6lsb50g4sig4 false
    ./make_plots.sh $TYPE "E6LSB50G6SIG2" _e6lsb50g6sig2 false
    ./make_plots.sh $TYPE "E6LSB50G6SIG3" _e6lsb50g6sig3 false
    ./make_plots.sh $TYPE "E6LSB50G6SIG4" _e6lsb50g6sig4 false
  else
    ./make_plots.sh $TYPE E6LSB50 _n6lsb50 false
    ./make_plots.sh $TYPE LSB50G2 _lsb50g2 false
    ./make_plots.sh $TYPE SIG2 _2sig false
    ./make_plots.sh $TYPE SIG3 _3sig false
    ./make_plots.sh $TYPE SIG4 _4sig false
  fi
  ./make_plots.sh _ _ _ true
done
