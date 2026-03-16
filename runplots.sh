#!/bin/bash

TYPE=$1

./make_plots.sh $TYPE E6LSB50 _n6lsb50 false
./make_plots.sh $TYPE LSB50G2 _lsb50g2 false
./make_plots.sh $TYPE SIG2 _2sig false
./make_plots.sh $TYPE SIG3 _3sig false
./make_plots.sh $TYPE SIG4 _4sig true
