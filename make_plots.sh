#!/bin/bash

OBJTYPE=$1
CELLSEL=$2
CELLLAB=$3
MAKETAR=$4

PLOTDIR_BASE=perf_plots
PLOT_SUBDIR=""

plot_cmd() {
    mkdir -p "${PLOTDIR_BASE}/${PLOT_SUBDIR}"
    gep-perf plot "$@" --plotdir "${PLOTDIR_BASE}/${PLOT_SUBDIR}"
}

echo $OBJTYPE
echo $CELLSEL
echo $CELLLAB
echo $MAKETAR

## Jets
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "jet" ]]; then
    echo "Jets"
    PLOT_SUBDIR="jet"
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n1_pt100.npz --name jet --plotlabel _pt100$CELLLAB --plottext "J100 [VBF HH->4b]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n2_pt60.npz --name jet --plotlabel _pt60$CELLLAB --plottext "2J60 [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n3_pt50.npz --name jet --plotlabel _pt50$CELLLAB --plottext "3J50 [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n4_pt40.npz --name jet --plotlabel _pt40$CELLLAB --plottext "4J40 [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n1_rate50.npz --name jet --plotlabel _rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n2_rate50.npz --name jet --plotlabel _rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n3_rate75.npz --name jet --plotlabel _rate75$CELLLAB --plottext "75 kHz [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n4_rate100.npz --name jet --plotlabel _rate100$CELLLAB --plottext "100 kHz [VBF HH->4b]" --nobj 4 --xmax 100. --noerr
    
    echo "Boosted Jets"
    PLOT_SUBDIR="boosted_jet"
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n1_boosted_pt100.npz --name boosted_jet --plotlabel _boosted_pt100$CELLLAB --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 1 --xmax 200. --noerr --dofulleff
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n2_boosted_pt60.npz --name boosted_jet --plotlabel _boosted_pt60$CELLLAB --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n3_boosted_pt50.npz --name boosted_jet --plotlabel _boosted_pt50$CELLLAB --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n4_boosted_pt40.npz --name boosted_jet --plotlabel _boosted_pt40$CELLLAB --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n1_boosted_rate50.npz --name boosted_jet --plotlabel _boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n2_boosted_rate50.npz --name boosted_jet --plotlabel _boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n3_boosted_rate75.npz --name boosted_jet --plotlabel _boosted_rate75$CELLLAB --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n4_boosted_rate100.npz --name boosted_jet --plotlabel _boosted_rate100$CELLLAB --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr

    echo "Jets (TTbar)"
    PLOT_SUBDIR="jet_ttbar"
    plot_cmd perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n1_pt100.npz --name jet_ttbar --plotlabel _pt100$CELLLAB --plottext "J100 [TTbar]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n2_pt60.npz --name jet_ttbar --plotlabel _pt60$CELLLAB --plottext "2J60 [TTbar]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n3_pt50.npz --name jet_ttbar --plotlabel _pt50$CELLLAB --plottext "3J50 [TTbar]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n4_pt40.npz --name jet_ttbar --plotlabel _pt40$CELLLAB --plottext "4J40 [TTbar]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n1_rate50.npz --name jet_ttbar --plotlabel _rate50$CELLLAB --plottext "50 kHz [TTbar]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n2_rate50.npz --name jet_ttbar --plotlabel _rate50$CELLLAB --plottext "50 kHz [TTbar]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n3_rate75.npz --name jet_ttbar --plotlabel _rate75$CELLLAB --plottext "75 kHz [TTbar]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,L*jFexSRJetRoI}_n4_rate100.npz --name jet_ttbar --plotlabel _rate100$CELLLAB --plottext "100 kHz [TTbar]" --nobj 4 --xmax 100. --noerr
fi
    

## Pileup-suppression jet comparison (none, SK, EtaSK)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "pujet" ]]; then
    echo "Pileup-suppression Jets"
    PLOT_SUBDIR="pujet"
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,Jet_A*422*SKJets,Jet_A*$CELLSEL*Tower*SKJets,Jet_A*Offline*Tower*SKJets,Jet_A*422*EtaSKJets,Jet_A*$CELLSEL*Tower*EtaSKJets,Jet_A*Offline*Tower*EtaSKJets,L*jFexSRJetRoI}_n1_pt100.npz --name pujet --plotlabel _pt100$CELLLAB --plottext "J100 [VBF HH->4b]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,Jet_A*422*SKJets,Jet_A*$CELLSEL*Tower*SKJets,Jet_A*Offline*Tower*SKJets,Jet_A*422*EtaSKJets,Jet_A*$CELLSEL*Tower*EtaSKJets,Jet_A*Offline*Tower*EtaSKJets,L*jFexSRJetRoI}_n2_pt60.npz --name pujet --plotlabel _pt60$CELLLAB --plottext "2J60 [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,A*Offline*TowerJets,Jet_A*422*SKJets,Jet_A*$CELLSEL*Tower*SKJets,Jet_A*Offline*Tower*SKJets,Jet_A*422*EtaSKJets,Jet_A*$CELLSEL*Tower*EtaSKJets,Jet_A*Offline*Tower*EtaSKJets,L*jFexSRJetRoI}_n1_rate50.npz --name pujet --plotlabel _rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 1 --xmax 200. --noerr
fi

## SKJets
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "skjet" ]]; then
    echo "SKJets"
    PLOT_SUBDIR="skjet"
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n1_pt100.npz --name skjet --plotlabel _pt100$CELLLAB --plottext "J100 [VBF HH->4b]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n2_pt60.npz --name skjet --plotlabel _pt60$CELLLAB --plottext "2J60 [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n3_pt50.npz --name skjet --plotlabel _pt50$CELLLAB --plottext "3J50 [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n4_pt40.npz --name skjet --plotlabel _pt40$CELLLAB --plottext "4J40 [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n1_rate50.npz --name skjet --plotlabel _rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n2_rate50.npz --name skjet --plotlabel _rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n3_rate75.npz --name skjet --plotlabel _rate75$CELLLAB --plottext "75 kHz [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n4_rate100.npz --name skjet --plotlabel _rate100$CELLLAB --plottext "100 kHz [VBF HH->4b]" --nobj 4 --xmax 100. --noerr
    
    echo "Boosted SKJets"
    PLOT_SUBDIR="boosted_skjet"
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n1_boosted_pt100.npz --name boosted_skjet --plotlabel _boosted_pt100$CELLLAB --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 1 --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n2_boosted_pt60.npz --name boosted_skjet --plotlabel _boosted_pt60$CELLLAB --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n3_boosted_pt50.npz --name boosted_skjet --plotlabel _boosted_pt50$CELLLAB --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n4_boosted_pt40.npz --name boosted_skjet --plotlabel _boosted_pt40$CELLLAB --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n1_boosted_rate50.npz --name boosted_skjet --plotlabel _boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 1 --noerr
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n2_boosted_rate50.npz --name boosted_skjet --plotlabel _boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n3_boosted_rate75.npz --name boosted_skjet --plotlabel _boosted_rate75$CELLLAB --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n4_boosted_rate100.npz --name boosted_skjet --plotlabel _boosted_rate100$CELLLAB --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr

    echo "SKJets"
    PLOT_SUBDIR="skjet_ttbar"
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n1_pt100.npz --name skjet_ttbar --plotlabel _pt100$CELLLAB --plottext "J100 [TTbar]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n2_pt60.npz --name skjet_ttbar --plotlabel _pt60$CELLLAB --plottext "2J60 [TTbar]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n3_pt50.npz --name skjet_ttbar --plotlabel _pt50$CELLLAB --plottext "3J50 [TTbar]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n4_pt40.npz --name skjet_ttbar --plotlabel _pt40$CELLLAB --plottext "4J40 [TTbar]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n1_rate50.npz --name skjet_ttbar --plotlabel _rate50$CELLLAB --plottext "50 kHz [TTbar]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n2_rate50.npz --name skjet_ttbar --plotlabel _rate50$CELLLAB --plottext "50 kHz [TTbar]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n3_rate75.npz --name skjet_ttbar --plotlabel _rate75$CELLLAB --plottext "75 kHz [TTbar]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n4_rate100.npz --name skjet_ttbar --plotlabel _rate100$CELLLAB --plottext "100 kHz [TTbar]" --nobj 4 --xmax 100. --noerr
fi
    
## EtaSKJets
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "etaskjet" ]]; then
    echo "EtaSKJets"
    PLOT_SUBDIR="etaskjet"
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n1_pt100.npz --name etaskjet --plotlabel _pt100$CELLLAB --plottext "J100 [VBF HH->4b]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n2_pt60.npz --name etaskjet --plotlabel _pt60$CELLLAB --plottext "2J60 [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n3_pt50.npz --name etaskjet --plotlabel _pt50$CELLLAB --plottext "3J50 [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n4_pt40.npz --name etaskjet --plotlabel _pt40$CELLLAB --plottext "4J40 [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n1_rate50.npz --name etaskjet --plotlabel _rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n2_rate50.npz --name etaskjet --plotlabel _rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n3_rate75.npz --name etaskjet --plotlabel _rate75$CELLLAB --plottext "75 kHz [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n4_rate100.npz --name etaskjet --plotlabel _rate100$CELLLAB --plottext "100 kHz [VBF HH->4b]" --nobj 4 --xmax 100. --noerr
    
    echo "Boosted EtaSKJets"
    PLOT_SUBDIR="boosted_etaskjet"
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n1_boosted_pt100.npz --name boosted_etaskjet --plotlabel _boosted_pt100$CELLLAB --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 1 --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n2_boosted_pt60.npz --name boosted_etaskjet --plotlabel _boosted_pt60$CELLLAB --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n3_boosted_pt50.npz --name boosted_etaskjet --plotlabel _boosted_pt50$CELLLAB --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n4_boosted_pt40.npz --name boosted_etaskjet --plotlabel _boosted_pt40$CELLLAB --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n1_boosted_rate50.npz --name boosted_etaskjet --plotlabel _boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 1 --noerr
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n2_boosted_rate50.npz --name boosted_etaskjet --plotlabel _boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n3_boosted_rate75.npz --name boosted_etaskjet --plotlabel _boosted_rate75$CELLLAB --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n4_boosted_rate100.npz --name boosted_etaskjet --plotlabel _boosted_rate100$CELLLAB --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr

    echo "EtaSKJets"
    PLOT_SUBDIR="etaskjet_ttbar"
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n1_pt100.npz --name etaskjet_ttbar --plotlabel _pt100$CELLLAB --plottext "J100 [TTbar]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n2_pt60.npz --name etaskjet_ttbar --plotlabel _pt60$CELLLAB --plottext "2J60 [TTbar]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n3_pt50.npz --name etaskjet_ttbar --plotlabel _pt50$CELLLAB --plottext "3J50 [TTbar]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n4_pt40.npz --name etaskjet_ttbar --plotlabel _pt40$CELLLAB --plottext "4J40 [TTbar]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n1_rate50.npz --name etaskjet_ttbar --plotlabel _rate50$CELLLAB --plottext "50 kHz [TTbar]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n2_rate50.npz --name etaskjet_ttbar --plotlabel _rate50$CELLLAB --plottext "50 kHz [TTbar]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n3_rate75.npz --name etaskjet_ttbar --plotlabel _rate75$CELLLAB --plottext "75 kHz [TTbar]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n4_rate100.npz --name etaskjet_ttbar --plotlabel _rate100$CELLLAB --plottext "100 kHz [TTbar]" --nobj 4 --xmax 100. --noerr
fi
    
## Jets (VBF)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "jet" ]]; then
    echo "VBF Jets"
    PLOT_SUBDIR="vbf_jet"
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n1_pt100.npz --name jet --plotlabel _mjj_pt100$CELLLAB --plottext "J100 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n2_pt60.npz --name jet --plotlabel _mjj_pt60$CELLLAB --plottext "2J60 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n3_pt50.npz --name jet --plotlabel _mjj_pt50$CELLLAB --plottext "3J50 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n4_pt40.npz --name jet --plotlabel _mjj_pt40$CELLLAB --plottext "4J40 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n1_rate50.npz --name jet --plotlabel _mjj_rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n2_rate50.npz --name jet --plotlabel _mjj_rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n3_rate75.npz --name jet --plotlabel _mjj_rate75$CELLLAB --plottext "75 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n4_rate100.npz --name jet --plotlabel _mjj_rate100$CELLLAB --plottext "100 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    
    echo "VBF Boosted Jets"
    PLOT_SUBDIR="vbf_boosted_jet"
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n1_boosted_pt100.npz --name boosted_jet --plotlabel _mjj_boosted_pt100$CELLLAB --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n2_boosted_pt60.npz --name boosted_jet --plotlabel _mjj_boosted_pt60$CELLLAB --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n3_boosted_pt50.npz --name boosted_jet --plotlabel _mjj_boosted_pt50$CELLLAB --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n4_boosted_pt40.npz --name boosted_jet --plotlabel _mjj_boosted_pt40$CELLLAB --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n1_boosted_rate50.npz --name boosted_jet --plotlabel _mjj_boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n2_boosted_rate50.npz --name boosted_jet --plotlabel _mjj_boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n3_boosted_rate75.npz --name boosted_jet --plotlabel _mjj_boosted_rate75$CELLLAB --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,A*Offline*TowerJets,jFexSRJetRoI}_n4_boosted_rate100.npz --name boosted_jet --plotlabel _mjj_boosted_rate100$CELLLAB --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
fi

## SKJets (VBF)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "skjet" ]]; then
    echo "VBF SKJets"
    PLOT_SUBDIR="vbf_skjet"
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n1_pt100.npz --name skjet --plotlabel _mjj_pt100$CELLLAB --plottext "J100 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n2_pt60.npz --name skjet --plotlabel _mjj_pt60$CELLLAB --plottext "2J60 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n3_pt50.npz --name skjet --plotlabel _mjj_pt50$CELLLAB --plottext "3J50 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n4_pt40.npz --name skjet --plotlabel _mjj_pt40$CELLLAB --plottext "4J40 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n1_rate50.npz --name skjet --plotlabel _mjj_rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n2_rate50.npz --name skjet --plotlabel _mjj_rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n3_rate75.npz --name skjet --plotlabel _mjj_rate75$CELLLAB --plottext "75 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n4_rate100.npz --name skjet --plotlabel _mjj_rate100$CELLLAB --plottext "100 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    
    echo "VBF Boosted SKJets"
    PLOT_SUBDIR="vbf_boosted_skjet"
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n1_boosted_pt100.npz --name boosted_skjet --plotlabel _mjj_boosted_pt100$CELLLAB --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n2_boosted_pt60.npz --name boosted_skjet --plotlabel _mjj_boosted_pt60$CELLLAB --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n3_boosted_pt50.npz --name boosted_skjet --plotlabel _mjj_boosted_pt50$CELLLAB --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n4_boosted_pt40.npz --name boosted_skjet --plotlabel _mjj_boosted_pt40$CELLLAB --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n1_boosted_rate50.npz --name boosted_skjet --plotlabel _mjj_boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n2_boosted_rate50.npz --name boosted_skjet --plotlabel _mjj_boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n3_boosted_rate75.npz --name boosted_skjet --plotlabel _mjj_boosted_rate75$CELLLAB --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}SKJets_n4_boosted_rate100.npz --name boosted_skjet --plotlabel _mjj_boosted_rate100$CELLLAB --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
fi

## EtaSKJets (VBF)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "etaskjet" ]]; then
    echo "VBF EtaSKJets"
    PLOT_SUBDIR="vbf_etaskjet"
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n1_pt100.npz --name etaskjet --plotlabel _mjj_pt100$CELLLAB --plottext "J100 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n2_pt60.npz --name etaskjet --plotlabel _mjj_pt60$CELLLAB --plottext "2J60 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n3_pt50.npz --name etaskjet --plotlabel _mjj_pt50$CELLLAB --plottext "3J50 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n4_pt40.npz --name etaskjet --plotlabel _mjj_pt40$CELLLAB --plottext "4J40 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n1_rate50.npz --name etaskjet --plotlabel _mjj_rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n2_rate50.npz --name etaskjet --plotlabel _mjj_rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n3_rate75.npz --name etaskjet --plotlabel _mjj_rate75$CELLLAB --plottext "75 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n4_rate100.npz --name etaskjet --plotlabel _mjj_rate100$CELLLAB --plottext "100 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    
    echo "VBF Boosted EtaSKJets"
    PLOT_SUBDIR="vbf_boosted_etaskjet"
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n1_boosted_pt100.npz --name boosted_etaskjet --plotlabel _mjj_boosted_pt100$CELLLAB --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n2_boosted_pt60.npz --name boosted_etaskjet --plotlabel _mjj_boosted_pt60$CELLLAB --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n3_boosted_pt50.npz --name boosted_etaskjet --plotlabel _mjj_boosted_pt50$CELLLAB --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n4_boosted_pt40.npz --name boosted_etaskjet --plotlabel _mjj_boosted_pt40$CELLLAB --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n1_boosted_rate50.npz --name boosted_etaskjet --plotlabel _mjj_boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n2_boosted_rate50.npz --name boosted_etaskjet --plotlabel _mjj_boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n3_boosted_rate75.npz --name boosted_etaskjet --plotlabel _mjj_boosted_rate75$CELLLAB --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower,*Offline*Tower}EtaSKJets_n4_boosted_rate100.npz --name boosted_etaskjet --plotlabel _mjj_boosted_rate100$CELLLAB --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
fi


## Taus
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "tau" ]]; then
    echo "Taus"
    PLOT_SUBDIR="tau"
    plot_cmd perf_results/Tau_*{422Jets,$CELLSEL*TowerJets,eTauRoISim,jFexSRJetRoI}_n1_pt150.npz --name tau --plotlabel _pt150$CELLLAB --plottext "Tau150 [y*->tt]" --nobj 1 --xmax 230. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Tau_*{422Jets,$CELLSEL*TowerJets,eTauRoISim,jFexSRJetRoI}_n2_pt40.npz --name tau --plotlabel _pt40$CELLLAB --plottext "2Tau40 [y*->tt]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Tau_*{422Jets,$CELLSEL*TowerJets,eTauRoISim,jFexSRJetRoI}_n1_rate10.npz --name tau --plotlabel _rate10$CELLLAB --plottext "10 kHz [y*->tt]" --nobj 1 --xmax 230. --noerr
    plot_cmd perf_results/Tau_*{422Jets,$CELLSEL*TowerJets,eTauRoISim,jFexSRJetRoI}_n2_rate200.npz --name tau --plotlabel _rate200$CELLLAB --plottext "200 kHz [y*->tt]" --nobj 2 --xmax 100. --noerr
fi


## Pileup-suppression tau comparison (none, SK, EtaSK)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "putau" ]]; then
    echo "Pileup-suppression Taus"
    PLOT_SUBDIR="putau"
    plot_cmd perf_results/Tau_{*422Jets,*$CELLSEL*TowerJets,Tau_A*422*SKJets,Tau_A*$CELLSEL*Tower*SKJets,Tau_A*422*EtaSKJets,Tau_A*$CELLSEL*Tower*EtaSKJets,eTauRoISim,jFexSRJetRoI}_n1_pt150.npz --name putau --plotlabel _pt150$CELLLAB --plottext "Tau150 [y*->tt]" --nobj 1 --xmax 250. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Tau_{*422Jets,*$CELLSEL*TowerJets,Tau_A*422*SKJets,Tau_A*$CELLSEL*Tower*SKJets,Tau_A*422*EtaSKJets,Tau_A*$CELLSEL*Tower*EtaSKJets,eTauRoISim,jFexSRJetRoI}_n2_pt40.npz --name putau --plotlabel _pt40$CELLLAB --plottext "2Tau40 [y*->tt]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Tau_{*422Jets,*$CELLSEL*TowerJets,Tau_A*422*SKJets,Tau_A*$CELLSEL*Tower*SKJets,Tau_A*422*EtaSKJets,Tau_A*$CELLSEL*Tower*EtaSKJets,eTauRoISim,jFexSRJetRoI}_n1_rate10.npz --name putau --plotlabel _rate10$CELLLAB --plottext "10 kHz [y*->tt]" --nobj 1 --xmax 250. --noerr
    plot_cmd perf_results/Tau_{*422Jets,*$CELLSEL*TowerJets,Tau_A*422*SKJets,Tau_A*$CELLSEL*Tower*SKJets,Tau_A*422*EtaSKJets,Tau_A*$CELLSEL*Tower*EtaSKJets,eTauRoISim,jFexSRJetRoI}_n2_rate200.npz --name putau --plotlabel _rate200$CELLLAB --plottext "200 kHz [y*->tt]" --nobj 2 --xmax 100. --noerr
fi

## SKTaus
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "sktau" ]]; then
    echo "SKTaus"
    PLOT_SUBDIR="sktau"
    plot_cmd perf_results/Tau_A{*422,*$CELLSEL*Tower}SKJets_n1_pt150.npz --name sktau --plotlabel _pt150$CELLLAB --plottext "Tau150 [y*->tt]" --nobj 1 --xmax 250. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Tau_A{*422,*$CELLSEL*Tower}SKJets_n2_pt40.npz --name sktau --plotlabel _pt40$CELLLAB --plottext "2Tau40 [y*->tt]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Tau_A{*422,*$CELLSEL*Tower}SKJets_n1_rate10.npz --name sktau --plotlabel _rate10$CELLLAB --plottext "10 kHz [y*->tt]" --nobj 1 --xmax 250. --noerr
    plot_cmd perf_results/Tau_A{*422,*$CELLSEL*Tower}SKJets_n2_rate200.npz --name sktau --plotlabel _rate200$CELLLAB --plottext "200 kHz [y*->tt]" --nobj 2 --xmax 100. --noerr
fi

## EtaEtaSKTaus
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "etasktau" ]]; then
    echo "EtaSKTaus"
    PLOT_SUBDIR="etasktau"
    plot_cmd perf_results/Tau_A{*422,*$CELLSEL*Tower}EtaSKJets_n1_pt150.npz --name etasktau --plotlabel _pt150$CELLLAB --plottext "Tau150 [y*->tt]" --nobj 1 --xmax 250. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Tau_A{*422,*$CELLSEL*Tower}EtaSKJets_n2_pt40.npz --name etasktau --plotlabel _pt40$CELLLAB --plottext "2Tau40 [y*->tt]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Tau_A{*422,*$CELLSEL*Tower}EtaSKJets_n1_rate10.npz --name etasktau --plotlabel _rate10$CELLLAB --plottext "10 kHz [y*->tt]" --nobj 1 --xmax 250. --noerr
    plot_cmd perf_results/Tau_A{*422,*$CELLSEL*Tower}EtaSKJets_n2_rate200.npz --name etasktau --plotlabel _rate200$CELLLAB --plottext "200 kHz [y*->tt]" --nobj 2 --xmax 100. --noerr
fi

## Eles
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "ele" ]]; then
    echo "Eles"
    PLOT_SUBDIR="ele"
    plot_cmd perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n1_pt20.npz --name ele --plotlabel _pt20$CELLLAB --plottext "EM20 [Z->ee]" --nobj 1 --xmax 100. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n2_pt20.npz --name ele --plotlabel _pt20$CELLLAB --plottext "2EM20 [Z->ee]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n1_rate50.npz --name ele --plotlabel _rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n2_rate50.npz --name ele --plotlabel _rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 2 --xmax 100. --noerr
    
    echo "Eratio Eles"
    PLOT_SUBDIR="eratio_ele"
    plot_cmd perf_results/Ele_*eEMRoISim*${CELLSEL}*_n1_eratio_pt20.npz --name eratio_ele --plotlabel _eratio_pt20$CELLLAB --plottext "EM20 [Z->ee]" --nobj 1 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_*eEMRoISim*${CELLSEL}*_n2_eratio_pt20.npz --name eratio_ele --plotlabel _eratio_pt20$CELLLAB --plottext "2EM20 [Z->ee]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_*eEMRoISim*${CELLSEL}*_n1_eratio_rate50.npz --name eratio_ele --plotlabel _eratio_rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Ele_*eEMRoISim*${CELLSEL}*_n2_eratio_rate50.npz --name eratio_ele --plotlabel _eratio_rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 2 --xmax 100. --noerr
    
    echo "Barrel Eles"
    PLOT_SUBDIR="barrel_ele"
    plot_cmd perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n1_barrel_pt20.npz --name barrel_ele --plotlabel _barrel_pt20$CELLLAB --plottext "EM20 [Z->ee]" --nobj 1 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n2_barrel_pt20.npz --name barrel_ele --plotlabel _barrel_pt20$CELLLAB --plottext "2EM20 [Z->ee]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n1_barrel_rate50.npz --name barrel_ele --plotlabel _barrel_rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n2_barrel_rate50.npz --name barrel_ele --plotlabel _barrel_rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 2 --xmax 100. --noerr
    
    echo "Barrel Eratio Eles"
    PLOT_SUBDIR="barrel_eratio_ele"
    plot_cmd perf_results/Ele_*eEMRoISim*${CELLSEL}*_n1_barrel_eratio_pt20.npz --name barrel_eratio_ele --plotlabel _barrel_eratio_pt20$CELLLAB --plottext "EM20 [Z->ee]" --nobj 1 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_*eEMRoISim*${CELLSEL}*_n2_barrel_eratio_pt20.npz --name barrel_eratio_ele --plotlabel _barrel_eratio_pt20$CELLLAB --plottext "2EM20 [Z->ee]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_*eEMRoISim*${CELLSEL}*_n1_barrel_eratio_rate50.npz --name barrel_eratio_ele --plotlabel _barrel_eratio_rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Ele_*eEMRoISim*${CELLSEL}*_n2_barrel_eratio_rate50.npz --name barrel_eratio_ele --plotlabel _barrel_eratio_rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 2 --xmax 100. --noerr
fi

## Phos
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "pho" ]]; then
    echo "Phos"
    PLOT_SUBDIR="pho"
    plot_cmd perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n1_pt20.npz --name pho --plotlabel _pt20$CELLLAB --plottext "EM20 [H->yy]" --nobj 1 --xmax 100. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n2_pt20.npz --name pho --plotlabel _pt20$CELLLAB --plottext "2EM20 [H->yy]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n1_rate50.npz --name pho --plotlabel _rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n2_rate50.npz --name pho --plotlabel _rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 2 --xmax 100. --noerr
    
    echo "Eratio Phos"
    PLOT_SUBDIR="eratio_pho"
    plot_cmd perf_results/Pho_*eEMRoISim*${CELLSEL}*_n1_eratio_pt20.npz --name eratio_pho --plotlabel _eratio_pt20$CELLLAB --plottext "EM20 [H->yy]" --nobj 1 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_*eEMRoISim*${CELLSEL}*_n2_eratio_pt20.npz --name eratio_pho --plotlabel _eratio_pt20$CELLLAB --plottext "2EM20 [H->yy]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_*eEMRoISim*${CELLSEL}*_n1_eratio_rate50.npz --name eratio_pho --plotlabel _eratio_rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Pho_*eEMRoISim*${CELLSEL}*_n2_eratio_rate50.npz --name eratio_pho --plotlabel _eratio_rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 2 --xmax 100. --noerr
    
    echo "Barrel Phos"
    PLOT_SUBDIR="barrel_pho"
    plot_cmd perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n1_barrel_pt20.npz --name barrel_pho --plotlabel _barrel_pt20$CELLLAB --plottext "EM20 [H->yy]" --nobj 1 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n2_barrel_pt20.npz --name barrel_pho --plotlabel _barrel_pt20$CELLLAB --plottext "2EM20 [H->yy]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n1_barrel_rate50.npz --name barrel_pho --plotlabel _barrel_rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n2_barrel_rate50.npz --name barrel_pho --plotlabel _barrel_rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 2 --xmax 100. --noerr
    
    echo "Barrel Eratio Phos"
    PLOT_SUBDIR="barrel_eratio_pho"
    plot_cmd perf_results/Pho_*eEMRoISim*${CELLSEL}*_n1_barrel_eratio_pt20.npz --name barrel_eratio_pho --plotlabel _barrel_eratio_pt20$CELLLAB --plottext "EM20 [H->yy]" --nobj 1 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_*eEMRoISim*${CELLSEL}*_n2_barrel_eratio_pt20.npz --name barrel_eratio_pho --plotlabel _barrel_eratio_pt20$CELLLAB --plottext "2EM20 [H->yy]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_*eEMRoISim*${CELLSEL}*_n1_barrel_eratio_rate50.npz --name barrel_eratio_pho --plotlabel _barrel_eratio_rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Pho_*eEMRoISim*${CELLSEL}*_n2_barrel_eratio_rate50.npz --name barrel_eratio_pho --plotlabel _barrel_eratio_rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 2 --xmax 100. --noerr
fi


## Pileup-suppression MET comparison (none, SK, EtaSK)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "pumet" ]]; then
    echo "Pileup-suppression MET"
    PLOT_SUBDIR="pumet"
    plot_cmd perf_results/MET_{gFex*,*422,*Offline*Tower,*$CELLSEL*Tower,MET_*422*SK,MET_*Offline*Tower*SK,MET_*$CELLSEL*Tower*SK,MET_*422*EtaSK,MET_*Offline*Tower*EtaSK,MET_*$CELLSEL*Tower*EtaSK}_n1_pt150.npz --name pumet --plotlabel _pt150$CELLLAB --plottext "MET150 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/MET_{gFex*,*422,*Offline*Tower,*$CELLSEL*Tower,MET_*422*SK,MET_*Offline*Tower*SK,MET_*$CELLSEL*Tower*SK,MET_*422*EtaSK,MET_*Offline*Tower*EtaSK,MET_*$CELLSEL*Tower*EtaSK}_n1_pt200.npz --name pumet --plotlabel _pt200$CELLLAB --plottext "MET200 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff
    plot_cmd perf_results/MET_{gFex*,*422,*Offline*Tower,*$CELLSEL*Tower,MET_*422*SK,MET_*Offline*Tower*SK,MET_*$CELLSEL*Tower*SK,MET_*422*EtaSK,MET_*Offline*Tower*EtaSK,MET_*$CELLSEL*Tower*EtaSK}_n1_rate60.npz --name pumet --plotlabel _rate60$CELLLAB --plottext "60 kHz [ZH->vvbb]" --nobj 1 --xmax 500. --noerr
fi

## MET
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "met" ]]; then
    echo "MET"
    PLOT_SUBDIR="met"
    plot_cmd perf_results/MET_{gFex*,*422,*Offline*Tower,*$CELLSEL*Tower}_n1_pt150.npz --name met --plotlabel _pt150$CELLLAB --plottext "MET150 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/MET_{gFex*,*422,*Offline*Tower,*$CELLSEL*Tower}_n1_pt200.npz --name met --plotlabel _pt200$CELLLAB --plottext "MET200 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff
    plot_cmd perf_results/MET_{gFex*,*422,*Offline*Tower,*$CELLSEL*Tower}_n1_rate60.npz --name met --plotlabel _rate60$CELLLAB --plottext "60 kHz [ZH->vvbb]" --nobj 1 --xmax 500. --noerr
fi
    
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "skmet" ]]; then
    echo "SKMET"
    PLOT_SUBDIR="skmet"
    plot_cmd perf_results/MET_*{422*,Offline*Tower,$CELLSEL*Tower}SK_n1_pt150.npz --name skmet --plotlabel _pt150$CELLLAB --plottext "MET150 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/MET_*{422*,Offline*Tower,$CELLSEL*Tower}SK_n1_pt200.npz --name skmet --plotlabel _pt200$CELLLAB --plottext "MET200 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff
    plot_cmd perf_results/MET_*{422*,Offline*Tower,$CELLSEL*Tower}SK_n1_rate60.npz --name skmet --plotlabel _rate60$CELLLAB --plottext "60 kHz [ZH->vvbb]" --nobj 1 --xmax 500. --noerr
fi

if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "etaskmet" ]]; then
    echo "EtaSKMET"
    PLOT_SUBDIR="etaskmet"
    plot_cmd perf_results/MET_*{422*,Offline*Tower,$CELLSEL*Tower}EtaSK_n1_pt150.npz --name etaskmet --plotlabel _pt150$CELLLAB --plottext "MET150 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/MET_*{422*,Offline*Tower,$CELLSEL*Tower}EtaSK_n1_pt200.npz --name etaskmet --plotlabel _pt200$CELLLAB --plottext "MET200 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff
    plot_cmd perf_results/MET_*{422*,Offline*Tower,$CELLSEL*Tower}EtaSK_n1_rate60.npz --name etaskmet --plotlabel _rate60$CELLLAB --plottext "60 kHz [ZH->vvbb]" --nobj 1 --xmax 500. --noerr
fi


if [[ "$MAKETAR" == "true" ]]; then
    tar -cvz -f perf_plots.tar.gz "${PLOTDIR_BASE}/"
    mapfile -t eff_rate_files < <(find "${PLOTDIR_BASE}" -type f -name "*efficiency*rate*" -print)
    if (( ${#eff_rate_files[@]} > 0 )); then
        tar -cvz -f perf_plots_eff_rate.tar.gz "${eff_rate_files[@]}" --transform "s|^${PLOTDIR_BASE}|perf_plots_eff_rate|"
    fi

    mapfile -t eff_pt_files < <(find "${PLOTDIR_BASE}" -type f -name "*efficiency*pt*" -print)
    if (( ${#eff_pt_files[@]} > 0 )); then
        tar -cvz -f perf_plots_eff_pt.tar.gz "${eff_pt_files[@]}" --transform "s|^${PLOTDIR_BASE}|perf_plots_eff_pt|"
    fi

    mapfile -t resp_resol_files < <(find "${PLOTDIR_BASE}" -type f \( -name "*resp*" -o -name "*resol*" \) -print)
    if (( ${#resp_resol_files[@]} > 0 )); then
        tar -cvz -f perf_plots_respresol.tar.gz "${resp_resol_files[@]}" --transform "s|^${PLOTDIR_BASE}|perf_plots_respresol|"
    fi
fi
