#!/bin/bash

OBJTYPE=$1
CELLSEL=$2
CELLLAB=$3
MAKETAR=$4

echo $OBJTYPE
echo $CELLSEL
echo $CELLLAB
echo $MAKETAR

## Jets
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "jet" ]]; then
    echo "Jets"
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n1_pt100.npz --plotdir perf_plots --name jet --plotlabel _pt100$CELLLAB --plottext "J100 [VBF HH->4b]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n2_pt60.npz --plotdir perf_plots --name jet --plotlabel _pt60$CELLLAB --plottext "2J60 [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n3_pt50.npz --plotdir perf_plots --name jet --plotlabel _pt50$CELLLAB --plottext "3J50 [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n4_pt40.npz --plotdir perf_plots --name jet --plotlabel _pt40$CELLLAB --plottext "4J40 [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n1_rate50.npz --plotdir perf_plots --name jet --plotlabel _rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 1 --xmax 200. --noerr
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n2_rate50.npz --plotdir perf_plots --name jet --plotlabel _rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n3_rate75.npz --plotdir perf_plots --name jet --plotlabel _rate75$CELLLAB --plottext "75 kHz [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n4_rate100.npz --plotdir perf_plots --name jet --plotlabel _rate100$CELLLAB --plottext "100 kHz [VBF HH->4b]" --nobj 4 --xmax 100. --noerr
    
    echo "Boosted Jets"
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n1_boosted_pt100.npz --plotdir perf_plots --name boosted_jet --plotlabel _boosted_pt100$CELLLAB --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 1 --xmax 200. --noerr --dofulleff
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n2_boosted_pt60.npz --plotdir perf_plots --name boosted_jet --plotlabel _boosted_pt60$CELLLAB --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n3_boosted_pt50.npz --plotdir perf_plots --name boosted_jet --plotlabel _boosted_pt50$CELLLAB --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n4_boosted_pt40.npz --plotdir perf_plots --name boosted_jet --plotlabel _boosted_pt40$CELLLAB --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n1_boosted_rate50.npz --plotdir perf_plots --name boosted_jet --plotlabel _boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 1 --xmax 200. --noerr
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n2_boosted_rate50.npz --plotdir perf_plots --name boosted_jet --plotlabel _boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n3_boosted_rate75.npz --plotdir perf_plots --name boosted_jet --plotlabel _boosted_rate75$CELLLAB --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    gep-perf plot perf_results/Jet_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n4_boosted_rate100.npz --plotdir perf_plots --name boosted_jet --plotlabel _boosted_rate100$CELLLAB --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr

    echo "Jets (TTbar)"
    gep-perf plot perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n1_pt100.npz --plotdir perf_plots --name jet_ttbar --plotlabel _pt100$CELLLAB --plottext "J100 [TTbar]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    gep-perf plot perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n2_pt60.npz --plotdir perf_plots --name jet_ttbar --plotlabel _pt60$CELLLAB --plottext "2J60 [TTbar]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n3_pt50.npz --plotdir perf_plots --name jet_ttbar --plotlabel _pt50$CELLLAB --plottext "3J50 [TTbar]" --nobj 3 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n4_pt40.npz --plotdir perf_plots --name jet_ttbar --plotlabel _pt40$CELLLAB --plottext "4J40 [TTbar]" --nobj 4 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n1_rate50.npz --plotdir perf_plots --name jet_ttbar --plotlabel _rate50$CELLLAB --plottext "50 kHz [TTbar]" --nobj 1 --xmax 200. --noerr
    gep-perf plot perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n2_rate50.npz --plotdir perf_plots --name jet_ttbar --plotlabel _rate50$CELLLAB --plottext "50 kHz [TTbar]" --nobj 2 --xmax 150. --noerr
    gep-perf plot perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n3_rate75.npz --plotdir perf_plots --name jet_ttbar --plotlabel _rate75$CELLLAB --plottext "75 kHz [TTbar]" --nobj 3 --xmax 150. --noerr
    gep-perf plot perf_results/Jet_ttbar_{A*422Jets,A*$CELLSEL*TowerJets,L*jFexSRJetRoI}_n4_rate100.npz --plotdir perf_plots --name jet_ttbar --plotlabel _rate100$CELLLAB --plottext "100 kHz [TTbar]" --nobj 4 --xmax 100. --noerr
    
## SKJets
    echo "SKJets"
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n1_pt100.npz --plotdir perf_plots --name skjet --plotlabel _pt100$CELLLAB --plottext "J100 [VBF HH->4b]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n2_pt60.npz --plotdir perf_plots --name skjet --plotlabel _pt60$CELLLAB --plottext "2J60 [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n3_pt50.npz --plotdir perf_plots --name skjet --plotlabel _pt50$CELLLAB --plottext "3J50 [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n4_pt40.npz --plotdir perf_plots --name skjet --plotlabel _pt40$CELLLAB --plottext "4J40 [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n1_rate50.npz --plotdir perf_plots --name skjet --plotlabel _rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 1 --xmax 200. --noerr
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n2_rate50.npz --plotdir perf_plots --name skjet --plotlabel _rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n3_rate75.npz --plotdir perf_plots --name skjet --plotlabel _rate75$CELLLAB --plottext "75 kHz [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n4_rate100.npz --plotdir perf_plots --name skjet --plotlabel _rate100$CELLLAB --plottext "100 kHz [VBF HH->4b]" --nobj 4 --xmax 100. --noerr
    
    echo "Boosted SKJets"
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n1_boosted_pt100.npz --plotdir perf_plots --name boosted_skjet --plotlabel _boosted_pt100$CELLLAB --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 1 --noerr --dofulleff
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n2_boosted_pt60.npz --plotdir perf_plots --name boosted_skjet --plotlabel _boosted_pt60$CELLLAB --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n3_boosted_pt50.npz --plotdir perf_plots --name boosted_skjet --plotlabel _boosted_pt50$CELLLAB --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n4_boosted_pt40.npz --plotdir perf_plots --name boosted_skjet --plotlabel _boosted_pt40$CELLLAB --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n1_boosted_rate50.npz --plotdir perf_plots --name boosted_skjet --plotlabel _boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 1 --noerr
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n2_boosted_rate50.npz --plotdir perf_plots --name boosted_skjet --plotlabel _boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n3_boosted_rate75.npz --plotdir perf_plots --name boosted_skjet --plotlabel _boosted_rate75$CELLLAB --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    gep-perf plot perf_results/Jet_A{*422,*$CELLSEL*Tower}SKJets_n4_boosted_rate100.npz --plotdir perf_plots --name boosted_skjet --plotlabel _boosted_rate100$CELLLAB --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr

    echo "SKJets"
    gep-perf plot perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower}SKJets_n1_pt100.npz --plotdir perf_plots --name skjet_ttbar --plotlabel _pt100$CELLLAB --plottext "J100 [TTbar]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    gep-perf plot perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower}SKJets_n2_pt60.npz --plotdir perf_plots --name skjet_ttbar --plotlabel _pt60$CELLLAB --plottext "2J60 [TTbar]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower}SKJets_n3_pt50.npz --plotdir perf_plots --name skjet_ttbar --plotlabel _pt50$CELLLAB --plottext "3J50 [TTbar]" --nobj 3 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower}SKJets_n4_pt40.npz --plotdir perf_plots --name skjet_ttbar --plotlabel _pt40$CELLLAB --plottext "4J40 [TTbar]" --nobj 4 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower}SKJets_n1_rate50.npz --plotdir perf_plots --name skjet_ttbar --plotlabel _rate50$CELLLAB --plottext "50 kHz [TTbar]" --nobj 1 --xmax 200. --noerr
    gep-perf plot perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower}SKJets_n2_rate50.npz --plotdir perf_plots --name skjet_ttbar --plotlabel _rate50$CELLLAB --plottext "50 kHz [TTbar]" --nobj 2 --xmax 150. --noerr
    gep-perf plot perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower}SKJets_n3_rate75.npz --plotdir perf_plots --name skjet_ttbar --plotlabel _rate75$CELLLAB --plottext "75 kHz [TTbar]" --nobj 3 --xmax 150. --noerr
    gep-perf plot perf_results/Jet_ttbar_A{*422,*$CELLSEL*Tower}SKJets_n4_rate100.npz --plotdir perf_plots --name skjet_ttbar --plotlabel _rate100$CELLLAB --plottext "100 kHz [TTbar]" --nobj 4 --xmax 100. --noerr
    
## Jets (VBF)
    echo "VBF Jets"
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n1_pt100.npz --plotdir perf_plots --name jet --plotlabel _mjj_pt100$CELLLAB --plottext "J100 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n2_pt60.npz --plotdir perf_plots --name jet --plotlabel _mjj_pt60$CELLLAB --plottext "2J60 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n3_pt50.npz --plotdir perf_plots --name jet --plotlabel _mjj_pt50$CELLLAB --plottext "3J50 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n4_pt40.npz --plotdir perf_plots --name jet --plotlabel _mjj_pt40$CELLLAB --plottext "4J40 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n1_rate50.npz --plotdir perf_plots --name jet --plotlabel _mjj_rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n2_rate50.npz --plotdir perf_plots --name jet --plotlabel _mjj_rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n3_rate75.npz --plotdir perf_plots --name jet --plotlabel _mjj_rate75$CELLLAB --plottext "75 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n4_rate100.npz --plotdir perf_plots --name jet --plotlabel _mjj_rate100$CELLLAB --plottext "100 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    
    echo "VBF Boosted Jets"
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n1_boosted_pt100.npz --plotdir perf_plots --name boosted_jet --plotlabel _mjj_boosted_pt100$CELLLAB --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n2_boosted_pt60.npz --plotdir perf_plots --name boosted_jet --plotlabel _mjj_boosted_pt60$CELLLAB --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n3_boosted_pt50.npz --plotdir perf_plots --name boosted_jet --plotlabel _mjj_boosted_pt50$CELLLAB --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n4_boosted_pt40.npz --plotdir perf_plots --name boosted_jet --plotlabel _mjj_boosted_pt40$CELLLAB --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n1_boosted_rate50.npz --plotdir perf_plots --name boosted_jet --plotlabel _mjj_boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n2_boosted_rate50.npz --plotdir perf_plots --name boosted_jet --plotlabel _mjj_boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n3_boosted_rate75.npz --plotdir perf_plots --name boosted_jet --plotlabel _mjj_boosted_rate75$CELLLAB --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_*{422Jets,$CELLSEL*TowerJets,jFexSRJetRoI}_n4_boosted_rate100.npz --plotdir perf_plots --name boosted_jet --plotlabel _mjj_boosted_rate100$CELLLAB --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1

## SKJets (VBF)

    echo "VBF SKJets"
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n1_pt100.npz --plotdir perf_plots --name skjet --plotlabel _mjj_pt100$CELLLAB --plottext "J100 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n2_pt60.npz --plotdir perf_plots --name skjet --plotlabel _mjj_pt60$CELLLAB --plottext "2J60 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n3_pt50.npz --plotdir perf_plots --name skjet --plotlabel _mjj_pt50$CELLLAB --plottext "3J50 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n4_pt40.npz --plotdir perf_plots --name skjet --plotlabel _mjj_pt40$CELLLAB --plottext "4J40 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n1_rate50.npz --plotdir perf_plots --name skjet --plotlabel _mjj_rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n2_rate50.npz --plotdir perf_plots --name skjet --plotlabel _mjj_rate50$CELLLAB --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n3_rate75.npz --plotdir perf_plots --name skjet --plotlabel _mjj_rate75$CELLLAB --plottext "75 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n4_rate100.npz --plotdir perf_plots --name skjet --plotlabel _mjj_rate100$CELLLAB --plottext "100 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    
    echo "VBF Boosted SKJets"
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n1_boosted_pt100.npz --plotdir perf_plots --name boosted_skjet --plotlabel _mjj_boosted_pt100$CELLLAB --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n2_boosted_pt60.npz --plotdir perf_plots --name boosted_skjet --plotlabel _mjj_boosted_pt60$CELLLAB --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n3_boosted_pt50.npz --plotdir perf_plots --name boosted_skjet --plotlabel _mjj_boosted_pt50$CELLLAB --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n4_boosted_pt40.npz --plotdir perf_plots --name boosted_skjet --plotlabel _mjj_boosted_pt40$CELLLAB --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n1_boosted_rate50.npz --plotdir perf_plots --name boosted_skjet --plotlabel _mjj_boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n2_boosted_rate50.npz --plotdir perf_plots --name boosted_skjet --plotlabel _mjj_boosted_rate50$CELLLAB --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n3_boosted_rate75.npz --plotdir perf_plots --name boosted_skjet --plotlabel _mjj_boosted_rate75$CELLLAB --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    gep-perf plot perf_results/Jet_dijet_mass_A{*422,*$CELLSEL*Tower}SKJets_n4_boosted_rate100.npz --plotdir perf_plots --name boosted_skjet --plotlabel _mjj_boosted_rate100$CELLLAB --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
fi

## Taus
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "tau" ]]; then
    echo "Taus"
    gep-perf plot perf_results/Tau_*{422Jets,$CELLSEL*TowerJets,eTauRoISim,jFexSRJetRoI}_n1_pt150.npz --plotdir perf_plots --name tau --plotlabel _pt150$CELLLAB --plottext "Tau150 [y*->tt]" --nobj 1 --xmax 230. --noerr --dofulleff --dorespresol
    gep-perf plot perf_results/Tau_*{422Jets,$CELLSEL*TowerJets,eTauRoISim,jFexSRJetRoI}_n2_pt40.npz --plotdir perf_plots --name tau --plotlabel _pt40$CELLLAB --plottext "2Tau40 [y*->tt]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Tau_*{422Jets,$CELLSEL*TowerJets,eTauRoISim,jFexSRJetRoI}_n1_rate10.npz --plotdir perf_plots --name tau --plotlabel _rate10$CELLLAB --plottext "10 kHz [y*->tt]" --nobj 1 --xmax 230. --noerr
    gep-perf plot perf_results/Tau_*{422Jets,$CELLSEL*TowerJets,eTauRoISim,jFexSRJetRoI}_n2_rate200.npz --plotdir perf_plots --name tau --plotlabel _rate200$CELLLAB --plottext "200 kHz [y*->tt]" --nobj 2 --xmax 100. --noerr

## SKTaus

    echo "SKTaus"
    gep-perf plot perf_results/Tau_A{*422,*$CELLSEL*Tower}SKJets_n1_pt150.npz --plotdir perf_plots --name sktau --plotlabel _pt150$CELLLAB --plottext "Tau150 [y*->tt]" --nobj 1 --xmax 250. --noerr --dofulleff --dorespresol
    gep-perf plot perf_results/Tau_A{*422,*$CELLSEL*Tower}SKJets_n2_pt40.npz --plotdir perf_plots --name sktau --plotlabel _pt40$CELLLAB --plottext "2Tau40 [y*->tt]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Tau_A{*422,*$CELLSEL*Tower}SKJets_n1_rate10.npz --plotdir perf_plots --name sktau --plotlabel _rate10$CELLLAB --plottext "10 kHz [y*->tt]" --nobj 1 --xmax 250. --noerr
    gep-perf plot perf_results/Tau_A{*422,*$CELLSEL*Tower}SKJets_n2_rate200.npz --plotdir perf_plots --name sktau --plotlabel _rate200$CELLLAB --plottext "200 kHz [y*->tt]" --nobj 2 --xmax 100. --noerr
fi

## Eles
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "ele" ]]; then
    echo "Eles"
    gep-perf plot perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n1_pt20.npz --plotdir perf_plots --name ele --plotlabel _pt20$CELLLAB --plottext "EM20 [Z->ee]" --nobj 1 --xmax 100. --noerr --dofulleff --dorespresol
    gep-perf plot perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n2_pt20.npz --plotdir perf_plots --name ele --plotlabel _pt20$CELLLAB --plottext "2EM20 [Z->ee]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n1_rate50.npz --plotdir perf_plots --name ele --plotlabel _rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 1 --xmax 100. --noerr
    gep-perf plot perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n2_rate50.npz --plotdir perf_plots --name ele --plotlabel _rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 2 --xmax 100. --noerr
    
    echo "Eratio Eles"
    gep-perf plot perf_results/Ele_*eEMRoISim*${CELLSEL}*_n1_eratio_pt20.npz --plotdir perf_plots --name eratio_ele --plotlabel _eratio_pt20$CELLLAB --plottext "EM20 [Z->ee]" --nobj 1 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Ele_*eEMRoISim*${CELLSEL}*_n2_eratio_pt20.npz --plotdir perf_plots --name eratio_ele --plotlabel _eratio_pt20$CELLLAB --plottext "2EM20 [Z->ee]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Ele_*eEMRoISim*${CELLSEL}*_n1_eratio_rate50.npz --plotdir perf_plots --name eratio_ele --plotlabel _eratio_rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 1 --xmax 100. --noerr
    gep-perf plot perf_results/Ele_*eEMRoISim*${CELLSEL}*_n2_eratio_rate50.npz --plotdir perf_plots --name eratio_ele --plotlabel _eratio_rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 2 --xmax 100. --noerr
    
    echo "Barrel Eles"
    gep-perf plot perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n1_barrel_pt20.npz --plotdir perf_plots --name barrel_ele --plotlabel _barrel_pt20$CELLLAB --plottext "EM20 [Z->ee]" --nobj 1 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n2_barrel_pt20.npz --plotdir perf_plots --name barrel_ele --plotlabel _barrel_pt20$CELLLAB --plottext "2EM20 [Z->ee]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n1_barrel_rate50.npz --plotdir perf_plots --name barrel_ele --plotlabel _barrel_rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 1 --xmax 100. --noerr
    gep-perf plot perf_results/Ele_*{$CELLSEL*TowerJets,eEMRoISim}_n2_barrel_rate50.npz --plotdir perf_plots --name barrel_ele --plotlabel _barrel_rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 2 --xmax 100. --noerr
    
    echo "Barrel Eratio Eles"
    gep-perf plot perf_results/Ele_*eEMRoISim*${CELLSEL}*_n1_barrel_eratio_pt20.npz --plotdir perf_plots --name barrel_eratio_ele --plotlabel _barrel_eratio_pt20$CELLLAB --plottext "EM20 [Z->ee]" --nobj 1 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Ele_*eEMRoISim*${CELLSEL}*_n2_barrel_eratio_pt20.npz --plotdir perf_plots --name barrel_eratio_ele --plotlabel _barrel_eratio_pt20$CELLLAB --plottext "2EM20 [Z->ee]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Ele_*eEMRoISim*${CELLSEL}*_n1_barrel_eratio_rate50.npz --plotdir perf_plots --name barrel_eratio_ele --plotlabel _barrel_eratio_rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 1 --xmax 100. --noerr
    gep-perf plot perf_results/Ele_*eEMRoISim*${CELLSEL}*_n2_barrel_eratio_rate50.npz --plotdir perf_plots --name barrel_eratio_ele --plotlabel _barrel_eratio_rate50$CELLLAB --plottext "50 kHz [Z->ee]" --nobj 2 --xmax 100. --noerr
fi

## Phos
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "pho" ]]; then
    echo "Phos"
    gep-perf plot perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n1_pt20.npz --plotdir perf_plots --name pho --plotlabel _pt20$CELLLAB --plottext "EM20 [H->yy]" --nobj 1 --xmax 100. --noerr --dofulleff --dorespresol
    gep-perf plot perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n2_pt20.npz --plotdir perf_plots --name pho --plotlabel _pt20$CELLLAB --plottext "2EM20 [H->yy]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n1_rate50.npz --plotdir perf_plots --name pho --plotlabel _rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 1 --xmax 100. --noerr
    gep-perf plot perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n2_rate50.npz --plotdir perf_plots --name pho --plotlabel _rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 2 --xmax 100. --noerr
    
    echo "Eratio Phos"
    gep-perf plot perf_results/Pho_*eEMRoISim*${CELLSEL}*_n1_eratio_pt20.npz --plotdir perf_plots --name eratio_pho --plotlabel _eratio_pt20$CELLLAB --plottext "EM20 [H->yy]" --nobj 1 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Pho_*eEMRoISim*${CELLSEL}*_n2_eratio_pt20.npz --plotdir perf_plots --name eratio_pho --plotlabel _eratio_pt20$CELLLAB --plottext "2EM20 [H->yy]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Pho_*eEMRoISim*${CELLSEL}*_n1_eratio_rate50.npz --plotdir perf_plots --name eratio_pho --plotlabel _eratio_rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 1 --xmax 100. --noerr
    gep-perf plot perf_results/Pho_*eEMRoISim*${CELLSEL}*_n2_eratio_rate50.npz --plotdir perf_plots --name eratio_pho --plotlabel _eratio_rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 2 --xmax 100. --noerr
    
    echo "Barrel Phos"
    gep-perf plot perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n1_barrel_pt20.npz --plotdir perf_plots --name barrel_pho --plotlabel _barrel_pt20$CELLLAB --plottext "EM20 [H->yy]" --nobj 1 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n2_barrel_pt20.npz --plotdir perf_plots --name barrel_pho --plotlabel _barrel_pt20$CELLLAB --plottext "2EM20 [H->yy]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n1_barrel_rate50.npz --plotdir perf_plots --name barrel_pho --plotlabel _barrel_rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 1 --xmax 100. --noerr
    gep-perf plot perf_results/Pho_*{$CELLSEL*TowerJets,eEMRoISim}_n2_barrel_rate50.npz --plotdir perf_plots --name barrel_pho --plotlabel _barrel_rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 2 --xmax 100. --noerr
    
    echo "Barrel Eratio Phos"
    gep-perf plot perf_results/Pho_*eEMRoISim*${CELLSEL}*_n1_barrel_eratio_pt20.npz --plotdir perf_plots --name barrel_eratio_pho --plotlabel _barrel_eratio_pt20$CELLLAB --plottext "EM20 [H->yy]" --nobj 1 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Pho_*eEMRoISim*${CELLSEL}*_n2_barrel_eratio_pt20.npz --plotdir perf_plots --name barrel_eratio_pho --plotlabel _barrel_eratio_pt20$CELLLAB --plottext "2EM20 [H->yy]" --nobj 2 --xmax 100. --noerr --dofulleff
    gep-perf plot perf_results/Pho_*eEMRoISim*${CELLSEL}*_n1_barrel_eratio_rate50.npz --plotdir perf_plots --name barrel_eratio_pho --plotlabel _barrel_eratio_rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 1 --xmax 100. --noerr
    gep-perf plot perf_results/Pho_*eEMRoISim*${CELLSEL}*_n2_barrel_eratio_rate50.npz --plotdir perf_plots --name barrel_eratio_pho --plotlabel _barrel_eratio_rate50$CELLLAB --plottext "50 kHz [H->yy]" --nobj 2 --xmax 100. --noerr
fi

## MET
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "met" ]]; then
    echo "MET"
    gep-perf plot perf_results/MET_{*422,*Offline*Tower,*$CELLSEL*Tower}_n1_pt150.npz --plotdir perf_plots --name met --plotlabel _pt150$CELLLAB --plottext "MET150 [ZH->vvbb]" --nobj 1 --xmax 1000. --noerr --dofulleff --dorespresol
    gep-perf plot perf_results/MET_{*422,*Offline*Tower,*$CELLSEL*Tower}_n1_pt200.npz --plotdir perf_plots --name met --plotlabel _pt200$CELLLAB --plottext "MET200 [ZH->vvbb]" --nobj 1 --xmax 1000. --noerr --dofulleff
    gep-perf plot perf_results/MET_{*422,*Offline*Tower,*$CELLSEL*Tower}_n1_rate60.npz --plotdir perf_plots --name met --plotlabel _rate60$CELLLAB --plottext "60 kHz [ZH->vvbb]" --nobj 1 --xmax 1000. --noerr
    
    echo "SKMET"
    gep-perf plot perf_results/MET_*{Offline,$CELLSEL}*TowerSK_n1_pt150.npz --plotdir perf_plots --name skmet --plotlabel _pt150$CELLLAB --plottext "MET150 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff --dorespresol
    gep-perf plot perf_results/MET_*{Offline,$CELLSEL}*TowerSK_n1_pt200.npz --plotdir perf_plots --name skmet --plotlabel _pt200$CELLLAB --plottext "MET200 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff
    gep-perf plot perf_results/MET_*{Offline,$CELLSEL}*TowerSK_n1_rate60.npz --plotdir perf_plots --name skmet --plotlabel _rate60$CELLLAB --plottext "60 kHz [ZH->vvbb]" --nobj 1 --xmax 500. --noerr
fi

if [[ "$MAKETAR" == "true" ]]; then
    tar -cvz -f perf_plots.tar.gz perf_plots/
    tar -cvz -f perf_plots_eff_rate.tar.gz perf_plots/efficiency*rate* --transform s/perf_plots/perf_plots_eff_rate/
    tar -cvz -f perf_plots_eff_pt.tar.gz perf_plots/efficiency*pt* --transform s/perf_plots/perf_plots_eff_pt/
    tar -cvz -f perf_plots_respresol.tar.gz perf_plots/*{resp,resol}* --transform s/perf_plots/perf_plots_respresol/
fi
