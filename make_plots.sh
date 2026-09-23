#!/bin/bash
#
# Produces the overlay comparison plots from saved gep-perf results.
#
# GEPBase note: unlike the old GEPEnc production, there is no more GEP
# encoding-parameter scan (E6/E8/E10 x LSB50/100 x G2/4/6 x SIG2/3/4) --
# each algorithm now has exactly one nominal working point plus (where it
# exists) an EtaSK pileup-suppressed variant. There is no plain (non-EtaSK)
# "SK" variant at all any more, so the old skjet/sktau/skmet/vbf_skjet
# sections have been dropped rather than translated.

OBJTYPE=$1
MAKETAR=$2

PLOTDIR_BASE=perf_plots
PLOT_SUBDIR=""

plot_cmd() {
    mkdir -p "${PLOTDIR_BASE}/${PLOT_SUBDIR}"
    gep-perf plot "$@" --plotdir "${PLOTDIR_BASE}/${PLOT_SUBDIR}"
}

echo $OBJTYPE
echo $MAKETAR

## Jets
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "jet" ]]; then
    echo "Jets"
    PLOT_SUBDIR="jet"
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_pt100.npz --name jet --plotlabel _pt100 --plottext "J100 [VBF HH->4b]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_pt60.npz --name jet --plotlabel _pt60 --plottext "2J60 [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n3_pt50.npz --name jet --plotlabel _pt50 --plottext "3J50 [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n4_pt40.npz --name jet --plotlabel _pt40 --plottext "4J40 [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_rate50.npz --name jet --plotlabel _rate50 --plottext "50 kHz [VBF HH->4b]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_rate50.npz --name jet --plotlabel _rate50 --plottext "50 kHz [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n3_rate75.npz --name jet --plotlabel _rate75 --plottext "75 kHz [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n4_rate100.npz --name jet --plotlabel _rate100 --plottext "100 kHz [VBF HH->4b]" --nobj 4 --xmax 100. --noerr

    echo "Boosted Jets"
    PLOT_SUBDIR="boosted_jet"
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_boosted_pt100.npz --name boosted_jet --plotlabel _boosted_pt100 --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 1 --xmax 200. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_boosted_pt60.npz --name boosted_jet --plotlabel _boosted_pt60 --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n3_boosted_pt50.npz --name boosted_jet --plotlabel _boosted_pt50 --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n4_boosted_pt40.npz --name boosted_jet --plotlabel _boosted_pt40 --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_boosted_rate50.npz --name boosted_jet --plotlabel _boosted_rate50 --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_boosted_rate50.npz --name boosted_jet --plotlabel _boosted_rate50 --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n3_boosted_rate75.npz --name boosted_jet --plotlabel _boosted_rate75 --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n4_boosted_rate100.npz --name boosted_jet --plotlabel _boosted_rate100 --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr

    echo "Jets (TTbar)"
    PLOT_SUBDIR="jet_ttbar"
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_pt100.npz --name jet_ttbar --plotlabel _pt100 --plottext "J100 [TTbar]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_pt60.npz --name jet_ttbar --plotlabel _pt60 --plottext "2J60 [TTbar]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n3_pt50.npz --name jet_ttbar --plotlabel _pt50 --plottext "3J50 [TTbar]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n4_pt40.npz --name jet_ttbar --plotlabel _pt40 --plottext "4J40 [TTbar]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_rate50.npz --name jet_ttbar --plotlabel _rate50 --plottext "50 kHz [TTbar]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_rate50.npz --name jet_ttbar --plotlabel _rate50 --plottext "50 kHz [TTbar]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n3_rate75.npz --name jet_ttbar --plotlabel _rate75 --plottext "75 kHz [TTbar]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n4_rate100.npz --name jet_ttbar --plotlabel _rate100 --plottext "100 kHz [TTbar]" --nobj 4 --xmax 100. --noerr
fi


## Pileup-suppression jet comparison (nominal vs EtaSK; EMPFlow has no
## pileup-suppressed variant so only its nominal line is shown as a reference)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "pujet" ]]; then
    echo "Pileup-suppression Jets"
    PLOT_SUBDIR="pujet"
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_pt100.npz --name pujet --plotlabel _pt100 --plottext "J100 [VBF HH->4b]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_pt60.npz --name pujet --plotlabel _pt60 --plottext "2J60 [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_rate50.npz --name pujet --plotlabel _rate50 --plottext "50 kHz [VBF HH->4b]" --nobj 1 --xmax 200. --noerr
fi

## EtaSKJets
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "etaskjet" ]]; then
    echo "EtaSKJets"
    PLOT_SUBDIR="etaskjet"
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n1_pt100.npz --name etaskjet --plotlabel _pt100 --plottext "J100 [VBF HH->4b]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n2_pt60.npz --name etaskjet --plotlabel _pt60 --plottext "2J60 [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n3_pt50.npz --name etaskjet --plotlabel _pt50 --plottext "3J50 [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n4_pt40.npz --name etaskjet --plotlabel _pt40 --plottext "4J40 [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n1_rate50.npz --name etaskjet --plotlabel _rate50 --plottext "50 kHz [VBF HH->4b]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n2_rate50.npz --name etaskjet --plotlabel _rate50 --plottext "50 kHz [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n3_rate75.npz --name etaskjet --plotlabel _rate75 --plottext "75 kHz [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n4_rate100.npz --name etaskjet --plotlabel _rate100 --plottext "100 kHz [VBF HH->4b]" --nobj 4 --xmax 100. --noerr

    echo "Boosted EtaSKJets"
    PLOT_SUBDIR="boosted_etaskjet"
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n1_boosted_pt100.npz --name boosted_etaskjet --plotlabel _boosted_pt100 --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 1 --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n2_boosted_pt60.npz --name boosted_etaskjet --plotlabel _boosted_pt60 --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n3_boosted_pt50.npz --name boosted_etaskjet --plotlabel _boosted_pt50 --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n4_boosted_pt40.npz --name boosted_etaskjet --plotlabel _boosted_pt40 --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n1_boosted_rate50.npz --name boosted_etaskjet --plotlabel _boosted_rate50 --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 1 --noerr
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n2_boosted_rate50.npz --name boosted_etaskjet --plotlabel _boosted_rate50 --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n3_boosted_rate75.npz --name boosted_etaskjet --plotlabel _boosted_rate75 --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n4_boosted_rate100.npz --name boosted_etaskjet --plotlabel _boosted_rate100 --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 4 --xmax 100. --noerr

    echo "EtaSKJets (TTbar)"
    PLOT_SUBDIR="etaskjet_ttbar"
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n1_pt100.npz --name etaskjet_ttbar --plotlabel _pt100 --plottext "J100 [TTbar]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n2_pt60.npz --name etaskjet_ttbar --plotlabel _pt60 --plottext "2J60 [TTbar]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n3_pt50.npz --name etaskjet_ttbar --plotlabel _pt50 --plottext "3J50 [TTbar]" --nobj 3 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n4_pt40.npz --name etaskjet_ttbar --plotlabel _pt40 --plottext "4J40 [TTbar]" --nobj 4 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n1_rate50.npz --name etaskjet_ttbar --plotlabel _rate50 --plottext "50 kHz [TTbar]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n2_rate50.npz --name etaskjet_ttbar --plotlabel _rate50 --plottext "50 kHz [TTbar]" --nobj 2 --xmax 150. --noerr
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n3_rate75.npz --name etaskjet_ttbar --plotlabel _rate75 --plottext "75 kHz [TTbar]" --nobj 3 --xmax 150. --noerr
    plot_cmd perf_results/Jet_ttbar_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n4_rate100.npz --name etaskjet_ttbar --plotlabel _rate100 --plottext "100 kHz [TTbar]" --nobj 4 --xmax 100. --noerr
fi

## Jets (VBF, m_jj turn-on)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "jet" ]]; then
    echo "VBF Jets"
    PLOT_SUBDIR="vbf_jet"
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_pt100.npz --name jet --plotlabel _mjj_pt100 --plottext "J100 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_pt60.npz --name jet --plotlabel _mjj_pt60 --plottext "2J60 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n3_pt50.npz --name jet --plotlabel _mjj_pt50 --plottext "3J50 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n4_pt40.npz --name jet --plotlabel _mjj_pt40 --plottext "4J40 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_rate50.npz --name jet --plotlabel _mjj_rate50 --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_rate50.npz --name jet --plotlabel _mjj_rate50 --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n3_rate75.npz --name jet --plotlabel _mjj_rate75 --plottext "75 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n4_rate100.npz --name jet --plotlabel _mjj_rate100 --plottext "100 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1

    echo "VBF Boosted Jets"
    PLOT_SUBDIR="vbf_boosted_jet"
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_boosted_pt100.npz --name boosted_jet --plotlabel _mjj_boosted_pt100 --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_boosted_pt60.npz --name boosted_jet --plotlabel _mjj_boosted_pt60 --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n3_boosted_pt50.npz --name boosted_jet --plotlabel _mjj_boosted_pt50 --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n4_boosted_pt40.npz --name boosted_jet --plotlabel _mjj_boosted_pt40 --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_boosted_rate50.npz --name boosted_jet --plotlabel _mjj_boosted_rate50 --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_boosted_rate50.npz --name boosted_jet --plotlabel _mjj_boosted_rate50 --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n3_boosted_rate75.npz --name boosted_jet --plotlabel _mjj_boosted_rate75 --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n4_boosted_rate100.npz --name boosted_jet --plotlabel _mjj_boosted_rate100 --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
fi

## EtaSKJets (VBF, m_jj turn-on)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "etaskjet" ]]; then
    echo "VBF EtaSKJets"
    PLOT_SUBDIR="vbf_etaskjet"
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n1_pt100.npz --name etaskjet --plotlabel _mjj_pt100 --plottext "J100 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n2_pt60.npz --name etaskjet --plotlabel _mjj_pt60 --plottext "2J60 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n3_pt50.npz --name etaskjet --plotlabel _mjj_pt50 --plottext "3J50 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n4_pt40.npz --name etaskjet --plotlabel _mjj_pt40 --plottext "4J40 [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n1_rate50.npz --name etaskjet --plotlabel _mjj_rate50 --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n2_rate50.npz --name etaskjet --plotlabel _mjj_rate50 --plottext "50 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n3_rate75.npz --name etaskjet --plotlabel _mjj_rate75 --plottext "75 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n4_rate100.npz --name etaskjet --plotlabel _mjj_rate100 --plottext "100 kHz [VBF HH->4b]" --nobj 0 --noerr --xmax -1

    echo "VBF Boosted EtaSKJets"
    PLOT_SUBDIR="vbf_boosted_etaskjet"
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n1_boosted_pt100.npz --name boosted_etaskjet --plotlabel _mjj_boosted_pt100 --plottext "J100 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n2_boosted_pt60.npz --name boosted_etaskjet --plotlabel _mjj_boosted_pt60 --plottext "2J60 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n3_boosted_pt50.npz --name boosted_etaskjet --plotlabel _mjj_boosted_pt50 --plottext "3J50 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n4_boosted_pt40.npz --name boosted_etaskjet --plotlabel _mjj_boosted_pt40 --plottext "4J40 (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n1_boosted_rate50.npz --name boosted_etaskjet --plotlabel _mjj_boosted_rate50 --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n2_boosted_rate50.npz --name boosted_etaskjet --plotlabel _mjj_boosted_rate50 --plottext "50 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n3_boosted_rate75.npz --name boosted_etaskjet --plotlabel _mjj_boosted_rate75 --plottext "75 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_dijet_mass_{AntiKt4CaloTopoClusters422EtaSKAlgJets,AntiKt4GEPCellsTowerEtaSKAlgJets}_n4_boosted_rate100.npz --name boosted_etaskjet --plotlabel _mjj_boosted_rate100 --plottext "100 kHz (Boosted) [VBF HH->4b]" --nobj 0 --noerr --xmax -1
fi

## Jets (Z'->jj dijet resonance) -- new for GEPBase, no VBF/HH-specific selector
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "zprimejet" ]]; then
    echo "Z' Jets"
    PLOT_SUBDIR="zprime_jet"
    plot_cmd perf_results/Jet_zprime_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_pt100.npz --name jet_zprime --plotlabel _pt100 --plottext "J100 [Z'->jj]" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Jet_zprime_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_pt60.npz --name jet_zprime --plotlabel _pt60 --plottext "2J60 [Z'->jj]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Jet_zprime_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_rate50.npz --name jet_zprime --plotlabel _rate50 --plottext "50 kHz [Z'->jj]" --nobj 1 --xmax 200. --noerr
    plot_cmd perf_results/Jet_zprime_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_rate50.npz --name jet_zprime --plotlabel _rate50 --plottext "50 kHz [Z'->jj]" --nobj 2 --xmax 150. --noerr

    echo "VBF-style m_jj (Z')"
    PLOT_SUBDIR="zprime_vbf_jet"
    plot_cmd perf_results/Jet_zprime_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_pt100.npz --name jet_zprime --plotlabel _mjj_pt100 --plottext "J100 [Z'->jj]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_zprime_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_pt60.npz --name jet_zprime --plotlabel _mjj_pt60 --plottext "2J60 [Z'->jj]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_zprime_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n1_rate50.npz --name jet_zprime --plotlabel _mjj_rate50 --plottext "50 kHz [Z'->jj]" --nobj 0 --noerr --xmax -1
    plot_cmd perf_results/Jet_zprime_dijet_mass_{AntiKt4CaloTopoClusters422AlgJets,AntiKt4GEPCellsTowerAlgJets,AntiKt4EMPFlowJets,L1_jFexSRJetRoISim,L1_gFexSRJetRoISim}_n2_rate50.npz --name jet_zprime --plotlabel _mjj_rate50 --plottext "50 kHz [Z'->jj]" --nobj 0 --noerr --xmax -1
fi


## Taus (AK1 nominal + eTau/jFex RoI + new offline reco_tau)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "tau" ]]; then
    echo "Taus"
    PLOT_SUBDIR="tau"
    plot_cmd perf_results/Tau_{AntiKt1CaloTopoClusters422AlgJets,AntiKt1GEPCellsTowerAlgJets,L1_eTauRoISim,L1_jFexSRJetRoISim,reco_tau}_n1_pt150.npz --name tau --plotlabel _pt150 --plottext "Tau150 [y*->tt]" --nobj 1 --xmax 230. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Tau_{AntiKt1CaloTopoClusters422AlgJets,AntiKt1GEPCellsTowerAlgJets,L1_eTauRoISim,L1_jFexSRJetRoISim,reco_tau}_n2_pt40.npz --name tau --plotlabel _pt40 --plottext "2Tau40 [y*->tt]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Tau_{AntiKt1CaloTopoClusters422AlgJets,AntiKt1GEPCellsTowerAlgJets,L1_eTauRoISim,L1_jFexSRJetRoISim,reco_tau}_n1_rate10.npz --name tau --plotlabel _rate10 --plottext "10 kHz [y*->tt]" --nobj 1 --xmax 230. --noerr
    plot_cmd perf_results/Tau_{AntiKt1CaloTopoClusters422AlgJets,AntiKt1GEPCellsTowerAlgJets,L1_eTauRoISim,L1_jFexSRJetRoISim,reco_tau}_n2_rate200.npz --name tau --plotlabel _rate200 --plottext "200 kHz [y*->tt]" --nobj 2 --xmax 100. --noerr
fi


## Pileup-suppression tau comparison (nominal vs EtaSK)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "putau" ]]; then
    echo "Pileup-suppression Taus"
    PLOT_SUBDIR="putau"
    plot_cmd perf_results/Tau_{AntiKt1CaloTopoClusters422AlgJets,AntiKt1CaloTopoClusters422EtaSKAlgJets,AntiKt1GEPCellsTowerAlgJets,AntiKt1GEPCellsTowerEtaSKAlgJets,L1_eTauRoISim,L1_jFexSRJetRoISim,reco_tau}_n1_pt150.npz --name putau --plotlabel _pt150 --plottext "Tau150 [y*->tt]" --nobj 1 --xmax 250. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Tau_{AntiKt1CaloTopoClusters422AlgJets,AntiKt1CaloTopoClusters422EtaSKAlgJets,AntiKt1GEPCellsTowerAlgJets,AntiKt1GEPCellsTowerEtaSKAlgJets,L1_eTauRoISim,L1_jFexSRJetRoISim,reco_tau}_n2_pt40.npz --name putau --plotlabel _pt40 --plottext "2Tau40 [y*->tt]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Tau_{AntiKt1CaloTopoClusters422AlgJets,AntiKt1CaloTopoClusters422EtaSKAlgJets,AntiKt1GEPCellsTowerAlgJets,AntiKt1GEPCellsTowerEtaSKAlgJets,L1_eTauRoISim,L1_jFexSRJetRoISim,reco_tau}_n1_rate10.npz --name putau --plotlabel _rate10 --plottext "10 kHz [y*->tt]" --nobj 1 --xmax 250. --noerr
    plot_cmd perf_results/Tau_{AntiKt1CaloTopoClusters422AlgJets,AntiKt1CaloTopoClusters422EtaSKAlgJets,AntiKt1GEPCellsTowerAlgJets,AntiKt1GEPCellsTowerEtaSKAlgJets,L1_eTauRoISim,L1_jFexSRJetRoISim,reco_tau}_n2_rate200.npz --name putau --plotlabel _rate200 --plottext "200 kHz [y*->tt]" --nobj 2 --xmax 100. --noerr
fi

## EtaSKTaus
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "etasktau" ]]; then
    echo "EtaSKTaus"
    PLOT_SUBDIR="etasktau"
    plot_cmd perf_results/Tau_{AntiKt1CaloTopoClusters422EtaSKAlgJets,AntiKt1GEPCellsTowerEtaSKAlgJets}_n1_pt150.npz --name etasktau --plotlabel _pt150 --plottext "Tau150 [y*->tt]" --nobj 1 --xmax 250. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Tau_{AntiKt1CaloTopoClusters422EtaSKAlgJets,AntiKt1GEPCellsTowerEtaSKAlgJets}_n2_pt40.npz --name etasktau --plotlabel _pt40 --plottext "2Tau40 [y*->tt]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Tau_{AntiKt1CaloTopoClusters422EtaSKAlgJets,AntiKt1GEPCellsTowerEtaSKAlgJets}_n1_rate10.npz --name etasktau --plotlabel _rate10 --plottext "10 kHz [y*->tt]" --nobj 1 --xmax 250. --noerr
    plot_cmd perf_results/Tau_{AntiKt1CaloTopoClusters422EtaSKAlgJets,AntiKt1GEPCellsTowerEtaSKAlgJets}_n2_rate200.npz --name etasktau --plotlabel _rate200 --plottext "200 kHz [y*->tt]" --nobj 2 --xmax 100. --noerr
fi

## Eles (AK1 nominal + eEM RoI + new offline Electrons/ForwardElectrons)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "ele" ]]; then
    echo "Eles"
    PLOT_SUBDIR="ele"
    plot_cmd perf_results/Ele_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Electrons,ForwardElectrons}_n1_pt20.npz --name ele --plotlabel _pt20 --plottext "EM20 [Z->ee]" --nobj 1 --xmax 100. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Ele_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Electrons,ForwardElectrons}_n2_pt20.npz --name ele --plotlabel _pt20 --plottext "2EM20 [Z->ee]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Electrons,ForwardElectrons}_n1_rate50.npz --name ele --plotlabel _rate50 --plottext "50 kHz [Z->ee]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Ele_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Electrons,ForwardElectrons}_n2_rate50.npz --name ele --plotlabel _rate50 --plottext "50 kHz [Z->ee]" --nobj 2 --xmax 100. --noerr

    echo "Eratio Eles"
    PLOT_SUBDIR="eratio_ele"
    plot_cmd perf_results/Ele_L1_eEMRoISim_n1_eratio_pt20.npz --name eratio_ele --plotlabel _eratio_pt20 --plottext "EM20 [Z->ee]" --nobj 1 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_L1_eEMRoISim_n2_eratio_pt20.npz --name eratio_ele --plotlabel _eratio_pt20 --plottext "2EM20 [Z->ee]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_L1_eEMRoISim_n1_eratio_rate50.npz --name eratio_ele --plotlabel _eratio_rate50 --plottext "50 kHz [Z->ee]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Ele_L1_eEMRoISim_n2_eratio_rate50.npz --name eratio_ele --plotlabel _eratio_rate50 --plottext "50 kHz [Z->ee]" --nobj 2 --xmax 100. --noerr

    echo "Barrel Eles"
    PLOT_SUBDIR="barrel_ele"
    plot_cmd perf_results/Ele_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Electrons,ForwardElectrons}_n1_barrel_pt20.npz --name barrel_ele --plotlabel _barrel_pt20 --plottext "EM20 [Z->ee]" --nobj 1 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Electrons,ForwardElectrons}_n2_barrel_pt20.npz --name barrel_ele --plotlabel _barrel_pt20 --plottext "2EM20 [Z->ee]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Electrons,ForwardElectrons}_n1_barrel_rate50.npz --name barrel_ele --plotlabel _barrel_rate50 --plottext "50 kHz [Z->ee]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Ele_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Electrons,ForwardElectrons}_n2_barrel_rate50.npz --name barrel_ele --plotlabel _barrel_rate50 --plottext "50 kHz [Z->ee]" --nobj 2 --xmax 100. --noerr

    echo "Barrel Eratio Eles"
    PLOT_SUBDIR="barrel_eratio_ele"
    plot_cmd perf_results/Ele_L1_eEMRoISim_n1_barrel_eratio_pt20.npz --name barrel_eratio_ele --plotlabel _barrel_eratio_pt20 --plottext "EM20 [Z->ee]" --nobj 1 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_L1_eEMRoISim_n2_barrel_eratio_pt20.npz --name barrel_eratio_ele --plotlabel _barrel_eratio_pt20 --plottext "2EM20 [Z->ee]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Ele_L1_eEMRoISim_n1_barrel_eratio_rate50.npz --name barrel_eratio_ele --plotlabel _barrel_eratio_rate50 --plottext "50 kHz [Z->ee]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Ele_L1_eEMRoISim_n2_barrel_eratio_rate50.npz --name barrel_eratio_ele --plotlabel _barrel_eratio_rate50 --plottext "50 kHz [Z->ee]" --nobj 2 --xmax 100. --noerr
fi

## Phos (AK1 nominal + eEM RoI + new offline Photons)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "pho" ]]; then
    echo "Phos"
    PLOT_SUBDIR="pho"
    plot_cmd perf_results/Pho_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Photons}_n1_pt20.npz --name pho --plotlabel _pt20 --plottext "EM20 [H->yy]" --nobj 1 --xmax 100. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/Pho_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Photons}_n2_pt20.npz --name pho --plotlabel _pt20 --plottext "2EM20 [H->yy]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Photons}_n1_rate50.npz --name pho --plotlabel _rate50 --plottext "50 kHz [H->yy]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Pho_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Photons}_n2_rate50.npz --name pho --plotlabel _rate50 --plottext "50 kHz [H->yy]" --nobj 2 --xmax 100. --noerr

    echo "Eratio Phos"
    PLOT_SUBDIR="eratio_pho"
    plot_cmd perf_results/Pho_L1_eEMRoISim_n1_eratio_pt20.npz --name eratio_pho --plotlabel _eratio_pt20 --plottext "EM20 [H->yy]" --nobj 1 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_L1_eEMRoISim_n2_eratio_pt20.npz --name eratio_pho --plotlabel _eratio_pt20 --plottext "2EM20 [H->yy]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_L1_eEMRoISim_n1_eratio_rate50.npz --name eratio_pho --plotlabel _eratio_rate50 --plottext "50 kHz [H->yy]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Pho_L1_eEMRoISim_n2_eratio_rate50.npz --name eratio_pho --plotlabel _eratio_rate50 --plottext "50 kHz [H->yy]" --nobj 2 --xmax 100. --noerr

    echo "Barrel Phos"
    PLOT_SUBDIR="barrel_pho"
    plot_cmd perf_results/Pho_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Photons}_n1_barrel_pt20.npz --name barrel_pho --plotlabel _barrel_pt20 --plottext "EM20 [H->yy]" --nobj 1 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Photons}_n2_barrel_pt20.npz --name barrel_pho --plotlabel _barrel_pt20 --plottext "2EM20 [H->yy]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Photons}_n1_barrel_rate50.npz --name barrel_pho --plotlabel _barrel_rate50 --plottext "50 kHz [H->yy]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Pho_{AntiKt1GEPCellsTowerAlgJets,L1_eEMRoISim,Photons}_n2_barrel_rate50.npz --name barrel_pho --plotlabel _barrel_rate50 --plottext "50 kHz [H->yy]" --nobj 2 --xmax 100. --noerr

    echo "Barrel Eratio Phos"
    PLOT_SUBDIR="barrel_eratio_pho"
    plot_cmd perf_results/Pho_L1_eEMRoISim_n1_barrel_eratio_pt20.npz --name barrel_eratio_pho --plotlabel _barrel_eratio_pt20 --plottext "EM20 [H->yy]" --nobj 1 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_L1_eEMRoISim_n2_barrel_eratio_pt20.npz --name barrel_eratio_pho --plotlabel _barrel_eratio_pt20 --plottext "2EM20 [H->yy]" --nobj 2 --xmax 100. --noerr --dofulleff
    plot_cmd perf_results/Pho_L1_eEMRoISim_n1_barrel_eratio_rate50.npz --name barrel_eratio_pho --plotlabel _barrel_eratio_rate50 --plottext "50 kHz [H->yy]" --nobj 1 --xmax 100. --noerr
    plot_cmd perf_results/Pho_L1_eEMRoISim_n2_barrel_eratio_rate50.npz --name barrel_eratio_pho --plotlabel _barrel_eratio_rate50 --plottext "50 kHz [H->yy]" --nobj 2 --xmax 100. --noerr
fi


## Pileup-suppression MET comparison (nominal vs EtaSK, plus gFEX/jFEX single-object RoIs)
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "pumet" ]]; then
    echo "Pileup-suppression MET"
    PLOT_SUBDIR="pumet"
    plot_cmd perf_results/MET_{gFex_JwoJ,gFex_RhoRMS,gFex_NC,jFex,GepMETCaloTopoClusters422,GepMETCaloTopoClusters422EtaSK,GepMETGEPCellsTower,GepMETGEPCellsTowerEtaSK}_n1_pt150.npz --name pumet --plotlabel _pt150 --plottext "MET150 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/MET_{gFex_JwoJ,gFex_RhoRMS,gFex_NC,jFex,GepMETCaloTopoClusters422,GepMETCaloTopoClusters422EtaSK,GepMETGEPCellsTower,GepMETGEPCellsTowerEtaSK}_n1_pt200.npz --name pumet --plotlabel _pt200 --plottext "MET200 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff
    plot_cmd perf_results/MET_{gFex_JwoJ,gFex_RhoRMS,gFex_NC,jFex,GepMETCaloTopoClusters422,GepMETCaloTopoClusters422EtaSK,GepMETGEPCellsTower,GepMETGEPCellsTowerEtaSK}_n1_rate60.npz --name pumet --plotlabel _rate60 --plottext "60 kHz [ZH->vvbb]" --nobj 1 --xmax 500. --noerr
fi

## MET
if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "met" ]]; then
    echo "MET"
    PLOT_SUBDIR="met"
    plot_cmd perf_results/MET_{gFex_JwoJ,gFex_RhoRMS,gFex_NC,jFex,GepMETCaloTopoClusters422,GepMETGEPCellsTower}_n1_pt150.npz --name met --plotlabel _pt150 --plottext "MET150 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/MET_{gFex_JwoJ,gFex_RhoRMS,gFex_NC,jFex,GepMETCaloTopoClusters422,GepMETGEPCellsTower}_n1_pt200.npz --name met --plotlabel _pt200 --plottext "MET200 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff
    plot_cmd perf_results/MET_{gFex_JwoJ,gFex_RhoRMS,gFex_NC,jFex,GepMETCaloTopoClusters422,GepMETGEPCellsTower}_n1_rate60.npz --name met --plotlabel _rate60 --plottext "60 kHz [ZH->vvbb]" --nobj 1 --xmax 500. --noerr
fi

if [[ "$OBJTYPE" == "all" ]] || [[ "$OBJTYPE" == "etaskmet" ]]; then
    echo "EtaSKMET"
    PLOT_SUBDIR="etaskmet"
    plot_cmd perf_results/MET_{GepMETCaloTopoClusters422EtaSK,GepMETGEPCellsTowerEtaSK}_n1_pt150.npz --name etaskmet --plotlabel _pt150 --plottext "MET150 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff --dorespresol
    plot_cmd perf_results/MET_{GepMETCaloTopoClusters422EtaSK,GepMETGEPCellsTowerEtaSK}_n1_pt200.npz --name etaskmet --plotlabel _pt200 --plottext "MET200 [ZH->vvbb]" --nobj 1 --xmax 500. --noerr --dofulleff
    plot_cmd perf_results/MET_{GepMETCaloTopoClusters422EtaSK,GepMETGEPCellsTowerEtaSK}_n1_rate60.npz --name etaskmet --plotlabel _rate60 --plottext "60 kHz [ZH->vvbb]" --nobj 1 --xmax 500. --noerr
fi


# Stage a subset of files under a renamed top-level directory, then tar it.
# (Avoids GNU tar's --transform, which BSD tar -- the macOS default -- does
# not support with the same syntax.)
stage_and_tar() {
    local archive="$1" topdir="$2"; shift 2
    local files=("$@")
    (( ${#files[@]} > 0 )) || return 0
    local stage
    stage=$(mktemp -d)
    for f in "${files[@]}"; do
        local rel="${f#${PLOTDIR_BASE}/}"
        mkdir -p "${stage}/${topdir}/$(dirname "$rel")"
        cp "$f" "${stage}/${topdir}/${rel}"
    done
    tar -czf "$archive" -C "$stage" "$topdir"
    rm -rf "$stage"
}

if [[ "$MAKETAR" == "true" ]]; then
    tar -cvz -f perf_plots.tar.gz "${PLOTDIR_BASE}/"

    eff_rate_files=()
    while IFS= read -r f; do eff_rate_files+=("$f"); done < <(find "${PLOTDIR_BASE}" -type f -name "*efficiency*rate*" -print)
    stage_and_tar perf_plots_eff_rate.tar.gz perf_plots_eff_rate "${eff_rate_files[@]}"

    eff_pt_files=()
    while IFS= read -r f; do eff_pt_files+=("$f"); done < <(find "${PLOTDIR_BASE}" -type f -name "*efficiency*pt*" -print)
    stage_and_tar perf_plots_eff_pt.tar.gz perf_plots_eff_pt "${eff_pt_files[@]}"

    resp_resol_files=()
    while IFS= read -r f; do resp_resol_files+=("$f"); done < <(find "${PLOTDIR_BASE}" -type f \( -name "*resp*" -o -name "*resol*" \) -print)
    stage_and_tar perf_plots_respresol.tar.gz perf_plots_respresol "${resp_resol_files[@]}"
fi
