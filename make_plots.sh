gep-perf plot perf_results/Jet_*n1_pt100.npz --plotdir perf_plots --name jet --plotlabel _pt100 --plottext "J100" --nobj 1 --xmax 200. --noerr --dofulleff --dorespresol
gep-perf plot perf_results/Jet_*n2_pt60.npz --plotdir perf_plots --name jet --plotlabel _pt60 --plottext "2J60" --nobj 2 --xmax 100. --noerr --dofulleff
gep-perf plot perf_results/Jet_*n3_pt50.npz --plotdir perf_plots --name jet --plotlabel _pt50 --plottext "3J50" --nobj 3 --xmax 100. --noerr --dofulleff
gep-perf plot perf_results/Jet_*n4_pt40.npz --plotdir perf_plots --name jet --plotlabel _pt40 --plottext "4J40" --nobj 4 --xmax 100. --noerr --dofulleff
gep-perf plot perf_results/Jet_*n1_rate50.npz --plotdir perf_plots --name jet --plotlabel _rate50 --plottext "50 kHz" --nobj 1 --xmax 200. --noerr
gep-perf plot perf_results/Jet_*n2_rate50.npz --plotdir perf_plots --name jet --plotlabel _rate50 --plottext "50 kHz" --nobj 2 --xmax 150. --noerr
gep-perf plot perf_results/Jet_*n3_rate75.npz --plotdir perf_plots --name jet --plotlabel _rate75 --plottext "75 kHz" --nobj 3 --xmax 150. --noerr
gep-perf plot perf_results/Jet_*n4_rate100.npz --plotdir perf_plots --name jet --plotlabel _rate100 --plottext "100 kHz" --nobj 4 --xmax 100. --noerr

gep-perf plot perf_results/Jet_*n1_boosted_pt100.npz --plotdir perf_plots --name boosted_jet --plotlabel _pt100 --plottext "J100 (Boosted)" --nobj 1 --xmax 200. --noerr --dofulleff
gep-perf plot perf_results/Jet_*n2_boosted_pt60.npz --plotdir perf_plots --name boosted_jet --plotlabel _pt60 --plottext "2J60 (Boosted)" --nobj 2 --xmax 100. --noerr --dofulleff
gep-perf plot perf_results/Jet_*n3_boosted_pt50.npz --plotdir perf_plots --name boosted_jet --plotlabel _pt50 --plottext "3J50 (Boosted)" --nobj 3 --xmax 100. --noerr --dofulleff
gep-perf plot perf_results/Jet_*n4_boosted_pt40.npz --plotdir perf_plots --name boosted_jet --plotlabel _pt40 --plottext "4J40 (Boosted)" --nobj 4 --xmax 100. --noerr --dofulleff
gep-perf plot perf_results/Jet_*n1_boosted_rate50.npz --plotdir perf_plots --name boosted_jet --plotlabel _rate50 --plottext "50 kHz (Boosted)" --nobj 1 --xmax 200. --noerr
gep-perf plot perf_results/Jet_*n2_boosted_rate50.npz --plotdir perf_plots --name boosted_jet --plotlabel _rate50 --plottext "50 kHz (Boosted)" --nobj 2 --xmax 150. --noerr
gep-perf plot perf_results/Jet_*n3_boosted_rate75.npz --plotdir perf_plots --name boosted_jet --plotlabel _rate75 --plottext "75 kHz (Boosted)" --nobj 3 --xmax 150. --noerr
gep-perf plot perf_results/Jet_*n4_boosted_rate100.npz --plotdir perf_plots --name boosted_jet --plotlabel _rate100 --plottext "100 kHz (Boosted)" --nobj 4 --xmax 100. --noerr

#perf_results/Jet_L1_jFexSRJetRoI_n1_boosted_pt100.npz   perf_results/Jet_L1_jFexSRJetRoI_n2_hhmass_pt50.npz     perf_results/Jet_L1_jFexSRJetRoI_n3_pt50.npz
#perf_results/Jet_L1_jFexSRJetRoI_n1_boosted_pt60.npz    perf_results/Jet_L1_jFexSRJetRoI_n2_hhmass_pt60.npz     perf_results/Jet_L1_jFexSRJetRoI_n3_pt90.npz
#perf_results/Jet_L1_jFexSRJetRoI_n1_boosted_rate50.npz  perf_results/Jet_L1_jFexSRJetRoI_n2_hhmass_rate50.npz   perf_results/Jet_L1_jFexSRJetRoI_n3_rate75.npz
#perf_results/Jet_L1_jFexSRJetRoI_n1_hhmass_pt100.npz    perf_results/Jet_L1_jFexSRJetRoI_n2_pt50.npz            perf_results/Jet_L1_jFexSRJetRoI_n4_boosted_pt40.npz
#perf_results/Jet_L1_jFexSRJetRoI_n1_hhmass_pt60.npz     perf_results/Jet_L1_jFexSRJetRoI_n2_pt60.npz            perf_results/Jet_L1_jFexSRJetRoI_n4_boosted_pt50.npz
#perf_results/Jet_L1_jFexSRJetRoI_n1_hhmass_rate50.npz   perf_results/Jet_L1_jFexSRJetRoI_n2_rate50.npz          perf_results/Jet_L1_jFexSRJetRoI_n4_boosted_rate100.npz
#perf_results/Jet_L1_jFexSRJetRoI_n1_pt100.npz           perf_results/Jet_L1_jFexSRJetRoI_n3_boosted_pt50.npz    perf_results/Jet_L1_jFexSRJetRoI_n4_hhmass_pt40.npz
#perf_results/Jet_L1_jFexSRJetRoI_n1_pt60.npz            perf_results/Jet_L1_jFexSRJetRoI_n3_boosted_pt90.npz    perf_results/Jet_L1_jFexSRJetRoI_n4_hhmass_pt50.npz
#perf_results/Jet_L1_jFexSRJetRoI_n1_rate50.npz          perf_results/Jet_L1_jFexSRJetRoI_n3_boosted_rate75.npz  perf_results/Jet_L1_jFexSRJetRoI_n4_hhmass_rate100.npz
#perf_results/Jet_L1_jFexSRJetRoI_n2_boosted_pt50.npz    perf_results/Jet_L1_jFexSRJetRoI_n3_hhmass_pt50.npz     perf_results/Jet_L1_jFexSRJetRoI_n4_pt40.npz
#perf_results/Jet_L1_jFexSRJetRoI_n2_boosted_pt60.npz    perf_results/Jet_L1_jFexSRJetRoI_n3_hhmass_pt90.npz     perf_results/Jet_L1_jFexSRJetRoI_n4_pt50.npz
#perf_results/Jet_L1_jFexSRJetRoI_n2_boosted_rate50.npz  perf_results/Jet_L1_jFexSRJetRoI_n3_hhmass_rate75.npz   perf_results/Jet_L1_jFexSRJetRoI_n4_rate100.npz
