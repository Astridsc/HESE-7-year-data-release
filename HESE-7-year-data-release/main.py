#from Astrid.data_processing import setup_paths
import numpy as np
#setup_paths()
from Astrid.plotting import plot_fig6, Fig10, plot_Fig8_data, plot_Fig8_mc, mockdata, test_smearing, test_binning, test_rebinning_old
from Astrid.plotting_new import fig6_new, fig8_new, test_morph, load_hese_data
from Astrid.effective_area import get_effective_area_dataframe, test_energy_smearing
from Astrid.HESE12 import plot_HESE12, plot_morph_hist
from Astrid.save_mc_histograms import get_mc_histogram

def main():
    # Initialize paths
    #setup_paths()

    # Call the desired plotting function
    #plot_Fig8_data()
    #plot_Fig8_mc()
    #test_binning()
    #mockdata()
    #Fig10()
    #test_smearing()
    #test_rebinning_old()
    #plot_HESE12()
    #test_energy_smearing()
    #plot_morph_hist()
    fig6_new()
    #load_hese_data()
    #fig8_new()
    #plot_fig6()
    #test_morph()
    #get_mc_histogram(gen2=True)
    
    #Edep = np.logspace(5, 7, num=40+1)
    #eff = get_effective_area_dataframe(Edep, gen2=True)
    #eff.to_csv('eff_Gen2.csv')
    
    

if __name__ == "__main__":
    main()