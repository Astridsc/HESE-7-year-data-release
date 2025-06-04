#from Astrid.data_processing import setup_paths

#setup_paths()
from Astrid.plotting import plot_fig6, plot_fig8, Fig10, plot_Fig8, mockdata, test_smearing, test_binning, test_rebinning_old

#from Astrid.HESE12 import plot_Fig8_HESE12

def main():
    # Initialize paths
    #setup_paths()

    # Call the desired plotting function
    plot_Fig8()
    #test_binning()
    #plot_fig6()
    #mockdata()
    #Fig10()
    #test_smearing()
    #test_rebinning_old()
    #plot_Fig8_HESE12()

if __name__ == "__main__":
    main()