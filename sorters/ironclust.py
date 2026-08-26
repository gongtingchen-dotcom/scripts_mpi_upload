# IronClust sorter — an alternative to Kilosort4, run via a container
# (singularity/apptainer) rather than a local install. Run via run_IC.sh.

# Loads a preprocessed recording, runs IronClust, computes the standard
# analyzer extensions (waveforms/PCA/templates), and exports to Phy format.
# ironclust.ipynb is the interactive notebook this was developed in — kept
# for reference, this .py script is the one actually run.
#
# USAGE: no CLI — edit the config below directly, then `sbatch run_IC.sh`
# (or `python ironclust.py` on a GPU node with the container set up).
#
# PARAMETERS to change per session:
#   root_folder / data_folder   session dir and its preprocessed/ folder to sort.
#   exp_folder                  output folder name (default: root_folder/IC_test)
# IronClust's own sorter-specific parameters (detection threshold, template
# settings, etc.) are left at default here — run
# ss.get_sorter_params_description("ironclust") (already called below) to see
# the full list; pass overrides as extra keyword args to ss.run_sorter(...) if
# you need to tune them.
#
# OUTPUTS, under exp_folder (default root_folder/IC_test/):
#   sorter_output/         — IronClust's own raw output.
#   si_analyzer/            — SpikeInterface SortingAnalyzer (waveforms, PCA,
#                       templates) computed from the sort — cached, so
#                       reruns only compute whatever extensions are missing.
#   export_to_phy/          — Phy-format export for manual curation. NOTE:
#                       the export_to_phy function used here may have been
#                       deprecated in newer SpikeInterface — check before
#                       relying on this step if it errors.

# %%
import spikeinterface as si
import spikeinterface.sorters as ss
from tempfile import TemporaryDirectory
import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import os

# %%
ss.get_sorter_params_description("ironclust")

# %%
root_folder = '/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0'
data_folder = f'{root_folder}/preprocessed_slice_550_1270s'
exp_folder = os.path.join(root_folder, 'IC_test')
rec = si.load(data_folder, n_jobs = -1)
print("data loaded")
print(rec)

# %%
rec = spre.astype(rec, dtype="int16", round = False)

# %%
# ss.IronClustSorter.set_ironclust_path("/u/gongt/scripts/ironclust")
print("sorting...")
sorting = ss.run_sorter(
    sorter_name="ironclust", 
    recording=rec, 
    folder=exp_folder, 
    verbose=True,
    singularity_image=True)

# %%
def smart_compute(analyzer_save, sorting):
    if os.path.exists(analyzer_save):
        print("analyzer folder exists, loading extensions...")
        sorting_analyzer = si.load_sorting_analyzer(folder = analyzer_save, format = 'binary_folder')
    else: 
        print("analyzer folder does not exist, creating new analyzer...")
        sorting_analyzer = si.create_sorting_analyzer(sorting = sorting, recording = rec, format = 'binary_folder', folder = analyzer_save)
    
    existing_exts = sorting_analyzer.get_loaded_extension_names()
    print("already available:", existing_exts)

    #print("computing sparsity...")
    #sparsity = si.compute_sparsity(sorting_analyzer, method = 'closest_channels', num_channels = 48)
    #print("finished computing sparsity")

    compute_dict = {
    'random_spikes': {}, #random spike selection across units 
    'waveforms':{},
    'principal_components':{'n_components': 6, 'whiten':False},
    'templates':{
    }
    }

    to_compute = {
        ext: params for ext, params in compute_dict.items() if ext not in existing_exts
    }
    print("need to compute:", list(to_compute.keys()))

    if len(to_compute) > 0: 
        print("computing...")
        sorting_analyzer.compute(to_compute, save = True, progress_bar = True, n_jobs = 8)
    
    return sorting_analyzer 

# %%
print(sorting)

# %%
analyzer_save = os.path.join(os.path.join(root_folder,'IC_test'), 'si_analyzer')
sorting_analyzer = smart_compute(analyzer_save,  sorting) 

# %%
sorting_analyzer.compute('templates', save = True, progress_bar = True, n_jobs = -1)

# %%
from spikeinterface.exporters import export_to_phy
export_to_phy(sorting_analyzer=sorting_analyzer, 
              output_folder= os.path.join(exp_folder, 'export_to_phy'),
              copy_binary = False, 
              remove_if_exists = True,
              n_jobs = -1
              )


