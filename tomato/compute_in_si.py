# %%
## IMPORT LIBRARIES 
import os 
import numpy as np
import spikeinterface.core as si 
import spikeinterface.widgets as sw
import spikeinterface.extractors as se
from spikeinterface.extractors import read_phy
from spikeinterface.preprocessing import common_reference
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

# %%
### PARAMETERS ### 
root_folder = r"/nexus/posix0/MBR-neuralsystems/ephys/GLC7189/GLC7189_c_g0" 
exp_folder = os.path.join(root_folder, 'KS4', 'MG_nc10_wr32_thu9_thl8_maxcd32')
sorter_output = os.path.join(exp_folder, 'sorter_output')
preprocessed_folder = os.path.join(root_folder, 'preprocessed')
analyzer_save = os.path.join(exp_folder, 'si_analyzer')

preprocessed_rec = si.load(preprocessed_folder)
preprocessed_rec

fs = preprocessed_rec.get_sampling_frequency()
nchan = preprocessed_rec.get_num_channels()

ks_preprocessed_file = os.path.join(sorter_output, "temp_wh.dat")
ks_preprocessed_rec = si.load_extractor(preprocessed_folder)

sorting = read_phy(sorter_output)

# %% [markdown]
# ## SPIKE OVER TRACE

# %%
spike_clusters = np.load(os.path.join(sorter_output, 'spike_clusters.npy'))
#spike_positions = np.load(os.path.join(sorter_output, 'spike_positions.npy'))
spike_templates = np.load(os.path.join(sorter_output,'spike_templates.npy'))
templates = np.load(os.path.join(sorter_output,'templates.npy'))
channel_positions = np.load(os.path.join(sorter_output,'channel_positions.npy'))
channel_map = np.load(os.path.join(sorter_output,'channel_map.npy')) 
spike_times = np.load(os.path.join(sorter_output, 'spike_times.npy'))
#ops = np.load(os.path.join(sorter_output, 'ops.npy'), allow_pickle = True)
#fs = ops.item()['fs']
pc_features = np.load(os.path.join(sorter_output, 'pc_features.npy'))  #shape: n_spikes * n_pcs * nearest_chans
pc_feature_ind = np.load(os.path.join(sorter_output, 'pc_feature_ind.npy')) #shape: 145 * nearest_chans


# %%
print(spike_clusters.shape)
print(len(pc_features[0][0]))
print(len(pc_feature_ind[0]))

# %%
def smart_compute(analyzer_save, compute_dict, sorting):
    '''
    if os.path.exists(analyzer_save):
        print("analyzer folder exists, loading extensions...")
        sorting_analyzer = si.load_sorting_analyzer(folder = analyzer_save, format = 'binary_folder', sparse=False)
    else: 
        print("analyzer folder does not exist, creating new analyzer...")
        sorting_analyzer = si.create_sorting_analyzer(sorting = sorting, recording = preprocessed_rec, format = 'binary_folder', folder = analyzer_save, sparse=False)
    '''
    print("creating analyzer folder...")
    sorting_analyzer = si.create_sorting_analyzer(sorting = sorting, recording = preprocessed_rec, format = 'binary_folder', folder = analyzer_save, sparse=False)

    existing_exts = sorting_analyzer.get_loaded_extension_names()
    print("already available:", existing_exts)

    to_compute = {
        ext: params for ext, params in compute_dict.items() if ext not in existing_exts
    }
    print("need to compute:", list(to_compute.keys()))

    if len(to_compute) > 0: 
        print("computing...")
        sorting_analyzer.compute(to_compute, save = True, progress_bar = True, n_jobs = 4)
    
    return sorting_analyzer 

# %%
compute_dict = {
    'random_spikes': {}, 
    'waveforms': {},
    'templates': {},
    'noise_levels': {},
    'spike_amplitudes': {},
    'unit_locations': {'method': 'monopolar_triangulation'},
    'spike_locations': {'method': 'center_of_mass'},
    'template_similarity': {}, 
    'quality_metrics': {
        'metric_names': ['snr', 'isi_violation', 'presence_ratio', 'amplitude_cutoff']
    },
    'principal_components': {'n_components': 6, 'whiten': False},
}

sorting_analyzer = smart_compute(analyzer_save, compute_dict, sorting) 

# %%
from spikeinterface.exporters import export_to_phy
export_to_phy(sorting_analyzer=sorting_analyzer, 
              output_folder= os.path.join(exp_folder, 'export_to_phy'),
              copy_binary = False, 
              remove_if_exists = True,
              n_jobs = 4
              )
