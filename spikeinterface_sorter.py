#!/usr/bin/env python3

# %%
import tqdm
import os
import numpy as np
import matplotlib.pyplot as plta
import spikeinterface.full as si
import scipy.signal
from tqdm.auto import tqdm, trange
import urllib.request
import spikeinterface.sorters as ss
import argparse

# %%
parser = argparse.ArgumentParser(description="Run Kilosort4 with custom parameters")
parser.add_argument("--nearest_chans", type=int, default=10,
    help="Number of nearest channels used by Kilosort4")
parser.add_argument("--whitening_range", type=int, default=0,
    help="Whitening range (0 disables whitening)")
parser.add_argument("--Th_universal", type=int, default=9,
    help="Universal threshold (for detecting universal templates)")
parser.add_argument("--Th_learned", type=int, default=8,
    help="Learned threshold (for detecting learned templates)")
parser.add_argument("--max_channel_distance", type=int, default=32,
    help="Maximum channel distance for template matching")
args = parser.parse_args()

root_folder = '/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_g_g0'
#root_folder = '/ptmp/gongt/GLC6962_g_g0'
data_folder = os.path.join(root_folder, 'preprocessed_slice_550_1270s')
#data_folder = '/ptmp/gongt/GLC6962_g_g0/preprocessed'
output_folder_name = f"nc{args.nearest_chans}_wr{args.whitening_range}_thu{args.Th_universal}_thl{args.Th_learned}_maxcd{args.max_channel_distance}"
#output_folder_name = 'test_MG_params2'
output_folder = f"{root_folder}/KS4/{output_folder_name}_short"
#output_folder = '/nexus/posix0/MBR-neuralsystems/ephys/GLC6962/GLC6962_l_g1/KS4/running_from_ptmp'
params_file = os.path.join(output_folder, "kilosort_params.txt")

if "preprocessed" in data_folder: 
    rec = si.load_extractor(data_folder)
if "imec" in data_folder: 
    rec = si.read_spikeglx(data_folder, stream_name='imec0.ap', load_sync_channel=False) 
print(rec)

'''
fs = rec.get_sampling_frequency()
rec = rec.frame_slice(start_frame=550*fs, end_frame=1270*fs)
print(rec)
'''

# %%
ks_params = si.get_default_sorter_params('kilosort4')
ks_params['nblocks'] = 0
ks_params['Th_learned']=args.Th_learned
ks_params['Th_universal']=args.Th_universal
ks_params['whitening_range'] = args.whitening_range
#ks_params['nt0min'] = 20
ks_params['nearest_chans'] = args.nearest_chans
ks_params['max_channel_distance']=args.max_channel_distance
ks_params['templates_from_data'] = True
ks_params['n_pcs'] = 12
#ks_params['acg_threshold']=1.0
#ks_params['ccg_threshold']=1.5
#ks_params['cluster_downsampling'] = 20
#ks_params['max_cluster_subset']=np.inf
ks_params['save_extra_vars']=False
ks_params['delete_recording_dat']=True
ks_params['save_preprocessed_copy']=False
ks_params['do_correction'] = False
ks_params['do_CAR'] = False
ks_params['n_jobs']= -1
ks_params['max_threads_per_worker']=4
if args.whitening_range == 0: 
    ks_params['skip_kilosort_preprocessing'] = True
    ks_params['highpass_cutoff'] = 0
else: 
    ks_params['skip_kilosort_preprocessing'] = False
    ks_params['highpass_cutoff'] = 300



from datetime import datetime

if os.path.exists(output_folder):
    import shutil
    shutil.rmtree(output_folder) 

sorting = si.run_sorter(
    'kilosort4',
    rec,
    folder=output_folder,
    docker_image=False,
    verbose=True,
    **ks_params,
)

with open(params_file, "w") as f:
    f.write(f"Kilosort parameters saved on {datetime.now()}\n")
    f.write(f"SpikeInterface version: {si.__version__}\n\n")
    f.write("Parameters:\n")
    for key, value in ks_params.items():
        f.write(f"{key}: {value}\n")
print(f"Full Kilosort parameters saved to: {params_file}") 