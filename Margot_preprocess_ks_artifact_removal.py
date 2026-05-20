# just for this machine
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# %%
from pathlib import Path

# %%
import numpy as np
import matplotlib.pyplot as plt
import spikeinterface.full as si
from tqdm.auto import tqdm, trange

# %%

# %%
#data_dir = Path("~/data").expanduser()
#scratch_dir = Path("~/scratch").expanduser()

# %%
base_folder = Path(r"/ptmp/gongt/GLC7189/GLC7189_g_g1/")
spikeglx_folder = base_folder / 'GLC7189_g_g1_imec0'

rec_dir = spikeglx_folder #data_dir / "cuttlefish" / "GLC6962_g"
exp_dir = spikeglx_folder / 'kilosort4_si_artifact_removal' #data_dir / "cuttlefish" / "GLC6962_g_res"

# folder where a preprocessed copy of the recording will be saved
# scratch_dir is on a fast SSD on my computer. see the commment
# in the last cell before KS stuff below for more on this.
# ppx_dir = Path(r"E:\scratch\\") / "tmp_preprocessing"

# %%
si.set_global_job_kwargs(n_jobs=-1)

# %%
for meta_filename in Path(spikeglx_folder).glob("*ap.meta"):
    print(meta_filename)

word = 'imSampRate'
with open(meta_filename, 'r') as meta_info:
    # read all lines in a list
    lines = meta_info.readlines()
    for line in lines:
        # check if string present on a current line
        if line.find(word) != -1:
            index = lines.index(line)

line = lines[index]
fs = line[11:]
fs = float(fs)
print(fs)

# %% [markdown]
# ## Preprocessing

# %%
rec = si.read_spikeglx(spikeglx_folder, stream_name='imec0.ap', load_sync_channel=False)#si.read_spikeglx(base_folder)
rec.get_probe().to_dataframe()
rec

# %%
# preprocess
rec0 = si.read_spikeglx(rec_dir, stream_name='imec0.ap', load_sync_channel=False)  # to avoid stacking the ppx chain accidentally.

# %%
print("reading outlier list...")
outliers_list = np.loadtxt(os.path.join(rec_dir, "outliers_list_sqr_overview.txt"), delimiter=',')
#####

#print(eInkOff - eInkOn)
list_triggers = outliers_list
#print(list_triggers)
list_triggers_samp = list_triggers*fs
print(len(list_triggers))

ms_before = 150
ms_after = 150
rec_clean = si.remove_artifacts(recording=rec0, list_triggers=list_triggers_samp, mode='zeros',ms_before=ms_before,ms_after = ms_after)


# %% [markdown]
# ## ks thing

# %%
acg_threshold = 0.1
ccg_threshold = 0.05
tmin = 0
tmax = float('inf')

# %%
settings = si.get_default_sorter_params('kilosort4')
print(settings)

# %%
# then alter ks_params to: 
settings = si.get_default_sorter_params('kilosort4')

settings['delete_recording_dat'] = True
settings['do_CAR'] = True
settings['skip_kilosort_preprocessing'] = False
settings['do_correction'] = True
settings['invert_sign'] = False
#settings['NTbuff'] = 60122
#settings['n_chan_bin'] = 385
#settings['Nchan'] = 383
settings['torch_device'] = 'cuda'

# settings['data_dtype'] = 'int16'
# settings['fs'] = fs
settings['batch_size'] = 60000
settings['nblocks'] = 1
settings['Th_universal'] =  9.0
settings['Th_learned'] =  8.0
# settings['tmin'] =  tmin
# settings['tmax'] = tmax
settings['nt'] = 61
settings['shift'] = None
settings['scale'] = None
#settings['artifact_threshold'] = inf
settings['nskip'] = 25
settings['whitening_range'] = 32
settings['highpass_cutoff'] = 300.0
settings['binning_depth'] = 5.0
settings['sig_interp'] = 20.0
settings['drift_smoothing'] = [0.5, 0.5, 0.5]
settings['nt0min'] = 20
settings['dmin'] = None
settings['dminx'] = 32.0
settings['min_template_size'] = 16.0 # default = 10
settings['template_sizes'] = 8 # default = 5
settings['nearest_chans'] = 16 # default = 10
settings['nearest_templates'] = 100
# settings['max_channel_distance'] = None
settings['templates_from_data'] = True
settings['n_templates'] = 6
settings['n_pcs'] = 12
settings['Th_single_ch'] = 6.0
settings['acg_threshold'] = acg_threshold
settings['ccg_threshold'] = ccg_threshold
settings['cluster_downsampling'] = 10 # default 20
settings['x_centers'] = None
settings['duplicate_spike_ms'] = 0.25
settings['save_preprocessed_copy'] = False


# no need to do this, since we saved a copy with save_to_folder above
# but if that was not done, this would be good to do.
settings['save_preprocessed_copy'] = True #False CHANGED

# %%
sorting = si.run_sorter(
    'kilosort4',
    rec_clean, 
    folder=exp_dir / 'kilosort4_remove_artifact',
    docker_image=False,
    verbose=True,
    remove_existing_folder=True,
    **settings,
)


