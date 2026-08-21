#!/usr/bin/env python

# NOT USED — use preprocess_artifact_removal_by_sqr_deviation.py (via run_preprocess.sh)
# instead, the current pipeline.
#
# NOTE (legacy/reference, not used anymore):
# Earlier, exploratory version of the preprocessing pipeline.
#

# %%
# just for this machine
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
#import torch
import spikeinterface.full as si
import scipy.signal
from tqdm.auto import tqdm, trange

# %%
#data_dir = Path("~/data").expanduser()
#scratch_dir = Path("~/scratch").expanduser()

# %%
'''
base_folder = Path(r"E:\GLC6644\GLC6644h_g0\\")
spikeglx_folder = base_folder / 'GLC6644h_g0_imec0'

rec_dir = spikeglx_folder #data_dir / "cuttlefish" / "GLC6962_g"
exp_dir = spikeglx_folder / 'kilosort4_cw' #data_dir / "cuttlefish" / "GLC6962_g_res"
'''
# folder where a preprocessed copy of the recording will be saved
# scratch_dir is on a fast SSD on my computer. see the commment
# in the last cell before KS stuff below for more on this.
# ppx_dir = Path(r"E:\scratch\\") / "tmp_preprocessing"

base_folder = "/u/gongt/GLC7189_g_g1"
spikeglx_folder = os.path.join(base_folder, 'GLC7189_g_g1_imec0')

rec_dir = spikeglx_folder 
#exp_dir = spikeglx_folder / 'kilosort4_cw' / 'kilosort4' / ' '#data_dir / "cuttlefish" / "GLC6962_g_res"

# %%
si.set_global_job_kwargs(n_jobs=4)

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
# this is setup for visualization
n_seconds = rec.get_duration()
n_mins = n_seconds / 60
chlen = 512
t_start = rec.get_start_time()
min_start = int(np.ceil(t_start / 60))

# %%
imkw = dict(aspect='auto', interpolation='none')
for min_start in range(min_start, min_start + int(n_mins), 5):
    sample_start = rec.time_to_sample_index(min_start * 60)
    print(sample_start)
    chunk = rec.get_traces(0, sample_start, sample_start + chlen)

    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(chunk.T, **imkw)
    plt.colorbar(im, ax=ax, shrink=0.5)
    plt.show()
    plt.close(fig)

# %%
# preprocess
rec0 = si.read_spikeglx(rec_dir, stream_name='imec0.ap', load_sync_channel=False)  # to avoid stacking the ppx chain accidentally.
rec = si.highpass_filter(rec0, dtype="float32")
bc, labels = si.detect_bad_channels(rec, seed=0)
print(f"{bc=}")

# %%
rec = rec.remove_channels(bc)

# %%
chunks = si.get_random_data_chunks(rec,  num_chunks_per_segment=100, seed=0)

# %%
std = chunks.std(0)

# %%
peaks, _ = scipy.signal.find_peaks(std, threshold=0.75)
plt.plot(std)
plt.plot(peaks, std[peaks], "x")
plt.xlabel('channel')
plt.ylabel('standard deviation');

# %%
rec = rec.remove_channels(rec.channel_ids[peaks])
rec = si.common_reference(rec)

# %%

# list_triggers=np.asarray([200.74, 567.52, 570.9, 638.4, 635.23, 683.01, 1184.5, 1184.85, 1470.75, 1476.7, 1482.04, 1485.54, 1508.13, 1869.19, 2167.46, 2435.75, 2639.78])
# list_triggers_samp = list_triggers*fs
# rec_clean = si.remove_artifacts(recording=rec, list_triggers=list_triggers_samp, mode='zeros',ms_before=100,ms_after = 700)


# %%
### Extract TTLs ### 
raw_rec = si.read_spikeglx(base_folder, stream_name='nidq', load_sync_channel=False)
raw_rec

# %%
ttlName = ['sync','stim','side','overview','hi-res']
input_name = input("Enter a name: ")
chanInterest = ttlName.index(input_name)
chanInterest

# %%
ttl_sig = raw_rec.get_traces()[:,chanInterest]
norm_sig = (ttl_sig - np.min(ttl_sig)) / (np.max(ttl_sig) - np.min(ttl_sig))
#median_signal = np.median(ttl_sig[ttl_sig > 0])
threshold = 0.5
thresh_signal = (norm_sig >= threshold).astype(int)
diff_sig = np.diff(thresh_signal)

indOn = np.where(diff_sig > 0.6)[0]
indOff = np.where(diff_sig < -0.6)[0]

ttl_on = indOn/fs
ttl_off = indOff/fs

len(ttl_on)

# %%
filename = f"ttl_{input_name}_on2.txt"
save_filename = os.path.join(base_folder, filename)
with open(save_filename, "w") as f:
    f.write(",".join(map(str, ttl_on)))

# %%
### FIND EINK ARTIFACTS ### 
'''
############## BY CONVOLUTION ###############
def moving_average(signal, window_size):
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='same')

rec1 = si.read_spikeglx(base_folder, stream_name='imec0.ap', load_sync_channel=False)

print("getting traces...")
chunk_size = 300000
n_frames = rec1.get_num_frames()
n_chunks = int(np.ceil(n_frames / chunk_size))
accum_sig = []
for i in tqdm(range(n_chunks), desc="Processing chunks"):
    start_frame = i * chunk_size
    end_frame = min((i + 1) * chunk_size, n_frames)
    traces = rec1.get_traces(start_frame=start_frame, end_frame=end_frame)
    # compute mean across channels immediately
    chunk_mean = np.mean(traces, axis=1)
    accum_sig.append(chunk_mean)
mean_sig = np.concatenate(accum_sig)
print(mean_sig.shape)
print("traces retrieved")
meanSub_sig = np.abs(mean_sig) - np.mean(mean_sig)

print("Computing convolution...")
filt_sig = moving_average(meanSub_sig,500)
norm_sig = (filt_sig - np.min(filt_sig)) / (np.max(filt_sig) - np.min(filt_sig))

print("detecting time points")
buffer_window = [500,3000]
threshold = 0.057
thresh_signal = (norm_sig >= threshold).astype(int)
diff_sig = np.diff(thresh_signal)

indOn = np.where(diff_sig > 0.6)[0]
indOff = np.where(diff_sig < -0.6)[0]

if len(indOn) == len(indOff):
    print("Arrays match in length")
else:
    print("Arrays do not match in length")

print(len(indOn))

eInkOn = (indOn - buffer_window[0])/fs
eInkOff = (indOn + buffer_window[1])/fs
print(eInkOn,eInkOff)
'''

# %%
### FIND EINK ARTIFACTS ###
################ BY MEAN AND STD ###################
rec1 = si.read_spikeglx(base_folder, stream_name='imec0.ap', load_sync_channel=False)

save_folder = Path(os.path.join(base_folder, "artifact_plots", "detection"))
save_folder.mkdir(parents=True, exist_ok=True)

window_size_sec = 60
threshold_std = 1.5
percent_outliers_threshold = 11
threshold_std_extreme = 10
overlap_sec = 1 #when scanning the entire trace, the next chunk has 1s overlap with the previous chunk ==> aoivd an artifact being split into 2 chunks 
total_samples = rec1.get_num_samples(segment_index=0)

window_size = int(window_size_sec * fs)-1
overlap_size = int(overlap_sec * fs)
step_size = window_size - overlap_size

print(f"Total samples: {total_samples}, analyzing in {window_size_sec}s windows")

close_artifact_segments = []  # store sample ranges where space_to_prev <400 for 5 consecutive times
extreme_outliers = []

for start_sample in tqdm(range(0, total_samples, step_size), desc="Scanning dataset by minute"):
    end_sample = min(start_sample + window_size, total_samples)
    trace = rec1.get_traces(0, start_sample, end_sample)
    trace_mean = np.mean(trace, axis=1)
    mean_val = np.mean(trace_mean)
    std_val = np.std(trace_mean)
    upper_threshold = mean_val + threshold_std * std_val
    lower_threshold = mean_val - threshold_std * std_val
    upper_threshold_extreme = mean_val + threshold_std_extreme * std_val
    lower_threshold_extreme = mean_val - threshold_std_extreme * std_val

    # screen for close arttifact clusters 
    outlier_indices = np.where((trace_mean > upper_threshold) | (trace_mean < lower_threshold))[0]
    percent_outliers = (len(outlier_indices) / len(trace_mean)) * 100
    print(f"Chunk {start_sample}–{end_sample}: {percent_outliers:.2f}% of samples exceed threshold")
    if percent_outliers < percent_outliers_threshold: # this filters out chunks where std is small with no huge outliers ==> a lot of samples are above threshold 
        if len(outlier_indices) > 0:
            diffs = np.diff(outlier_indices)
            consecutive_groups = np.split(outlier_indices, np.where(diffs != 1)[0] + 1) #consecutive outliers 
            group_bounds = []
            prev_end = None
            for group in consecutive_groups:
                if len(group) >= 10: # >10 consecutive outliers 
                    group_start = start_sample + group[0]
                    group_end = start_sample + group[-1]
                    if prev_end is not None:
                        space_to_prev = group_start - prev_end #space to previous outlier clusters 
                    else:
                        space_to_prev = None  # still counts as <400
                    group_bounds.append((group_start, group_end, space_to_prev))
                    prev_end = group_end
    
            # compute consecutive small gaps (<400 OR None)
            consec_count = 0
            for i in range(len(group_bounds)):
                gap = group_bounds[i][2]
                if gap is None or gap < 400: #for eInk artifacts, gap is usually ~368 ==> closely spaced outlier clusters 
                    consec_count += 1
                    if consec_count >= 5: # > 5 closely spaced outlier clusters, neuronal spikes should not be able to fire so consecutively 
                        # mark start of the first group in this run and end of current
                        start_run = group_bounds[i - 4][0]
                        end_run = group_bounds[i][1]
                        close_artifact_segments.append((start_run, end_run))
                        #print(start_run, end_run) 
                        consec_count = 0  # reset after noting
                else:
                    consec_count = 0
    # screen for extreme outliers 
    outlier_indices_extreme = np.where((trace_mean > upper_threshold_extreme) | (trace_mean < lower_threshold_extreme))[0] + start_sample
    print("outlier_indices_extreme", outlier_indices_extreme) 
    extreme_outliers.append(outlier_indices_extreme)
    
    # plot chunk if consecutive outlier clusters are detected 
    segments_in_window = [(s, e) for (s, e) in close_artifact_segments if not (e < start_sample or s > end_sample)]
    if segments_in_window:
        print(segments_in_window)
        # plot results to check 
    x = np.arange(start_sample, end_sample)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3), sharey=True)
    ax.plot(x, trace_mean)

    # highlight all detected outlier groups in red
    for (seg_start, seg_end) in close_artifact_segments:
        seg_mask = (x >= seg_start) & (x <= seg_end)
        ax.plot(x[seg_mask], trace_mean[seg_mask], color='red', linewidth=2)

    extreme_indices_rel = outlier_indices_extreme - start_sample
    # plot dots for extreme outliers outside the red segments
    if extreme_indices_rel.size > 0:
        y_extreme = np.full(len(extreme_indices_rel), upper_threshold_extreme)
        ax.plot(x[extreme_indices_rel], y_extreme, 'ro', markersize=4, label='Extreme Outliers')

    ax.axhline(upper_threshold, color='red', linestyle='--', linewidth=1)
    ax.axhline(lower_threshold, color='red', linestyle='--', linewidth=1)
    ax.axhline(upper_threshold_extreme, color='red', linestyle='--', linewidth=1)
    ax.axhline(lower_threshold_extreme, color='red', linestyle='--', linewidth=1)
    
    ax.set_title("Original Trace with Dense Artifact Sequences")

    save_path = save_folder / f"artifact_{start_sample}_{end_sample}.png"
    plt.savefig(save_path)
    plt.close(fig)


# %%
# print all detected artifact segments
eInkOn = []
eInkOff = []
for seg in close_artifact_segments:
    eInkOn.append(seg[0]/fs)
    eInkOff.append(seg[1]/fs)

# %%
extreme_outliers = np.concatenate(extreme_outliers).tolist()
print(extreme_outliers)

# %%
print("eInkOn:", eInkOn)
print("eInkOff:", eInkOff)

# %%
def save_array(arr, label):
    filename = f"{label}_{input_name}.txt"
    save_path = os.path.join(base_folder, filename)
    with open(save_path, "w") as f:
        f.write(",".join(map(str, arr)))

save_array(eInkOn, "eInk_on")
save_array(eInkOff, "eInk_off")
save_array(extreme_outliers, "extreme_outliers")

# %%
#### if triggers already known ## 
eInkOn = np.loadtxt(f"{base_folder}/eInk_on_overview.txt", delimiter=',')
eInkOff = np.loadtxt(f"{base_folder}/eInk_off_overview.txt", delimiter=',')
extreme_outliers = np.loadtxt(f"{base_folder}/extreme_outliers_overview.txt", delimiter=',')
#####

#print(eInkOff - eInkOn)
eInkOn = np.array(eInkOn)
eInkOff = np.array(eInkOff)
list_triggers = eInkOn
list_triggers_samp = list_triggers*fs
print(len(list_triggers))

ms_before = 150
ms_after = np.max(eInkOff - eInkOn)*1000+150
print("ms_after:", ms_after)

ms_outliers = 50
rec_clean = si.remove_artifacts(recording=rec, list_triggers=list_triggers_samp, mode='zeros',ms_before=ms_before,ms_after = ms_after)
rec_clean = si.remove_artifacts(recording=rec_clean, list_triggers=extreme_outliers, mode='zeros',ms_before=ms_outliers,ms_after = ms_outliers)

print(list_triggers)

# %%
####### SWEEP ACROSS THE WHOLE DATASET - CHECK IF ALL ARTIFACTS ARE REMOVED ###########
rec1 = si.read_spikeglx(base_folder, stream_name='imec0.ap', load_sync_channel=False)

save_folder = Path(os.path.join(base_folder, "artifact_plots", "confirmation"))
save_folder.mkdir(parents=True, exist_ok=True)

window_size_sec = 20
threshold_std = 1.5
percent_outliers_threshold = 11
overlap_sec = 1 #when scanning the entire trace, the next chunk has 1s overlap with the previous chunk ==> aoivd an artifact being split into 2 chunks 
total_samples = rec1.get_num_samples(segment_index=0)

window_size = int(window_size_sec * fs)-1
overlap_size = int(overlap_sec * fs)
step_size = window_size - overlap_size

print(f"Total samples: {total_samples}, analyzing in {window_size_sec}s windows")

close_artifact_segments = []  # store sample ranges where space_to_prev <400 for 5 consecutive times

rec_clean_astype = rec_clean.astype('float16')

remaining_outliers = []
for start_sample in tqdm(range(0, total_samples, step_size), desc="Scanning dataset by minute"):
    end_sample = min(start_sample + window_size, total_samples)
    trace = rec1.get_traces(0, start_sample, end_sample)
    trace_mean = np.mean(trace, axis=1)
    mean_val = np.mean(trace_mean)
    std_val = np.std(trace_mean)
    upper_threshold = mean_val + threshold_std * std_val
    lower_threshold = mean_val - threshold_std * std_val
    outlier_indices = np.where((trace_mean > upper_threshold) | (trace_mean < lower_threshold))[0]
    percent_outliers = (len(outlier_indices) / len(trace_mean)) * 100
    print(f"Chunk {start_sample}–{end_sample}: {percent_outliers:.2f}% of samples exceed threshold")

    if percent_outliers < percent_outliers_threshold: # this filters out chunks where std is small with no huge outliers ==> a lot of samples are above threshold  
        clean_trace = rec_clean_astype.get_traces(0, start_sample, end_sample)
        clean_trace_mean = np.mean(clean_trace, axis=1)
        clean_mean_val = np.mean(clean_trace_mean)
        clean_std_val = np.std(clean_trace_mean)
        clean_upper_threshold = clean_mean_val + threshold_std * clean_std_val
        clean_lower_threshold = clean_mean_val - threshold_std * clean_std_val
        clean_outlier_indices = np.where((clean_trace_mean > clean_upper_threshold) | (clean_trace_mean < clean_lower_threshold))[0]
        clean_percent_outliers = (len(clean_outlier_indices) / len(clean_trace_mean)) * 100
        
        if clean_percent_outliers < percent_outliers_threshold and len(clean_outlier_indices) > 0: 
            remaining_outliers.append(clean_outlier_indices+start_sample) 
        print(f"Chunk {start_sample}–{end_sample}: {clean_percent_outliers:.2f}% of samples exceed threshold")
        
        x = np.arange(start_sample, end_sample) / fs
        fig, (ax, ay) = plt.subplots(nrows=2, ncols=1, figsize=(20, 6), sharey=True)
        ax.plot(x, trace_mean[:])
        ax.set_title('Original Trace')
        ay.plot(x, clean_trace_mean[:])
        ay.set_title('Clean Trace')
        zero_mask = (clean_trace_mean == 0)
        if np.any(zero_mask):
            ay.scatter(x[zero_mask], clean_trace_mean[zero_mask], color='red', s=10, label='Zeroed samples')
            
        save_path = save_folder / f"artifact_{start_sample}_{end_sample}.png"
        plt.savefig(save_path)
        plt.close(fig)

# %%
print(remaining_outliers)

if len(remaining_outliers) > 0: 
    raise Exception("There may be remaining outliers in the cleaned trace. please check manually!")

# %%
###################################################

# %%
rec = rec_clean

# %%
chunks = si.get_random_data_chunks(rec,  num_chunks_per_segment=100, seed=0)
chunks.shape

# %%
std = chunks.std(0)
gstd = chunks.std()
plt.plot(std)
plt.axhline(gstd, color='k')
plt.xlabel('channel')
plt.ylabel('standard deviation')
save_path = save_folder / f"std_channel.png"
plt.savefig(save_path)
plt.close(fig)

# %%
rec = si.scale(rec, gain=1.0 / gstd, dtype='float16')

# %%
n_seconds = rec.get_duration()
n_mins = n_seconds / 60
chlen = 512
t_start = rec.get_start_time()
min_start = int(np.ceil(t_start / 60))

# %%
imkw0 = dict(aspect='auto', interpolation='none', cmap='viridis')
imkw = dict(aspect='auto', interpolation='none', cmap='seismic', vmin=-8, vmax=8)
rechp = si.highpass_filter(si.read_spikeglx(rec_dir, stream_name='imec0.ap', load_sync_channel=False))

for min_start in range(min_start, min_start + int(n_mins), 5):
    sample_start = rec.time_to_sample_index(min_start * 60)
    sample_end = sample_start + chlen
    t_start = rec.sample_index_to_time(sample_start) * 1000
    t_end = rec.sample_index_to_time(sample_end) * 1000
    print(sample_start)

    chunk0 = rechp.get_traces(0, sample_start, sample_end, return_scaled=True)
    chunk = rec.get_traces(0, sample_start, sample_end)

    fig, (ax, ay) = plt.subplots(ncols=2, figsize=(10, 5))

    im = ax.imshow(chunk0.T, extent=(t_start, t_end, 0, rec.get_num_channels()), **imkw0)
    plt.colorbar(im, ax=ax, shrink=0.5, label='uV')

    im = ay.imshow(chunk.T, extent=(t_start, t_end, 0, rec.get_num_channels()), **imkw)
    plt.colorbar(im, ax=ay, shrink=0.5, label='standard units')

    ax.set_title('highpass filtered')
    ay.set_title('bad chan + hpf+cmr+zscore')
    ax.set_xlabel("time (ms)")
    ay.set_xlabel("time (ms)")
    
    plt.show()
    plt.close(fig)

# %%
rec = rec.astype('float16')

# %%
rec

# %%
rec.save_to_folder(os.path.join(base_folder, "preprocessed"), overwrite=True, n_jobs=8)

# %% [markdown]
# ## ks thing

# %%
'''
# then alter ks_params to: 
ks_params = si.get_default_sorter_params('kilosort4')
ks_params['do_CAR'] = False # so as not to common reference twice

# comprises highpass filter (>300 Hz) and whitening (right? see https://kilosort.readthedocs.io/en/latest/api.html#kilosort.run_kilosort.compute_preprocessing)
ks_params['skip_kilosort_preprocessing'] = False 
ks_params['do_correction'] = True
#ks_params['n_jobs'] = -1
ks_params['nblocks'] = 0
ks_params['n_pcs'] = 12
ks_params['delete_recording_dat']= False
ks_params['ccg_threshold'] = 0.1
#ks_params['tmax'] = 500

# no need to do this, since we saved a copy with save_to_folder above
# but if that was not done, this would be good to do.
ks_params['save_preprocessed_copy'] = True #False CHANGED
'''

# %%
'''
sorting = si.run_sorter(
    'kilosort4',
    rec, 
    folder=exp_dir / 'kilosort4',
    docker_image=False,
    verbose=True,
    remove_existing_folder=True,
    **ks_params,
)
'''

# %% [markdown]
# ## Below is dartsort stuff, please ignore :) 

# %%
# rec = si.read_binary_folder(ppx_dir)

# cfg = dartsort.DeveloperConfig(
#     nn_denoiser_class_name='Decollider',
#     nn_denoiser_pretrained_path=None,
#     n_jobs_cpu=4,
#     n_jobs_gpu=2,
#     save_intermediate_features=True,
#     save_intermediate_labels=True,
#     alignment_ms=2.0,
# )

# # shutil.rmtree(exp_dir / "dartsort")

# dartsort.dartsort(
#     rec,
#     exp_dir / "dartsort",
#     cfg=cfg,
#     overwrite=True,
# )

# dartvis.visualize_all_sorting_steps(rec, exp_dir / "dartsort", vis_dir / "dartsort")

# %%



