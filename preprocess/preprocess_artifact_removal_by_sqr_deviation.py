#!/usr/bin/env python

# %%
# just for this machine
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# %%
from pathlib import Path

# %%
import numpy as np
import matplotlib.pyplot as plt
#import torch
import spikeinterface.full as si
import scipy.signal
from tqdm.auto import tqdm, trange
import argparse

# %%
si.set_global_job_kwargs(n_jobs=-1)

# %% load_rec
def load_rec(rec_dir, start_sec=None, end_sec=None):
    if "imec" in rec_dir:
        rec = si.read_spikeglx(rec_dir, stream_name='imec0.ap', load_sync_channel=False)
    else:
        rec = si.load(rec_dir)
    if start_sec is not None or end_sec is not None:
        fs = rec.get_sampling_frequency()
        start_frame = int(start_sec * fs) if start_sec is not None else 0
        end_frame = int(end_sec * fs) if end_sec is not None else rec.get_num_frames()
        rec = rec.frame_slice(start_frame=start_frame, end_frame=end_frame)
    return rec

# %%
parser = argparse.ArgumentParser()
parser.add_argument('--animal_tag', type=str, required=True)
parser.add_argument('--exp_tag', type=str, nargs='+', required=True,
                    help='One or more experiment tags. If multiple are given the '
                         'recordings are concatenated before preprocessing.')
parser.add_argument('--start_sec', type=float, default=None,
                    help='Start time in seconds for loading the recording. Default is None (start of recording).')
parser.add_argument('--end_sec', type=float, default=None,
                    help='End time in seconds for loading the recording. Default is None (end of recording).')

args = parser.parse_args()
animal_tag = args.animal_tag
exp_tags   = args.exp_tag          # always a list, even when only one tag given
start_sec  = args.start_sec
end_sec    = args.end_sec

def _find_rec_dir(animal_tag, exp_tag):
    """
    Return the recording directory for a single exp_tag.
    Priority: imec folder > concat_rec > preprocessed > raw
    Raises FileNotFoundError with a listing if nothing is found.
    """
    exp_num = f'{animal_tag}_{exp_tag}'
    # base    = f"/nexus/posix0/MBR-neuralsystems/ephys/{animal_tag}/{exp_num}"
    base = f"/ptmp/gongt/{animal_tag}/{exp_num}"
    items   = os.listdir(base)
    for item in items:
        if "imec" in item:
            path = os.path.join(base, f"{exp_num}_imec0")
            print(f"  [rec_dir] found imec folder: {path!r}")
            return path
    for candidate in ("concat_rec"):
        if candidate in items:
            path = os.path.join(base, candidate)
            print(f"  [rec_dir] found '{candidate}' folder: {path!r}")
            return path
    raise FileNotFoundError(
        f"No recognised recording folder found in {base!r}.\n"
        f"Contents: {items}"
    )

if len(exp_tags) == 1:
    # ── single recording — original behaviour ─────────────────────────────
    exp_tag    = exp_tags[0]
    exp_num    = f'{animal_tag}_{exp_tag}'
    base_folder = f"/ptmp/gongt/{animal_tag}/{exp_num}"
    # base_folder = f"/nexus/posix0/MBR-neuralsystems/ephys/{animal_tag}/{exp_num}"
    rec_dir    = _find_rec_dir(animal_tag, exp_tag)

else:
    # ── multiple recordings — concatenate and save ────────────────────────
    joined_tag  = "_".join(exp_tags)
    exp_num     = f'{animal_tag}_{joined_tag}'
    base_folder = f"/ptmp/gongt/{animal_tag}/{exp_num}"
    # base_folder = f"/nexus/posix0/MBR-neuralsystems/ephys/{animal_tag}/{exp_num}"
    concat_path = os.path.join(base_folder, "concat_rec")
    Path(base_folder).mkdir(parents=True, exist_ok=True)

    if os.path.exists(concat_path):
        print(f"Concatenated recording already exists at {concat_path!r} — loading.")
        rec_dir = concat_path
    else:
        print(f"Concatenating {len(exp_tags)} recordings: {exp_tags}")
        recs_to_concat = []
        for tag in exp_tags:
            d = _find_rec_dir(animal_tag, tag)
            print(f"  loading {d!r}")
            recs_to_concat.append(load_rec(d, start_sec=start_sec, end_sec=end_sec))
        concat_rec = si.concatenate_recordings(recs_to_concat)
        print(f"Saving concatenated recording to {concat_path!r} …")
        concat_rec.save_to_folder(concat_path, overwrite=False, n_jobs=-1)
        print("Concatenation saved.")
        rec_dir = concat_path

results_folder_check = base_folder
results_folder_data  = base_folder
Path(results_folder_data).mkdir(parents=True, exist_ok=True)

#exp_dir = spikeglx_folder / 'kilosort4_cw' / 'kilosort4' / ' '#data_dir / "cuttlefish" / "GLC6962_g_res"

# %% [markdown]
# ## Preprocessing

# %%
rec = load_rec(rec_dir, start_sec=start_sec, end_sec=end_sec)
rec.get_probe().to_dataframe()
rec

# %%
# this is setup for visualization
n_seconds = rec.get_duration()
n_mins = n_seconds / 60
chlen = 512
t_start = rec.get_start_time()
min_start = int(np.ceil(t_start / 60))
fs = rec.get_sampling_frequency()

# %%
save_folder = os.path.join(results_folder_check, "raw_neural_data")

if not os.path.exists(save_folder) or len(os.listdir(save_folder)) == 0:
    os.makedirs(save_folder, exist_ok=True)
    print(f"Created folder {save_folder} for saving images")
    imkw = dict(aspect='auto', interpolation='none')
    for min_start in range(min_start, min_start + int(n_mins), 5):
        sample_start = rec.time_to_sample_index(min_start * 60)
        print(sample_start)
        chunk = rec.get_traces(0, sample_start, sample_start + chlen)

        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(chunk.T, **imkw)
        plt.colorbar(im, ax=ax, shrink=0.5)

        filename = os.path.join(save_folder, f"{min_start:04d}min.png")
        plt.savefig(filename, bbox_inches = 'tight')
        plt.show()
        plt.close(fig)
    print("all plots successfully saved")
else: 
    print(f"Folder {save_folder} already exists and is not empty. Skip saving images.")

# %%
# preprocess
rec0 = load_rec(rec_dir, start_sec=start_sec, end_sec=end_sec)
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
plt.ylabel('standard deviation')

save_path = os.path.join(results_folder_check, "raw_channel_std.png")
if not os.path.exists(save_path):
    plt.savefig(save_path, bbox_inches = 'tight')
    print(f"Saved channel std plot to {save_path}")
else: 
    print(f"Channel std plot {save_path} already exists. Skip saving.")

plt.show()
plt.close()

# %%
rec = rec.remove_channels(rec.channel_ids[peaks])
rec = si.common_reference(rec)

# %%

# list_triggers=np.asarray([200.74, 567.52, 570.9, 638.4, 635.23, 683.01, 1184.5, 1184.85, 1470.75, 1476.7, 1482.04, 1485.54, 1508.13, 1869.19, 2167.46, 2435.75, 2639.78])
# list_triggers_samp = list_triggers*fs
# rec_clean = si.remove_artifacts(recording=rec, list_triggers=list_triggers_samp, mode='zeros',ms_before=100,ms_after = 700)


# %% [markdown]
# ## Extract TTLs

# %%
### Extract TTLs ### 
raw_rec = load_rec(rec_dir, start_sec=start_sec, end_sec=end_sec)
raw_rec

# %%
ttlName = ['sync','stim','side','overview','hi-res']
# input_name = input("Enter a name: ")
input_name = "overview"
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
save_filename = os.path.join(results_folder_check, filename)
with open(save_filename, "w") as f:
    f.write(",".join(map(str, ttl_on)))

# %% [markdown]
# ## FIND EINK ARTIFACTS

# %%
################ BY SQR DEVIATION ###################
#### if triggers already known ## 
outlier_file = os.path.join(results_folder_check, "outliers_list_sqr_overview.txt")
if os.path.exists(outlier_file):
    print(f"Reading existing outlier list from {outlier_file}...")
    outliers_list = np.loadtxt(outlier_file, delimiter=',')
else: 
    print("Outlier list not found. Finding eInk artifacts...")
    rec1 = load_rec(rec_dir, start_sec=start_sec, end_sec=end_sec)
    save_folder = Path(os.path.join(results_folder_check, "artifact_plots_sqd", "detection"))
    save_folder.mkdir(parents=True, exist_ok=True)

    window_size_sec = 60
    std_threshold = 1000
    overlap_sec = 1 #when scanning the entire trace, the next chunk has 1s overlap with the previous chunk ==> aoivd an artifact being split into 2 chunks 
    total_samples = rec1.get_num_samples(segment_index=0)

    window_size = int(window_size_sec * fs)-1
    overlap_size = int(overlap_sec * fs)
    step_size = window_size - overlap_size

    print(f"Total samples: {total_samples}, analyzing in {window_size_sec}s windows")

    outliers_list = []

    for start_sample in tqdm(range(0, total_samples, step_size), desc="Scanning dataset by minute"):
        end_sample = min(start_sample + window_size, total_samples)
        print(f"Processing samples {start_sample} to {end_sample}")
        trace = rec1.get_traces(0, start_sample, end_sample)
        trace_mean = np.mean(trace, axis=1)    
        mean_val = np.mean(trace_mean)
        std_val = np.std(trace_mean)
        time = np.arange(len(trace_mean)) + start_sample
        print(len(time))

        p_low, p_high = np.percentile(trace_mean, [20, 80])
        mask = (trace_mean >= p_low) & (trace_mean <= p_high)
        mean_robust, std_robust = np.mean(trace_mean[mask]), np.std(trace_mean[mask])
        sq_dev = (trace_mean - mean_robust) ** 2
        print(len(sq_dev))
        sq_dev_robust = sq_dev[mask]
        thresh_sq = np.mean(sq_dev_robust) + std_threshold * np.std(sq_dev_robust)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        ax1.plot(time, trace_mean, 'k-', alpha=0.8)
        ax1.set_title('Original Signal')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)

        ax2.plot(time, sq_dev, 'k-', alpha=0.8)
        ax2.axhline(thresh_sq, color='red', ls='--')
        ax2.set_title('Squared Deviation from Mean')
        ax2.set_ylabel('Squared Deviation')
        ax2.set_xlabel('Time (samples)')
        ax2.grid(True, alpha=0.3)
        
        outliers_sq_local = sq_dev > thresh_sq  # Local indices for plotting
        if np.any(outliers_sq_local): 
            outlier_indices_local = np.where(outliers_sq_local)[0]

            keep_mask = np.zeros_like(outlier_indices_local, dtype=bool)
            for i, idx in enumerate(outlier_indices_local):
                has_prev_neighbor = (i > 0) and (outlier_indices_local[i-1] == idx - 1)
                has_next_neighbor = (i < len(outlier_indices_local) - 1) and (outlier_indices_local[i+1] == idx + 1)
                if has_prev_neighbor or has_next_neighbor:
                    keep_mask[i] = True
            filtered_indices = outlier_indices_local[keep_mask]

            outlier_indices_global = filtered_indices + start_sample
            print(outlier_indices_global)
            outliers_list.extend(outlier_indices_global.tolist())

            for idx in filtered_indices:
                ax1.axvline(x=idx + start_sample, color='salmon', alpha=0.3, zorder = 1)
                ax2.axvline(x=idx + start_sample, color='salmon', alpha=0.3, zorder = 1)

        plt.tight_layout()
        
        save_path = save_folder / f"artifact_{start_sample}_{end_sample}.png"
        plt.savefig(save_path)
        plt.close(fig)

    print("Artifact detection complete!")

    outliers_list = np.array(outliers_list) / fs 
    print(outliers_list)

    input_name = "overview"
    def save_array(arr, label):
        filename = f"{label}_{input_name}.txt"
        save_path = os.path.join(results_folder_check, filename)
        with open(save_path, "w") as f:
            f.write(",".join(map(str, arr)))

    save_array(outliers_list, "outliers_list_sqr")

# %%
#print(eInkOff - eInkOn)
list_triggers = outliers_list
print(list_triggers) 

list_triggers_samp = list_triggers*fs
print(len(list_triggers))

ms_before = 150
ms_after = 150
if len(list_triggers_samp) == 0:
    print("No artifacts detected — skipping remove_artifacts, passing recording through unchanged.")
    rec_clean = rec
else:
    rec_clean = si.remove_artifacts(recording=rec, list_triggers=list_triggers_samp, mode='zeros', ms_before=ms_before, ms_after=ms_after)


# %%
####### SWEEP ACROSS THE WHOLE DATASET - CHECK IF ALL ARTIFACTS ARE REMOVED ###########
rec1 = load_rec(rec_dir, start_sec=start_sec, end_sec=end_sec)

save_folder = Path(os.path.join(results_folder_check, "artifact_plots_sqd", "confirmation"))
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
    print("There may be remaining outliers in the cleaned trace. please check manually!")


# %%
###################################################

# %%
#rec_clean = si.zscore(rec_clean)
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

save_path = os.path.join(results_folder_check, "processed_channel_std.png")
plt.savefig(save_path, bbox_inches = 'tight')
print(f"Saved channel std plot to {save_path}")
plt.close()

# %%
rec = si.scale(rec, gain=1.0 / gstd, dtype='float16')

# %%
n_seconds = rec.get_duration()
n_mins = n_seconds / 60
chlen = 512
t_start = rec.get_start_time()
min_start = int(np.ceil(t_start / 60))

# %%
save_folder = os.path.join(results_folder_check, "processed_neural_data")

if not os.path.exists(save_folder) or len(os.listdir(save_folder)) == 0:
    os.makedirs(save_folder, exist_ok=True)
    print(f"Created folder {save_folder} for saving images")
    imkw0 = dict(aspect='auto', interpolation='none', cmap='viridis')
    imkw = dict(aspect='auto', interpolation='none', cmap='seismic', vmin=-8, vmax=8)
    rechp = si.highpass_filter(load_rec(rec_dir, start_sec=start_sec, end_sec=end_sec))

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
        ay.set_title('bad chan + hpf+cmr')
        ax.set_xlabel("time (ms)")
        ay.set_xlabel("time (ms)")

        save_path = os.path.join(save_folder, f"segment_{min_start}min.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
    print("all plots successfully saved")
else: 
    print(f"Folder {save_folder} already exists and is not empty. Skip saving images.")

# %%
rec = rec.astype('float16')

# %%
rec

# %%
rec.save_to_folder(os.path.join(results_folder_data, "preprocessed"), overwrite=True, n_jobs=-1)