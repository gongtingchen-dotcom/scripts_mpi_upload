#!/usr/bin/env python3
#
# Plain Kilosort4 driver — same CLI and run-folder naming as
# ks4_checkpoint.py, just without the checkpoint/SLURM-requeue machinery.
# Usable, but the job will simply die if it doesn't finish inside one 24h
# SLURM allocation. Prefer ks4_checkpoint.py (run_ks4_checkpoint.sh) instead
# unless you have a specific reason not to — it does everything this does,
# plus survives the walltime limit.

# %%
import os
import argparse
import shutil 
import spikeinterface.full as si
from pathlib import Path
from datetime import datetime
import numpy as np

# %%
PARAM_ABBREV = {
    "batch_size": "bs", 
    "nblocks": "nb", 
    "Th_universal": "thu", 
    "Th_learned": "thl",
    "nt": "nt", 
    "shift": "shft", 
    "scale": "scl", 
    "artifact_threshold": "art",
    "nskip": "nskp", 
    "whitening_range": "wr", 
    "highpass_cutoff": "hp",
    "binning_depth": "bd", 
    "sig_interp": "si", 
    "drift_smoothing": "dsm",
    "nt0min": "nt0m", 
    "dmin": "dmin", 
    "dminx": "dmix", 
    "min_template_size": "mts",
    "template_sizes": "tsz", 
    "nearest_chans": "nc", 
    "nearest_templates": "ntmp",
    "max_channel_distance": "mcd", 
    "max_peels": "mp", 
    "templates_from_data": "tfd",
    "n_templates": "n_t", 
    "n_pcs": "pcs", 
    "Th_single_ch": "thsc", 
    "acg_threshold": "acg",
    "ccg_threshold": "ccg", 
    "cluster_neighbors": "cn", 
    "cluster_downsampling": "cd",
    "max_cluster_subset": "mcs", 
    "x_centers": "xc", 
    "duplicate_spike_ms": "dup",
    "position_limit": "plim", 
    "do_CAR": "car", 
    "invert_sign": "inv",
    "save_extra_vars": "evars", 
    "save_preprocessed_copy": "spc", 
    "torch_device": "gpu",
    "bad_channels": "bad", 
    "clear_cache": "clr", 
    "do_correction": "dc",
    "skip_kilosort_preprocessing": "skip", 
    "keep_good_only": "good",
    "use_binary_file": "bin", 
    "delete_recording_dat": "del", 
    "pool_engine": "pool",
    "n_jobs": "jobs", 
    "chunk_duration": "chk", 
    "max_threads_per_worker": "thrd",
    "time_end": "te",
    "time_start": "ts"
}

INT_PARAMS = [
    "nt0min", "nt", "batch_size", "nblocks", "whitening_range", 
    "nearest_chans", "n_templates", "n_pcs", "max_threads_per_worker", "time_end", "time_start"
]

def build_run_name(defaults, current, extra_params=None):
    """Generate folder name from changed parameters only"""
    tags = []
    for k, v in current.items():
        if k in defaults and str(defaults[k]) != str(v):
            abbr = PARAM_ABBREV.get(k, k)
            clean_v = int(v) if isinstance(v, (float, int)) and v == int(v) else v
            tags.append(f"{abbr}{clean_v}")
    # Add extra parameters (non-kilosort params like time_end)
    if extra_params:
        for k, v in extra_params.items():
            if v is not None:
                abbr = PARAM_ABBREV.get(k, k)
                clean_v = int(v) if isinstance(v, (float, int)) and v == int(v) else v
                tags.append(f"{abbr}{clean_v}")
    return "_".join(tags) if tags else "default"

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

# %%
ks4_defaults = si.get_default_sorter_params('kilosort4')
parser = argparse.ArgumentParser(description="set Kilosort4 params")

parser.add_argument("--root_folder", required=True)
parser.add_argument("--data_folder", required=True)
parser.add_argument("--time_end", type=int, default=None, help="End time in seconds to cleave data (optional)")
parser.add_argument("--time_start", type=int, default=0, help="Start time in seconds to cleave data (default: 0)")

for param, default_val in ks4_defaults.items():
    if isinstance(default_val, bool):
        parser.add_argument(f"--{param}", type=str2bool, default=default_val)
    elif isinstance(default_val, (list, tuple)):
        parser.add_argument(f"--{param}", nargs='+', type=float, default=default_val)
    else:
        parser.add_argument(f"--{param}", type=type(default_val) if default_val is not None else float, default=default_val)

args = parser.parse_args()
args_dict = vars(args)
current_ks_params = {}
for k in ks4_defaults.keys():
    val = args_dict[k]
    if k in INT_PARAMS and val is not None:
        current_ks_params[k] = int(val)
    else:
        current_ks_params[k] = val

time_end = args_dict.get('time_end', None)
time_start = args_dict.get('time_start', 0) or 0
extra_params = {}
if time_end is not None:
    extra_params["time_end"] = time_end
if time_start:
    extra_params["time_start"] = time_start

run_name = build_run_name(ks4_defaults, current_ks_params, extra_params)
output_folder = Path(args.root_folder) / "KS4" / f"{run_name}"
data_folder = args.data_folder

print(f"data folder: {args.data_folder}")
print(f"output folder: {output_folder.name}")

output_folder.mkdir(parents=True, exist_ok=True)
params_file = output_folder / "kilosort_params.txt"

params_file = os.path.join(output_folder, "kilosort_params.txt")

print(f"Data folder: {data_folder}")
if "imec" in data_folder: 
    spikeglx_folder = data_folder
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

    rec = si.read_spikeglx(spikeglx_folder, stream_name='imec0.ap', load_sync_channel=False)#si.read_spikeglx(base_folder)
    rec.get_probe().to_dataframe()
    rec

    # preprocess
    rec0 = si.read_spikeglx(spikeglx_folder, stream_name='imec0.ap', load_sync_channel=False)  # to avoid stacking the ppx chain accidentally.

    print("reading outlier list...")
    outliers_list = np.loadtxt(os.path.join(args.root_folder, "outliers_list_sqr_overview.txt"), delimiter=',')

    #print(eInkOff - eInkOn)
    list_triggers = outliers_list
    #print(list_triggers)
    list_triggers_samp = list_triggers*fs
    print(f"number of outliers: {len(list_triggers)}")

    if len(list_triggers) > 0:
        ms_before = 150
        ms_after = 150
        rec_clean = si.remove_artifacts(recording=rec0, list_triggers=list_triggers_samp, mode='zeros',ms_before=ms_before,ms_after = ms_after)
        rec = rec_clean
else: 
    rec = si.load(data_folder)
    fs = rec.get_sampling_frequency()
print(rec)

# Cleave data if time_start/time_end is specified (convert seconds to samples)
if time_start or time_end is not None:
    start_frame = int(time_start * fs) if time_start else 0
    end_frame = int(time_end * fs) if time_end is not None else rec.get_num_samples()
    print(f"Cleaving data from {time_start}s to {time_end}s ({start_frame}–{end_frame} samples)")
    rec = rec.frame_slice(start_frame=start_frame, end_frame=end_frame)
    print(f"Cleaved recording: {rec}")

if os.path.exists(output_folder):
    shutil.rmtree(output_folder) 

# %%
sorting = si.run_sorter(
    'kilosort4',
    rec,
    folder=output_folder,
    docker_image=False,
    verbose=True,
    **current_ks_params,
)

with open(params_file, "w") as f:
    f.write(f"Kilosort parameters saved on {datetime.now()}\n")
    f.write(f"SpikeInterface version: {si.__version__}\n\n")
    f.write("Parameters:\n")
    for key, value in current_ks_params.items():
        f.write(f"{key}: {value}\n")
print(f"Full Kilosort parameters saved to: {params_file}") 