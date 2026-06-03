# %% [markdown]
# **IMPORTS**

# %%
import numpy as np 
import matplotlib.pylab as plt 
import os        
import spikeinterface.full as si
from gudhi.clustering.tomato import Tomato
from spikeinterface.core import get_noise_levels
import torch
from tqdm.auto import tqdm
import threading
from queue import Queue


# %%
def plot_dgm(t,tau=0): ### to plot persistence diagram
    l = t.max_weight_per_cc_.min()
    r = t.max_weight_per_cc_.max()
    if t.diagram_.size > 0:
        plt.plot(t.diagram_[:, 0], t.diagram_[:, 1], "o", color="red")
        l = min(l, t.diagram_[:, 1].min())
        r = max(r, t.diagram_[:, 0].max())
    if l == r:
        if l > 0:
            l, r = 0.9 * l, 1.1 * r
        elif l < 0:
            l, r = 1.1 * l, 0.9 * r
        else:
            l, r = -1.0, 1.0
    plt.plot([l, r], [l, r])
    if tau>0:
        plt.plot([l, r], [l, r]-tau)
    plt.plot(
        t.max_weight_per_cc_, np.full(t.max_weight_per_cc_.shape, 1.1 * l - 0.1 * r), "o", color="green"
        )
    plt.xlabel('x')
    plt.ylabel('y')

# %% [markdown]
# **TIME WINDOW CONFIGURATION**

# %%
# ── Time window parameters: CLI args take priority, env vars are fallback ────
import argparse as _ap
_parser = _ap.ArgumentParser(description='Tomato spike clustering', add_help=True)
_parser.add_argument('--sample_start', type=int, default=None)
_parser.add_argument('--sample_end',   type=int, default=None)
_args, _ = _parser.parse_known_args()

sample_start = _args.sample_start if _args.sample_start is not None else int(os.environ.get('SAMPLE_START', 0))

if _args.sample_end is not None:
    sample_end_env = _args.sample_end          # already an int from argparse
elif os.environ.get('SAMPLE_END') is not None:
    sample_end_env = int(os.environ['SAMPLE_END'])
else:
    sample_end_env = None

data_path = '/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed'
rec = si.load(data_path)
n_samples_total = rec.get_num_samples()

sample_end = sample_end_env if sample_end_env is not None else n_samples_total

# Clamp to valid range
sample_start = max(0, sample_start)
sample_end = min(n_samples_total, sample_end)
assert sample_start < sample_end, f"Invalid range: {sample_start} >= {sample_end}"

# Create output suffix based on sample range
output_suffix = f'_s{sample_start}_e{sample_end}'
print(f'Sample range: [{sample_start:,}, {sample_end:,})  |  Suffix: {output_suffix}')

# %% [markdown]
# **LOAD DATA**

# %%
# ── Create plots output directory ────────────────────────────────────────────
folder_name = 'thresholding_peaks'
peaks_path = os.path.join(data_path, folder_name, 'peaks_combined2_sorted.npy')
plots_dir = os.path.join(data_path, folder_name, f'tomato_plots2{output_suffix}')
os.makedirs(plots_dir, exist_ok=True)

rec = si.load(data_path)
rec

# %%
fs = rec.get_sampling_frequency()
start_time = 100 
end_time = start_time + 0.1

traces = rec.get_traces(return_scaled=True, start_frame=int(start_time * fs), end_frame=int(end_time * fs))
fs = rec.get_sampling_frequency()
tt = np.arange(0, traces.shape[0]) / fs

offset = 15.0  

plt.figure(figsize=(10, 50))
num_chans = rec.get_num_channels()

for i in range(num_chans):
    plt.plot(tt, traces[:, i] + (i * offset), color='black', linewidth=0.5)

plt.xlim([0, 0.1])
plt.xlabel('Time (s)')
plt.ylabel('Channels (stacked)')
plt.title(f'Recordings on {num_chans} sites', fontsize=20)
plt.yticks([]) 
plt.tight_layout()
sample_traces_path = os.path.join(plots_dir, '01_sample_traces.png')
plt.savefig(sample_traces_path, dpi=100, bbox_inches='tight')
print(f'Saved: {sample_traces_path}')
plt.close()

# %%
pre_samp  = 15
post_samp = 15
snip_len  = pre_samp + post_samp + 1

peaks_full = np.load(peaks_path)  # Keep original full peaks for reference
peaks_mask = (peaks_full['sample_index'] >= sample_start) & (peaks_full['sample_index'] < sample_end)
peaks_global_indices = np.where(peaks_mask)[0]  # Indices into ORIGINAL peaks array
peaks = peaks_full[peaks_mask]  # Filtered subset

sp_global = peaks['sample_index']
n_spikes  = len(sp_global)
n_spikes_full = len(peaks_full)
n_chans   = rec.get_num_channels()
print(f'Total spikes in recording: {n_spikes_full:,}')
print(f'Spikes in this window: {n_spikes:,}  ({100*n_spikes/n_spikes_full:.1f}%)')
print(f'Channels: {n_chans}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using: {device}')
noise   = get_noise_levels(rec, return_scaled=True)          # (n_chans,)
noise_t = torch.from_numpy(noise.astype(np.float16)).to(device)

# %%
# ── Sanity check: plot 10 random waveforms on their peak channel ──────────────
idx10   = np.random.choice(len(peaks), size=10, replace=False)
tt_snip = np.arange(-pre_samp, post_samp + 1) / fs * 1000  # ms

plt.figure(figsize=(7, 4), layout='tight')
for k in idx10:
    sp  = int(peaks['sample_index'][k])
    ch  = int(peaks['channel_index'][k])
    snip = rec.get_traces(return_scaled=True,
                          start_frame=sp - pre_samp,
                          end_frame=sp + post_samp + 1)   # (snip_len, n_chans)
    wv_peak = snip[:, ch] / noise[ch]
    plt.plot(tt_snip, wv_peak, alpha=0.7, linewidth=1)

plt.axvline(0, color='k', linestyle='--', linewidth=0.8, label='t=0')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (noise units)')
plt.title('Peak-channel waveforms (10 random spikes)')
plt.legend()
sample_waveforms_path = os.path.join(plots_dir, '02_sample_waveforms.png')
plt.savefig(sample_waveforms_path, dpi=100, bbox_inches='tight')
print(f'Saved: {sample_waveforms_path}')
plt.close()


# %% [markdown]
# **Build Window**

# %%
n_near      = 6        # channels per spike 
window_size = 32        # spatial window width in channel-rank units
window_step = 16        # stride
chunk_s     = 60        # seconds per recording time-batch
chunk_samp  = int(chunk_s * fs)
gpu_sub     = 2000      # spikes per GPU mini-batch (avoids VRAM OOM)

locs_xy    = rec.get_channel_locations()                        # (C, 2)
depth_rank = np.argsort(locs_xy[:, 1])
rank_of_ch = np.empty(n_chans, dtype=np.int64)
rank_of_ch[depth_rank] = np.arange(n_chans)

win_starts  = np.arange(0, n_chans - window_size + 1, window_step)
win_centers = win_starts + window_size // 2
n_windows   = len(win_starts)

# ── Nearest n_near channels per channel (probe geometry) ──────────────────────
ds_mat     = np.sum((locs_xy[:, None] - locs_xy[None])**2, axis=2)
near_chans = np.argsort(ds_mat, axis=1)[:, :n_near]
near_t     = torch.from_numpy(near_chans.astype(np.int64)).to(device)

noise_t32  = torch.from_numpy(noise.astype(np.float32)).to(device)
tarange    = torch.arange(-pre_samp, post_samp + 1, device=device)

# ── Assign each spike to its nearest window ────────────────────────────────────
peak_ch_all   = peaks['channel_index']
sp_all        = peaks['sample_index']
peak_rank_all = rank_of_ch[peak_ch_all]

dist_to_wins  = np.abs(peak_rank_all[:, None] - win_centers[None, :])
spike_win_all = np.argmin(dist_to_wins, axis=1)

win_counts = np.bincount(spike_win_all, minlength=n_windows)
feat_dim   = n_near * snip_len
print(f'Channels: {n_chans}  |  Windows: {n_windows}  |  Feature dim: {feat_dim}')
print(f'Spikes/window: min={win_counts.min():,}  max={win_counts.max():,}  total={win_counts.sum():,}')
print(f'Peak RAM per window ≈ {win_counts.max() * feat_dim * 4 / 1e9:.2f} GB  (float32 features)')


# %% [markdown]
# **Build evts and TOMATO**

# %%
n_jobs   = -1    # CPU cores for non-keops fallback
# ── Use output_suffix so each sample-range run writes to its own folder ────
save_dir = os.path.join(data_path, folder_name, f'tomato_windowed2{output_suffix}')
os.makedirs(save_dir, exist_ok=True)
print(f'Results folder : {save_dir}')

# ── Save run parameters ───────────────────────────────────────────────────────
import json as _json, datetime as _dt
_params = dict(
    data_path        = data_path,
    peaks_path       = peaks_path,
    save_dir         = save_dir,
    plots_dir        = plots_dir,
    sample_start     = int(sample_start),
    sample_end       = int(sample_end),
    output_suffix    = output_suffix,
    pre_samp         = pre_samp,
    post_samp        = post_samp,
    snip_len         = snip_len,
    n_near           = n_near,
    window_size      = window_size,
    window_step      = window_step,
    chunk_s          = chunk_s,
    gpu_sub          = gpu_sub,
    feat_dim         = feat_dim,
    n_windows        = n_windows,
    n_spikes_window  = int(n_spikes),
    n_spikes_full    = int(n_spikes_full),
    tomato_dim       = 2,
    tomato_impl      = 'hnsw',
    tomato_random_seed = 42,
    n_jobs           = n_jobs,
    timestamp        = _dt.datetime.now().isoformat(),
)
_params_path = os.path.join(save_dir, 'params.json')
with open(_params_path, 'w') as _f:
    _json.dump(_params, _f, indent=2)
print(f'Saved params: {_params_path}')

# %%
def _extract_window_feats(w, pbar=None):
    """Extract waveform features for all spikes in window w.
    Reads through recording in 60s chunks; only spikes from this window are extracted.
    Returns feats (n_valid, feat_dim) float32, valid_peak_idx (n_valid,) into peaks.
    pbar: outer tqdm bar whose postfix is updated to show chunk progress.
    """
    peak_idx_w = np.where(spike_win_all == w)[0]
    sp_w = sp_all[peak_idx_w]
    ch_w = peak_ch_all[peak_idx_w]
    n_w  = len(sp_w)
    if n_w == 0:
        return np.empty((0, feat_dim), dtype=np.float32), np.empty(0, dtype=np.int64)

    feats_w  = np.zeros((n_w, feat_dim), dtype=np.float32)
    filled_w = np.zeros(n_w, dtype=bool)

    order_w   = np.argsort(sp_w)
    sp_sorted = sp_w[order_w]
    ch_sorted = ch_w[order_w]

    chunks   = np.arange(0, rec.get_num_samples(), chunk_samp)
    n_chunks = len(chunks)
    spikes_done = 0

    for i, b_start in enumerate(chunks):
        if pbar is not None:
            pbar.set_postfix(stage='extract', chunk=f'{i+1}/{n_chunks}',
                             spikes=f'{spikes_done:,}/{n_w:,}', refresh=True)

        b_end = min(b_start + chunk_samp, rec.get_num_samples())
        lo = int(np.searchsorted(sp_sorted, b_start + pre_samp,  side='left'))
        hi = int(np.searchsorted(sp_sorted, b_end   - post_samp, side='left'))
        if lo >= hi:
            continue

        local_idx = order_w[lo:hi]
        sp_batch  = sp_sorted[lo:hi]
        ch_batch  = ch_sorted[lo:hi]
        n_b       = len(sp_batch)

        t0    = int(sp_batch[0]  - pre_samp)
        t1    = int(sp_batch[-1] + post_samp + 1)
        block = rec.get_traces(return_in_uV=True, start_frame=t0, end_frame=t1)
        block_t = torch.from_numpy(block.astype(np.float32)).to(device)
        block_t = (block_t - torch.median(block_t, dim=0).values) / noise_t32

        for s in range(0, n_b, gpu_sub):
            e     = min(s + gpu_sub, n_b)
            idx_s = local_idx[s:e]
            sp_s  = sp_batch[s:e]
            ch_s  = ch_batch[s:e]
            n_s   = e - s

            locs  = torch.from_numpy((sp_s - t0).astype(np.int64)).to(device)
            t_idx = locs[:, None] + tarange[None, :]
            snips = block_t[t_idx, :]

            ch_t      = torch.from_numpy(ch_s.astype(np.int64)).to(device)
            ch_sel    = near_t[ch_t]
            snips_sel = snips.gather(2, ch_sel.unsqueeze(1).expand(n_s, snip_len, n_near))
            feats_w[idx_s] = snips_sel.permute(0, 2, 1).reshape(n_s, -1).cpu().numpy()
            filled_w[idx_s] = True

        spikes_done += n_b

    valid_local = np.where(filled_w)[0]
    return feats_w[valid_local], peak_idx_w[valid_local]


def _get_valid_peak_idx(w, n_feats):
    """Return indices into peaks[] for window w — no recording I/O needed.
    spike_win_all is deterministic so this is just np.where(spike_win_all==w).
    Falls back to full re-extraction only if boundary spikes were dropped
    (only possible within pre_samp samples of the very start/end of recording).
    """
    peak_idx_w = np.where(spike_win_all == w)[0]
    if len(peak_idx_w) == n_feats:
        return peak_idx_w
    print(f'[win {w:02d}] {len(peak_idx_w) - n_feats} boundary spikes dropped — recomputing index')
    _, valid_peak_idx_w = _extract_window_feats_threaded(w, pbar=None)
    return valid_peak_idx_w


def _extract_window_feats_threaded(w, pbar=None):
    """Extract waveform features with prefetched I/O in background thread.
    Overlaps disk reads with GPU processing to reduce I/O stalls.
    Returns feats (n_valid, feat_dim) float32, valid_peak_idx (n_valid,) into filtered peaks.
    """
    peak_idx_w = np.where(spike_win_all == w)[0]
    sp_w = sp_all[peak_idx_w]
    ch_w = peak_ch_all[peak_idx_w]
    n_w  = len(sp_w)
    if n_w == 0:
        return np.empty((0, feat_dim), dtype=np.float32), np.empty(0, dtype=np.int64)

    feats_w  = np.zeros((n_w, feat_dim), dtype=np.float32)
    filled_w = np.zeros(n_w, dtype=bool)

    order_w   = np.argsort(sp_w)
    sp_sorted = sp_w[order_w]
    ch_sorted = ch_w[order_w]

    chunks   = np.arange(0, rec.get_num_samples(), chunk_samp)
    n_chunks = len(chunks)
    spikes_done = 0

    # Queue to hold prefetched chunks: (i, block, lo, hi, local_idx, sp_batch, ch_batch)
    prefetch_queue = Queue(maxsize=2)
    
    def _prefetch_worker():
        """Background thread reads chunks ahead."""
        for i, b_start in enumerate(chunks):
            b_end = min(b_start + chunk_samp, rec.get_num_samples())
            lo = int(np.searchsorted(sp_sorted, b_start + pre_samp,  side='left'))
            hi = int(np.searchsorted(sp_sorted, b_end   - post_samp, side='left'))
            if lo >= hi:
                prefetch_queue.put((i, None, lo, hi, None, None, None))
                continue

            local_idx = order_w[lo:hi]
            sp_batch  = sp_sorted[lo:hi]
            ch_batch  = ch_sorted[lo:hi]
            
            t0    = int(sp_batch[0]  - pre_samp)
            t1    = int(sp_batch[-1] + post_samp + 1)
            block = rec.get_traces(return_in_uV=True, start_frame=t0, end_frame=t1)
            
            prefetch_queue.put((i, block, lo, hi, local_idx, sp_batch, ch_batch))
    
    # Start prefetch thread (daemon so it won't block shutdown)
    thread = threading.Thread(target=_prefetch_worker, daemon=True)
    thread.start()
    
    # Main GPU processing loop
    for i in range(n_chunks):
        if pbar is not None:
            pbar.set_postfix(stage='extract', chunk=f'{i+1}/{n_chunks}',
                             spikes=f'{spikes_done:,}/{n_w:,}', refresh=True)
        
        # Get prefetched chunk (blocks if not ready)
        chunk_idx, block, lo, hi, local_idx, sp_batch, ch_batch = prefetch_queue.get()
        
        if block is None:  # No spikes in this chunk
            continue
        
        n_b = len(sp_batch)
        t0  = int(sp_batch[0]  - pre_samp)
        
        # GPU processing (same as synchronous version)
        block_t = torch.from_numpy(block.astype(np.float32)).to(device)
        block_t = (block_t - torch.median(block_t, dim=0).values) / noise_t32

        for s in range(0, n_b, gpu_sub):
            e     = min(s + gpu_sub, n_b)
            idx_s = local_idx[s:e]
            sp_s  = sp_batch[s:e]
            ch_s  = ch_batch[s:e]
            n_s   = e - s

            locs  = torch.from_numpy((sp_s - t0).astype(np.int64)).to(device)
            t_idx = locs[:, None] + tarange[None, :]
            snips = block_t[t_idx, :]

            ch_t      = torch.from_numpy(ch_s.astype(np.int64)).to(device)
            ch_sel    = near_t[ch_t]
            snips_sel = snips.gather(2, ch_sel.unsqueeze(1).expand(n_s, snip_len, n_near))
            feats_w[idx_s] = snips_sel.permute(0, 2, 1).reshape(n_s, -1).cpu().numpy()
            filled_w[idx_s] = True

        spikes_done += n_b
    
    thread.join()  # Wait for prefetch thread to finish
    valid_local = np.where(filled_w)[0]
    return feats_w[valid_local], peak_idx_w[valid_local]


# ── Loop over all windows: extract → save feats → fit Tomato → save → plot ────
rep_win = int(np.argmax(win_counts))

with tqdm(range(n_windows), desc='Windows') as pbar:
    for w in pbar:
        center_ch   = int(depth_rank[win_centers[w]])
        tag         = f'win{w:02d}_ch{center_ch:03d}'
        feats_path  = os.path.join(save_dir, f'feats_{tag}.npy')
        tomato_path = os.path.join(save_dir, f'tomato_{tag}.npz')
        diag_path   = os.path.join(save_dir, f'persistence_{tag}.png')

        pbar.set_description(f'Windows [win {w:02d} ch {center_ch:03d}]')

        # ── load or extract features ──────────────────────────────────────────
        if os.path.exists(feats_path):
            pbar.set_postfix(stage='load feats', refresh=True)
            feats_w = np.load(feats_path)
            valid_peak_idx_w = None   # derived cheaply below if tomato is missing
        else:
            pbar.set_postfix(stage='extract', chunk='0/?', refresh=True)
            # Use threaded prefetch extractor to overlap I/O with GPU processing
            feats_w, valid_peak_idx_w = _extract_window_feats_threaded(w, pbar=pbar)
            np.save(feats_path, feats_w)

        n_w = len(feats_w)
        if n_w < 5:
            del feats_w; continue

        # ── load or fit Tomato ────────────────────────────────────────────────
        if os.path.exists(tomato_path):
            pbar.set_postfix(stage='load tomato', refresh=True)
            _d = np.load(tomato_path)
            class _Mock: pass
            t_w = _Mock()
            t_w.diagram_           = _d['diagram']
            t_w.max_weight_per_cc_ = _d['max_weight_per_cc']
            del _d
        else:
            if valid_peak_idx_w is None:
                pbar.set_postfix(stage='get peak idx', refresh=True)
                valid_peak_idx_w = _get_valid_peak_idx(w, n_w)   # instant — no I/O
            pbar.set_postfix(stage=f'Tomato {n_w:,}sp', refresh=True)
            t_w = Tomato(dim=2, n_jobs=n_jobs, implementation='hnsw', random_seed=42)
            t_w.fit(feats_w)
            np.savez(tomato_path,
                     diagram=t_w.diagram_,
                     max_weight_per_cc=t_w.max_weight_per_cc_,
                     labels=t_w.labels_,
                     valid_peak_idx=valid_peak_idx_w)

        del feats_w

        # ── plot and save persistence diagram ─────────────────────────────────
        pbar.set_postfix(stage='plot', refresh=True)
        fig, ax = plt.subplots(figsize=(6, 5), layout='tight')
        plt.sca(ax)
        plot_dgm(t_w)
        label = ' ◀ rep' if w == rep_win else ''

        ax.set_title(f'Persistence diagram — win {w:02d} ch {center_ch:03d}{label}')

        fig.savefig(diag_path, dpi=100)
        pbar.set_postfix(stage='done', refresh=True)
        plt.close()

        del t_w

# %% [markdown]
# **Post Processing Check**

# %%
import glob

# ── Quick sanity check: npz coverage vs peaks array ──────────────────────────
npz_files = sorted(glob.glob(os.path.join(save_dir, 'tomato_win*.npz')))
total_feat_spikes = sum(len(np.load(f)['labels']) for f in npz_files)
print(f'Windows with npz  : {len(npz_files)} / {n_windows}')
print(f'Spikes in feats   : {total_feat_spikes:,}')
print(f'Spikes in peaks   : {len(peaks):,}')
print(f'Difference        : {len(peaks) - total_feat_spikes:,}  (boundary-dropped)')


# %% [markdown]
# **Choose n_clusters per window**
# 
# Review the saved `persistence_*.png` files in `save_dir` to decide how many clusters each window has.
# The auto-detection below uses the largest relative gap in `max_weight_per_cc_` heights.
# Add window-specific overrides in `n_clusters_override` and re-run this cell.

# %% [DISABLED — choose-n_clusters step commented out]
# import glob as _glob
# 
# npz_files = sorted(_glob.glob(os.path.join(save_dir, 'tomato_win*.npz')))
# 
# def _auto_n_clusters(diagram, max_weight_per_cc):
#     """Auto-selection: use all connected components (= n_cc)."""
#     return int(len(max_weight_per_cc))
# 
# # ── Manual overrides: {window_index: n_clusters} ──────────────────────────────
# # Review persistence_*.png files in save_dir, then fill in as needed, e.g.:
# # n_clusters_override = {0: 5, 3: 2, 7: 4}
# # NOTE: left empty by default — fill in after reviewing the persistence diagrams
# n_clusters_override = {}
# 
# # ── Build n_clusters table ─────────────────────────────────────────────────────
# n_clusters_chosen = {}
# print(f"Persistence diagrams saved in:\n  {save_dir}\n")
# print(f"{'Win':>4}  {'center_ch':>9}  {'n_spikes':>9}  {'n_cc':>5}  {'auto':>5}  {'chosen':>6}")
# print("─" * 52)
# 
# for f in npz_files:
#     bname     = os.path.basename(f)
#     w_idx     = int(bname.split('win')[1].split('_')[0])
#     center_ch = int(bname.split('_ch')[1].replace('.npz', ''))
#     d         = np.load(f)
#     n_cc      = int(len(d['max_weight_per_cc']))
#     n_sp      = int(len(d['labels']))
#     n_auto    = _auto_n_clusters(d['diagram'], d['max_weight_per_cc'])
#     n_chosen  = int(n_clusters_override.get(w_idx, n_auto))
#     n_clusters_chosen[w_idx] = n_chosen
#     print(f"{w_idx:4d}  {center_ch:9d}  {n_sp:9,}  {n_cc:5d}  {n_auto:5d}  {n_chosen:6d}")
# 
# print(f"\nTotal windows   : {len(n_clusters_chosen)}")
# print(f"Tentative total : {sum(n_clusters_chosen.values())} clusters")
# print(f"\nUpdate n_clusters_override and re-run this cell to adjust before fitting.")

# %% [markdown]
# **Re-fit Tomato with chosen n_clusters**

# %% [DISABLED — re-fit step commented out]
# from gudhi.clustering.tomato import Tomato
# 
# win_labels_final = {}   # win_idx → labels array (n_spikes_in_window,)
# win_peak_idx_all = {}   # win_idx → valid_peak_idx into peaks[]
# 
# print("Re-fitting windows with chosen n_clusters (skips if cached)...\n")
# 
# for f in npz_files:
#     bname    = os.path.basename(f)                            # e.g. tomato_win02_ch064.npz
#     w_idx    = int(bname.split('win')[1].split('_')[0])
#     tag      = bname.replace('tomato_', '').replace('.npz', '')   # e.g. win02_ch064
#     n_chosen = n_clusters_chosen[w_idx]
# 
#     cache_path = os.path.join(save_dir, f'final_labels_{tag}_k{n_chosen}.npy')
# 
#     d = np.load(f)
#     if 'valid_peak_idx' in d:
#         valid_peak_idx_w = d['valid_peak_idx'].astype(np.int64)
#     else:
#         # Older npz without valid_peak_idx — recompute from spike_win_all
#         all_idx  = np.where(spike_win_all == w_idx)[0]
#         n_labels = int(len(d['labels']))
#         if len(all_idx) == n_labels:
#             valid_peak_idx_w = all_idx
#         else:
#             sp_w         = peaks['sample_index'][all_idx]
#             valid_mask   = (sp_w >= pre_samp) & (sp_w < rec.get_num_samples() - post_samp)
#             valid_peak_idx_w = all_idx[valid_mask][:n_labels]
# 
#     if os.path.exists(cache_path):
#         labels_w = np.load(cache_path)
#         n_uniq   = len(np.unique(labels_w))
#         # Stale cache check: if unique labels >> n_chosen, file was saved from raw
#         # leaf_labels_ (before n_clusters_ was set) → delete and recompute.
#         if n_uniq > n_chosen * 3:
#             print(f"  win {w_idx:02d} ({tag}):  STALE CACHE (n_uniq={n_uniq} >> n_chosen={n_chosen}) — deleting")
#             os.remove(cache_path)
#             # fall through to re-fit below
#         else:
#             print(f"  win {w_idx:02d} ({tag}):  CACHED  k={n_uniq}  n_spikes={len(labels_w):,}")
# 
#     if not os.path.exists(cache_path):
#         feats_tag  = f'feats_{tag}.npy'
#         feats_path = os.path.join(save_dir, feats_tag)
#         feats_w    = np.load(feats_path)
# 
#         t_w = Tomato(dim=2, n_jobs=-1, implementation='hnsw', random_seed=42)
#         t_w.fit(feats_w)
#         actual_k        = n_chosen
#         t_w.n_clusters_ = actual_k   # gudhi renumbers labels_ to 0..actual_k-1
#         labels_w        = t_w.labels_.copy()
#         del feats_w, t_w
# 
#         np.save(cache_path, labels_w)
# 
#         suffix = f' (capped to n_cc={actual_k})' if actual_k != n_chosen else ''
#         print(f"  win {w_idx:02d} ({tag}):  FITTED  k={actual_k}{suffix}  n_spikes={len(labels_w):,}")
# 
#     win_peak_idx_all[w_idx] = valid_peak_idx_w
#     win_labels_final[w_idx] = labels_w
# 
# print(f"\nDone. {len(win_labels_final)} windows processed.")

# %%
# ── Load win_labels_final / win_peak_idx_all directly from initial Tomato results ──
# (n_clusters choosing / re-fitting step disabled above)
import glob as _glob

npz_files = sorted(_glob.glob(os.path.join(save_dir, 'tomato_win*.npz')))

win_labels_final = {}   # win_idx → labels array (n_spikes_in_window,)
win_peak_idx_all = {}   # win_idx → valid_peak_idx into peaks[]

for f in npz_files:
    bname    = os.path.basename(f)
    w_idx    = int(bname.split('win')[1].split('_')[0])
    d        = np.load(f)
    labels_w = d['labels'].astype(np.int64)
    if 'valid_peak_idx' in d:
        valid_peak_idx_w = d['valid_peak_idx'].astype(np.int64)
    else:
        # Older npz without valid_peak_idx — recompute from spike_win_all
        all_idx  = np.where(spike_win_all == w_idx)[0]
        n_labels = len(labels_w)
        if len(all_idx) == n_labels:
            valid_peak_idx_w = all_idx
        else:
            sp_w       = peaks['sample_index'][all_idx]
            valid_mask = (sp_w >= pre_samp) & (sp_w < rec.get_num_samples() - post_samp)
            valid_peak_idx_w = all_idx[valid_mask][:n_labels]
    win_labels_final[w_idx] = labels_w
    win_peak_idx_all[w_idx] = valid_peak_idx_w

print(f"Loaded {len(win_labels_final)} windows from initial Tomato results (no re-fitting).")


# %% [markdown]
# **Global cluster IDs → relabeled by mean max channel**

# %%
# ── Assign global cluster IDs (offset per window so IDs are unique) ───────────
global_labels      = np.full(len(peaks), -1, dtype=np.int64)
offset             = 0
window_n_clusters  = {}   # win_idx → n clusters in that window

for w_idx in sorted(win_labels_final.keys()):
    labels_w   = win_labels_final[w_idx]
    peak_idx_w = win_peak_idx_all[w_idx]
    # Always renumber to contiguous 0..k-1.
    # gudhi labels_ = leaf_labels_ (point indices) unless n_clusters_ was explicitly set.
    # Stale cache files may contain raw leaf indices → renumber defensively.
    unique_ids, labels_w_cont = np.unique(labels_w, return_inverse=True)
    n_clust_w  = len(unique_ids)
    global_labels[peak_idx_w] = labels_w_cont + offset
    window_n_clusters[w_idx]  = n_clust_w
    offset += n_clust_w

n_raw_clusters = offset
assigned_mask  = global_labels >= 0
n_assigned     = int(assigned_mask.sum())

print(f"Raw global clusters : {n_raw_clusters}")
print(f"Spikes assigned     : {n_assigned:,}  /  {len(peaks):,}")
print(f"Boundary-dropped    : {len(peaks) - n_assigned:,}\n")

# ── Compute mean & range of peak channel per raw cluster ──────────────────────
peak_ch_all_spikes = peaks['channel_index'][assigned_mask]
labels_asgnd       = global_labels[assigned_mask]

mean_ch_raw = np.zeros(n_raw_clusters, dtype=np.float64)
min_ch_raw  = np.zeros(n_raw_clusters, dtype=np.int64)
max_ch_raw  = np.zeros(n_raw_clusters, dtype=np.int64)
n_spk_raw   = np.zeros(n_raw_clusters, dtype=np.int64)

for c in range(n_raw_clusters):
    m = labels_asgnd == c
    if m.sum() > 0:
        chs            = peak_ch_all_spikes[m]
        mean_ch_raw[c] = float(chs.mean())
        min_ch_raw[c]  = int(chs.min())
        max_ch_raw[c]  = int(chs.max())
        n_spk_raw[c]   = int(m.sum())

print(f"{'Cluster':>8}  {'mean_ch':>8}  {'ch_range':>14}  {'n_spikes':>9}")
print("─" * 55)
for c in range(n_raw_clusters):
    print(f"{c:8d}  {mean_ch_raw[c]:8.1f}  [{min_ch_raw[c]:4d}–{max_ch_raw[c]:4d}]  {n_spk_raw[c]:9,}")

# ── Relabel clusters by mean peak-channel (ascending: 0 = smallest channel) ──
new_order  = np.argsort(mean_ch_raw)     # new_order[i] = old id with i-th smallest mean_ch
old_to_new = np.empty(n_raw_clusters, dtype=np.int64)
for new_id, old_id in enumerate(new_order):
    old_to_new[old_id] = new_id

global_labels_final = np.full_like(global_labels, -1)
global_labels_final[assigned_mask] = old_to_new[labels_asgnd]

n_global_clusters = n_raw_clusters

print(f"\nRelabeled: cluster 0 = mean_ch {mean_ch_raw[new_order[0]]:.1f} "
      f"(smallest),  cluster {n_global_clusters-1} = mean_ch {mean_ch_raw[new_order[-1]]:.1f} (largest)")


# %% [markdown]
# **Summary: per-window and global, with bar chart**

# %%
# ── Per-window summary ─────────────────────────────────────────────────────────
print(f"{'='*65}")
print(f"PER-WINDOW SUMMARY")
print(f"{'='*65}")
print(f"{'Win':>4}  {'n_clust':>7}  {'assigned':>9}  {'total_in_win':>13}  {'coverage':>9}")
print(f"{'─'*57}")

for w_idx in sorted(win_labels_final.keys()):
    peak_idx_w = win_peak_idx_all[w_idx]
    n_c        = window_n_clusters[w_idx]
    n_all_w    = int((spike_win_all == w_idx).sum())
    cov        = len(peak_idx_w) / max(1, n_all_w) * 100
    print(f"{w_idx:4d}  {n_c:7d}  {len(peak_idx_w):9,}  {n_all_w:13,}  {cov:8.1f}%")

total_assigned = sum(len(v) for v in win_peak_idx_all.values())
print(f"\n  Totals: {n_global_clusters} clusters | {total_assigned:,} assigned | "
      f"{len(peaks):,} total peaks | {len(peaks) - total_assigned:,} unassigned (boundary drops)\n")

# ── Global summary (relabeled order) ──────────────────────────────────────────
spk_per_cl  = np.zeros(n_global_clusters, dtype=np.int64)
mean_ch_fin = np.zeros(n_global_clusters, dtype=np.float64)
for new_id in range(n_global_clusters):
    old_id           = new_order[new_id]
    spk_per_cl[new_id]  = n_spk_raw[old_id]
    mean_ch_fin[new_id] = mean_ch_raw[old_id]

print(f"{'='*65}")
print(f"GLOBAL SUMMARY  ({n_global_clusters} clusters, relabeled by mean max-channel)")
print(f"{'='*65}")
print(f"{'Cluster':>8}  {'mean_ch':>8}  {'n_spikes':>9}")
print(f"{'─'*30}")
for c in range(n_global_clusters):
    print(f"{c:8d}  {mean_ch_fin[c]:8.1f}  {spk_per_cl[c]:9,}")
print(f"{'─'*30}")
print(f"{'TOTAL':>8}  {'':>8}  {int(spk_per_cl.sum()):9,}")
print(f"\n  peaks array total: {len(peaks):,}  |  unassigned: {len(peaks) - int(spk_per_cl.sum()):,}")

# ── Bar chart ──────────────────────────────────────────────────────────────────
fig_w = min(max(8, n_global_clusters // 2), 40)
fig, ax = plt.subplots(figsize=(fig_w, 4), layout='tight')
ax.bar(range(n_global_clusters), spk_per_cl,
       color='steelblue', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Cluster ID  (0 = smallest mean peak channel)')
ax.set_ylabel('Spike count')
ax.set_title(f'{n_global_clusters} clusters  |  {int(spk_per_cl.sum()):,} assigned spikes  '
             f'(of {len(peaks):,} total peaks)')
if n_global_clusters <= 60:
    ax.set_xticks(range(n_global_clusters))
    ax.tick_params(axis='x', labelsize=max(5, 8 - n_global_clusters // 15))
cluster_bar_path = os.path.join(plots_dir, '03_cluster_spike_counts.png')
plt.savefig(cluster_bar_path, dpi=100, bbox_inches='tight')
print(f'Saved: {cluster_bar_path}')
plt.close()

# %% [markdown]
# **Save SpikeInterface sorting for build_export_to_phy**
# 
# Creates a folder-based `NumpySorting` that `build_export_to_phy.py` loads with `si.load(SORTING_PATH)`.
# The paths in `build_export_to_phy.py` already point to the right locations.

# %%
import shutil as _shutil

# ── Name sorting folder by sample range so runs don't overwrite each other ──
sorting_save_path = os.path.join(data_path, f'tomato_sorting2{output_suffix}')

# ── Build per-unit spike trains (only assigned spikes, global_labels_final >= 0) ─
sp_sample_all   = peaks['sample_index']
spike_trains_si = {}
for uid in range(n_global_clusters):
    mask = global_labels_final == uid
    spike_trains_si[uid] = np.sort(sp_sample_all[mask]).astype(np.int64)

# ── Create NumpySorting ────────────────────────────────────────────────────────
# Try the modern SI API first (>= 0.100: from_dict takes a list of one segment dict)
try:
    sorting = si.NumpySorting.from_dict(
        [spike_trains_si],
        sampling_frequency=float(fs)
    )
except Exception as _e1:
    try:
        # Older SI API
        sorting = si.NumpySorting.from_unit_dict(
            spike_trains_si,
            sampling_frequency=float(fs)
        )
    except Exception as _e2:
        raise RuntimeError(f"Could not create NumpySorting:\n  {_e1}\n  {_e2}")

print(f"Created NumpySorting:")
print(f"  n_units  : {len(sorting.unit_ids)}")
print(f"  unit_ids : {sorting.unit_ids[:8]}...  (type={type(sorting.unit_ids[0]).__name__})")
print(f"  n_spikes : {sum(len(v) for v in spike_trains_si.values()):,}")
print(f"  unassigned (excluded) : {int((global_labels_final == -1).sum()):,}")

# ── Save to disk (overwrite if exists) ────────────────────────────────────────
if os.path.exists(sorting_save_path):
    _shutil.rmtree(sorting_save_path)
sorting.save(folder=sorting_save_path)

print(f"\nSaved to: {sorting_save_path}")

print(f"\n{'='*70}")
print(f"TOMATO CLUSTERING COMPLETE")
print(f"{'='*70}")
print(f"\n✓ OUTPUTS SAVED:")
print(f"\n1. PLOTS (saved to {plots_dir}):")
print(f"   - 01_sample_traces.png          : Recording sample traces")
print(f"   - 02_sample_waveforms.png       : 10 random spike waveforms")
print(f"   - 03_cluster_spike_counts.png   : Bar chart of spike counts per cluster")
print(f"   - persistence_win*.png          : Persistence diagram for each window ({len([f for f in os.listdir(save_dir) if 'persistence_' in f])} files)")

print(f"\n2. CLUSTERING RESULTS (saved to {save_dir}):")
print(f"   - feats_*.npy                   : Extracted waveform features per window")
print(f"   - tomato_*.npz                  : Tomato clustering results per window")
print(f"   - final_labels_*.npy            : Final cluster labels per window")

print(f"\n3. SORTING (saved to {sorting_save_path}):")
print(f"   - spike_times.npy               : Global spike times")
print(f"   - spike_clusters.npy            : Global cluster assignments")
print(f"   - unit_ids.txt                  : Unit IDs")
print(f"   - properties.json               : Unit properties")

print(f"\n4. SUMMARY:")
print(f"   - {n_global_clusters} clusters detected")
print(f"   - {int(spk_per_cl.sum()):,} spikes assigned")
print(f"   - {len(peaks) - int(spk_per_cl.sum()):,} spikes unassigned (boundary drops)")

# ── Save run summary ──────────────────────────────────────────────────────────
_summary = dict(
    timestamp_end       = _dt.datetime.now().isoformat(),
    n_global_clusters   = int(n_global_clusters),
    n_spikes_assigned   = int(spk_per_cl.sum()),
    n_spikes_unassigned = int(len(peaks) - int(spk_per_cl.sum())),
    n_spikes_total      = int(len(peaks)),
    sorting_save_path   = sorting_save_path,
    per_cluster         = [
        dict(cluster_id=int(c), mean_ch=float(mean_ch_fin[c]), n_spikes=int(spk_per_cl[c]))
        for c in range(n_global_clusters)
    ],
    per_window          = [
        dict(win_idx=int(w), n_clusters=int(window_n_clusters[w]),
             n_assigned=int(len(win_peak_idx_all[w])),
             n_total=int((spike_win_all == w).sum()))
        for w in sorted(win_labels_final.keys())
    ],
)
_summary_path = os.path.join(save_dir, 'run_summary.json')
with open(_summary_path, 'w') as _f:
    _json.dump(_summary, _f, indent=2)
print(f'\nSaved summary : {_summary_path}')
print(f'Saved params  : {_params_path}')

