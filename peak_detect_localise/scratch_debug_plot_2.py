import spikeinterface.full as si
import numpy as np
import matplotlib.pyplot as plt

rec_dir = '/ptmp/gongt/GLC6962_l_g1/preprocessed'
ks4_dir = '/ptmp/gongt/GLC6962_l_g1/KS4/test_MG_params2'
rec = si.load(rec_dir)
sorting = si.read_kilosort(ks4_dir + '/sorter_output')

unit_id = 32
fs = rec.get_sampling_frequency()

spike_times = sorting.get_unit_spike_train(unit_id, segment_index=0)
timepoint_samples = int((1568.47+1.5) * fs)
idx_nearest = np.argmin(np.abs(spike_times - timepoint_samples))
center_sample = spike_times[idx_nearest]

channel_ids = rec.get_channel_ids()
probe = rec.get_probe()
positions = probe.contact_positions

templates = np.load(f'{ks4_dir}/sorter_output/templates.npy')
spike_templates = np.load(f'{ks4_dir}/sorter_output/spike_templates.npy')
spike_clusters = np.load(f'{ks4_dir}/sorter_output/spike_clusters.npy')

unit_mask = spike_clusters == unit_id
unit_template_idx = int(np.bincount(spike_templates[unit_mask]).argmax())
template = templates[unit_template_idx]
best_ch_idx = np.argmin(np.min(template, axis=0))

dists = np.linalg.norm(positions - positions[best_ch_idx], axis=1)
nearest_ch_indices = np.argsort(dists)[:32]

win_ms = 3.0
half_win = int(win_ms / 2 / 1000 * fs)
start = int(center_sample) - half_win
end   = int(center_sample) + half_win

traces = rec.get_traces(start_frame=start, end_frame=end, segment_index=0)

t_ms = np.linspace(-win_ms/2, win_ms/2, traces.shape[0])
center_idx = traces.shape[0] // 2

fig, ax = plt.subplots(figsize=(10, 6))
offset_step = np.max(np.abs(traces[:, nearest_ch_indices])) * 1.5

for i, ch_idx in enumerate(nearest_ch_indices):
    tr = traces[:, ch_idx]
    ax.plot(t_ms, tr - i*offset_step, color='k', lw=1)
    ax.plot(t_ms[center_idx], tr[center_idx] - i*offset_step, 'ro', markersize=4)

plt.savefig('/u/gongt/.gemini/antigravity/artifacts/debug_plot.png')
