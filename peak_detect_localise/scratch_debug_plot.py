import spikeinterface.full as si
import numpy as np

rec_dir = '/ptmp/gongt/GLC6962_l_g1/preprocessed'
ks4_dir = '/ptmp/gongt/GLC6962_l_g1/KS4/test_MG_params2/sorter_output'

rec = si.load(rec_dir)
sorting = si.read_kilosort(ks4_dir)

fs = rec.get_sampling_frequency()

# Find the best unit
amplitudes = np.load(f'{ks4_dir}/amplitudes.npy')
spike_clusters = np.load(f'{ks4_dir}/spike_clusters.npy')

unique, counts = np.unique(spike_clusters, return_counts=True)
valid_units = unique[counts > 500]

mean_amps = []
for u in valid_units:
    amps = amplitudes[spike_clusters == u]
    mean_amps.append(np.mean(amps))

best_unit = valid_units[np.argmax(mean_amps)]
print(f"Best unit: {best_unit} with mean amplitude: {np.max(mean_amps)}")

spike_times = sorting.get_unit_spike_train(best_unit, segment_index=0)

offsets = []
half_win = 60 # 60 samples
for i in range(50):
    center_sample = spike_times[i]
    start = int(center_sample) - half_win
    end = int(center_sample) + half_win

    traces = rec.get_traces(start_frame=start, end_frame=end, segment_index=0)
    best_chan = np.argmin(np.min(traces, axis=0))

    peak_idx = np.argmin(traces[:, best_chan])
    offsets.append(peak_idx - half_win)

print(f"Offsets of actual peak relative to KS spike_time (in samples):")
print(offsets)
print(f"Mean offset: {np.mean(offsets):.2f}, Median: {np.median(offsets):.2f}")

# Also let's check Kilosort's template for this unit
templates = np.load(f'{ks4_dir}/templates.npy')
spike_templates = np.load(f'{ks4_dir}/spike_templates.npy')
unit_mask = spike_clusters == best_unit
unit_template_idx = int(np.bincount(spike_templates[unit_mask]).argmax())
template = templates[unit_template_idx]
best_ch = np.argmin(np.min(template, axis=0))
template_trough = np.argmin(template[:, best_ch])
print(f"Template trough index: {template_trough} out of {template.shape[0]}")

