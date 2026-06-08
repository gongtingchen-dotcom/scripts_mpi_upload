import numpy as np

ks4_dir = '/ptmp/gongt/GLC6962_l_g1/KS4/test_MG_params2'
templates = np.load(f'{ks4_dir}/sorter_output/templates.npy')

print(f"Templates shape: {templates.shape}")
# shape is usually (n_templates, n_time_samples, n_channels)

# Find the center/peak for each template across time
# The best channel is the one with the maximum peak-to-peak amplitude
best_channels = np.argmax(np.ptp(templates, axis=1), axis=1)

peaks = []
for i in range(templates.shape[0]):
    best_ch = best_channels[i]
    # Kilosort templates are usually negative-going (extracellular spikes)
    trough_idx = np.argmin(templates[i, :, best_ch])
    peaks.append(trough_idx)

peaks = np.array(peaks)
print(f"Mean trough index: {np.mean(peaks)}")
print(f"Median trough index: {np.median(peaks)}")
print(f"Unique trough indices: {np.unique(peaks)}")
