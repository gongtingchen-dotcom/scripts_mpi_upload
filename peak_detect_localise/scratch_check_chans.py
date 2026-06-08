import spikeinterface.full as si
import numpy as np

rec_dir = '/ptmp/gongt/GLC6962_l_g1/preprocessed'
ks4_dir = '/ptmp/gongt/GLC6962_l_g1/KS4/test_MG_params2/sorter_output'

rec = si.load(rec_dir)
chan_map = np.load(f'{ks4_dir}/channel_map.npy')

print("Recording channel IDs:", rec.get_channel_ids()[:10])
print("KS channel map:", chan_map[:10])
print("Are they identical?", np.array_equal(rec.get_channel_ids(), chan_map))

