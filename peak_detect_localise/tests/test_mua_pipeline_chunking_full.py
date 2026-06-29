import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mua_pipeline as mp


def test_full_pipeline_chunking_concatenates_chunk_outputs(monkeypatch, tmp_path):
    class FakeRecording:
        def __init__(self, n_samples):
            self._n_samples = n_samples
            self.sampling_frequency = 1000.0

        def get_num_samples(self):
            return self._n_samples

        def frame_slice(self, start_frame, end_frame):
            rec = FakeRecording(end_frame - start_frame)
            rec._start_frame = start_frame
            return rec

        def astype(self, dtype):
            return self

        def get_num_channels(self):
            return 2

    def fake_thresholding_chunked(rec, fs, noise_levels, **kwargs):
        start = getattr(rec, '_start_frame', 0)
        peak_dtype = np.dtype([
            ('sample_index', 'int64'),
            ('channel_index', 'int64'),
            ('amplitude', 'float32'),
            ('segment_index', 'int64'),
            ('z', 'int64'),
        ])
        return np.array([(start + 1, 0, 1.0, 0, 0)], dtype=peak_dtype)

    def fake_spatial_and_temporal_filter(peaks0, noise_levels, fs, **kwargs):
        return [], peaks0

    def fake_filter_peaks_by_spatial_support(peaks, rec, noise_levels, **kwargs):
        return peaks

    def fake_get_prototype_and_waveforms_from_peaks(*args, **kwargs):
        return np.zeros(3), np.zeros((1, 3)), np.zeros(1)

    def fake_detect_peaks(recording, method=None, method_kwargs=None, verbose=False, job_kwargs=None):
        peak_dtype = np.dtype([
            ('sample_index', 'int64'),
            ('channel_index', 'int64'),
            ('amplitude', 'float32'),
            ('segment_index', 'int64'),
            ('z', 'int64'),
        ])
        return np.array([(2, 0, 1.0, 0, 0)], dtype=peak_dtype)

    monkeypatch.setattr(mp, 'thresholding_chunked', fake_thresholding_chunked)
    monkeypatch.setattr(mp, 'spatial_and_temporal_filter', fake_spatial_and_temporal_filter)
    monkeypatch.setattr(mp, 'filter_peaks_by_spatial_support', fake_filter_peaks_by_spatial_support)
    monkeypatch.setattr(mp, 'get_prototype_and_waveforms_from_peaks', fake_get_prototype_and_waveforms_from_peaks)
    monkeypatch.setattr(mp, 'detect_peaks', fake_detect_peaks)

    rec = FakeRecording(200)
    file_names = {
        'peaks0': str(tmp_path / 'peaks0.npy'),
        'peaks1': str(tmp_path / 'peaks1.npy'),
        'peaks2': str(tmp_path / 'peaks2.npy'),
        'peaks_matched': str(tmp_path / 'peaks_matched.npy'),
        'peaks_combined1': str(tmp_path / 'peaks_combined1.npy'),
        'peaks_combined2': str(tmp_path / 'peaks_combined2.npy'),
    }

    out = mp._run_pipeline_in_time_chunks(
        rec,
        fs=1000.0,
        n_chans=2,
        noise_levels=np.ones(2),
        detect_threshold=3.3,
        exclude_sweep_ms=0.3,
        spatial_radius=10,
        chunk_size_s=60.0,
        n_jobs=1,
        proto_n_jobs=1,
        save_dir=str(tmp_path),
        file_names=file_names,
        n_chunks=2,
    )

    assert out is not None
    peaks0 = np.load(file_names['peaks0'])
    peaks1 = np.load(file_names['peaks1'])
    peaks2 = np.load(file_names['peaks2'])
    peaks_matched = np.load(file_names['peaks_matched'])

    assert peaks0['sample_index'].tolist() == [1, 101]
    assert peaks1['sample_index'].tolist() == [1, 101]
    assert peaks2['sample_index'].tolist() == [1, 101]
    assert peaks_matched['sample_index'].tolist() == [2, 102]
