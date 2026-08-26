#!/usr/bin/env python3
"""
Optional refinement, not always needed — use if a single motion estimate
over the whole depth range is unreliable at the shallow/deep extremes (too
few peaks there). Combines several motion estimates that were each run on a
restricted depth range (e.g. peaks above a certain amplitude, or peaks within
a depth band) into one Motion object covering the full probe. Run by hand
after producing the relevant per-depth-range estimates with run_dredge_chunked.py.

Combine depth-specific motion estimates into unified Motion objects, by
direct concatenation:
    935–2885 µm   → chunks_n30/motion_dredge_amp20pct  (reliable central range)
    < 700 µm      → chunks_n32/motion_dredge_d0-700um  (estimated on shallow peaks)
    > 3200 µm     → chunks_n30/motion_dredge_d3200um+  (estimated on deep peaks)

Bins at 785, 3035, 3185 are absent (true gaps between the disjoint source
ranges, no source covers them); SpikeInterface's own linear interpolation
fills those automatically at correction time, using the nearest real bins on
either side.

A provenance.json is written into the output folder documenting the sources,
depth ranges, and bin-selection details for full reproducibility.

USAGE: no CLI — edit BASE (session's motion_estimates/ dir) and the depth-range
-> source-folder mapping in the CONFIG section below to match whichever
per-depth-range estimates you actually produced with run_dredge_chunked.py,
then `python combine_motions.py`.

OUTPUTS, under BASE/:
  chunks_combined/           — the combined Motion object (si.Motion.load(...))
  chunks_combined/*/provenance.json   — which source folder covered which depth range, for reproducibility.
"""

import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import spikeinterface as si
from spikeinterface.core import Motion

BASE = Path('/ptmp/gongt/GLC7189/GLC7189_g_g1/preprocessed/thresholding_peaks_3.3/motion_estimates')

# Depth range kept verbatim from amp20pct (determined by window reliability analysis)
AMP_DEPTH_MIN  = 935.0
AMP_DEPTH_MAX  = 2885.0

# Cutoffs for source selection in the direct strategy
DEPTH_LOW_CUT  = 700.0    # bins strictly below this come from d0-700um
DEPTH_HIGH_CUT = 3200.0   # bins strictly above this come from d3200um+


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _assert_temporal_match(m_amp, m_high, m_low):
    t = m_amp.temporal_bins_s[0]
    assert np.allclose(t, m_high.temporal_bins_s[0], atol=0.01), \
        "temporal bins mismatch: amp vs high"
    assert np.allclose(t, m_low.temporal_bins_s[0], atol=0.01), \
        "temporal bins mismatch: amp vs low"


def combine_direct(m_low, m_amp, m_high, low_dir, amp_dir, high_dir):
    """Direct concatenation.

    Output spatial bins (22 total):
      [35, 185, 335, 485, 635]   from d0-700um  (bins < 700 µm)
      [935, 1085, …, 2885]       from amp20pct  (bins 935–2885 µm, 14 bins)
      [3335, 3485, 3635]         from d3200um+  (bins > 3200 µm)

    Gaps at 785, 3035, 3185 µm are intentionally absent; SpikeInterface's
    linear interpolation fills them automatically during motion correction.
    """
    _assert_temporal_match(m_amp, m_high, m_low)

    low_mask  = m_low.spatial_bins_um  < DEPTH_LOW_CUT
    mid_mask  = ((m_amp.spatial_bins_um >= AMP_DEPTH_MIN) &
                 (m_amp.spatial_bins_um <= AMP_DEPTH_MAX))
    high_mask = m_high.spatial_bins_um > DEPTH_HIGH_CUT

    low_bins  = m_low.spatial_bins_um[low_mask]     # [35, 185, 335, 485, 635]
    mid_bins  = m_amp.spatial_bins_um[mid_mask]     # [935, …, 2885]
    high_bins = m_high.spatial_bins_um[high_mask]   # [3335, 3485, 3635]

    bins = np.concatenate([low_bins, mid_bins, high_bins])
    disp = np.concatenate([
        m_low.displacement[0][:, low_mask],
        m_amp.displacement[0][:, mid_mask],
        m_high.displacement[0][:, high_mask],
    ], axis=1)

    motion = Motion(
        displacement=disp,
        temporal_bins_s=m_amp.temporal_bins_s[0],
        spatial_bins_um=bins,
        direction='y',
        interpolation_method='linear',
    )

    provenance = {
        'strategy': 'direct_concatenation',
        'description': (
            'Displacement at each spatial bin is copied verbatim from the source '
            'motion whose peak-selection depth range covers that bin. '
            'Bins 785, 3035, 3185 µm are absent (gap between source ranges); '
            'SpikeInterface linearly interpolates them at correction time.'
        ),
        'amp20pct_depth_range_um': [AMP_DEPTH_MIN, AMP_DEPTH_MAX],
        'note_on_range': (
            'The 935–2885 µm range was chosen based on the window reliability '
            'analysis (test_motion_overlay_reliability_combined plots).'
        ),
        'output_spatial_bins_um': bins.tolist(),
        'sources': [
            {
                'role': 'shallow',
                'source_folder': str(Path(low_dir).relative_to(BASE)),
                'peak_selection': f'peaks localised to y < {DEPTH_LOW_CUT:.0f} µm',
                'bins_used_um': low_bins.tolist(),
                'bin_selection_criterion': f'source bins where depth < {DEPTH_LOW_CUT:.0f} µm',
                'method': 'direct — displacement copied verbatim',
            },
            {
                'role': 'middle',
                'source_folder': str(Path(amp_dir).relative_to(BASE)),
                'peak_selection': 'all peaks above amplitude percentile threshold (amp20pct)',
                'bins_used_um': mid_bins.tolist(),
                'bin_selection_criterion': (
                    f'source bins where {AMP_DEPTH_MIN:.0f} µm ≤ depth ≤ {AMP_DEPTH_MAX:.0f} µm '
                    '(reliable range from window reliability analysis)'
                ),
                'method': 'direct — displacement copied verbatim',
            },
            {
                'role': 'deep',
                'source_folder': str(Path(high_dir).relative_to(BASE)),
                'peak_selection': f'peaks localised to y > {DEPTH_HIGH_CUT:.0f} µm',
                'bins_used_um': high_bins.tolist(),
                'bin_selection_criterion': f'source bins where depth > {DEPTH_HIGH_CUT:.0f} µm',
                'method': 'direct — displacement copied verbatim',
            },
        ],
    }

    return motion, provenance


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    n30 = BASE / 'chunks_n30'
    n32 = BASE / 'chunks_n32'
    out1 = BASE / 'chunks_combined'
    out1.mkdir(exist_ok=True)

    amp_dirs = sorted(n30.glob('motion_dredge_amp20pct_*'))
    if not amp_dirs:
        raise RuntimeError(f"No amp20pct folders found under {n30}")

    for amp_dir in amp_dirs:
        # time tag like '0000min_0070min'
        parts = amp_dir.name.split('_')
        time_tag = f'{parts[-2]}_{parts[-1]}'

        high_dir = n30 / f'motion_dredge_d3200um+_{time_tag}'
        low_dir  = n32 / f'motion_dredge_d0-700um_{time_tag}'

        for d in (high_dir, low_dir):
            if not d.exists():
                print(f'SKIP {time_tag}: missing {d}')
                break
        else:
            m_amp  = si.Motion.load(str(amp_dir))
            m_high = si.Motion.load(str(high_dir))
            m_low  = si.Motion.load(str(low_dir))

            name = f'motion_dredge_{time_tag}'
            created = datetime.now(timezone.utc).isoformat()

            save1 = out1 / name
            if save1.exists():
                print(f'EXISTS  {save1.name}')
            else:
                m1, prov1 = combine_direct(m_low, m_amp, m_high, low_dir, amp_dir, high_dir)
                m1.save(str(save1))
                print(f'SAVED   {save1.name}  bins={m1.spatial_bins_um[[0,-1]]}')
            prov1_path = save1 / 'provenance.json'
            if not prov1_path.exists():
                prov1_data = {
                    'created_utc': created,
                    'script': str(Path(__file__).resolve()),
                    'spikeinterface_version': si.__version__,
                    **prov1,
                }
                prov1_path.write_text(json.dumps(prov1_data, indent=2))
                print(f'WROTE   {prov1_path.relative_to(BASE)}')

    print('\nDone.')


if __name__ == '__main__':
    main()
