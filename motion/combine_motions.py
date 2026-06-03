#!/usr/bin/env python3
"""
Combine depth-specific motion estimates into unified Motion objects.

Two output strategies, both with the same spatial bin layout:

chunks_combined  (direct concatenation)
  935–2885 µm   → chunks_n30/motion_dredge_amp20pct  (reliable central range)
  < 700 µm      → chunks_n32/motion_dredge_d0-700um  (estimated on shallow peaks)
  > 3200 µm     → chunks_n30/motion_dredge_d3200um+  (estimated on deep peaks)
  Bins at 785, 3035, 3185 are absent (no direct source in the gap regions).

chunks_combined_linear  (linear interpolation at boundaries)
  935–2885 µm              → amp20pct directly (same as above, identical values)
  35 µm anchor             → d0-700um directly
  185, 335, 485, 635, 785  → linearly interpolated between d0-700um@35 and amp20pct@935
  3035, 3185, 3335, 3485   → linearly interpolated between amp20pct@2885 and d3200um+@3635
  3635 µm anchor           → d3200um+ directly

A provenance.json is written into every output folder documenting the sources,
depth ranges, and interpolation details for full reproducibility.
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
    """Strategy 1: direct concatenation.

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


def combine_linear(m_low, m_amp, m_high, low_dir, amp_dir, high_dir):
    """Strategy 2: amp20pct bins 935–2885 µm kept verbatim; all other bins
    filled by linear interpolation between anchor motions.

    Output spatial bins (25 total, 35–3635 µm at 150 µm steps):
      35                         anchor — d0-700um directly
      185, 335, 485, 635, 785    interpolated: d0-700um@35 → amp20pct@935
      935, 1085, …, 2885         amp20pct directly (14 bins)
      3035, 3185, 3335, 3485     interpolated: amp20pct@2885 → d3200um+@3635
      3635                       anchor — d3200um+ directly
    """
    _assert_temporal_match(m_amp, m_high, m_low)

    # Core: amp20pct bins 935–2885
    mid_mask = ((m_amp.spatial_bins_um >= AMP_DEPTH_MIN) &
                (m_amp.spatial_bins_um <= AMP_DEPTH_MAX))
    mid_bins = m_amp.spatial_bins_um[mid_mask]      # [935, …, 2885]
    mid_disp = m_amp.displacement[0][:, mid_mask]   # (n_time, 14)

    # Bottom anchor: d0-700um @ 35 µm
    lo_d    = m_low.spatial_bins_um[0]              # 35.0
    lo_disp = m_low.displacement[0][:, 0]           # (n_time,)

    # Top of bottom / bottom of core: amp20pct @ 935 µm
    hi_d    = mid_bins[0]                           # 935.0
    hi_disp = mid_disp[:, 0]                        # (n_time,)

    # Interpolated bins 185, 335, 485, 635, 785 (between 35 and 935)
    interp_lo_bins = np.arange(lo_d + 150, hi_d, 150)
    t_lo = (interp_lo_bins - lo_d) / (hi_d - lo_d)
    interp_lo_disp = (
        (1 - t_lo)[np.newaxis, :] * lo_disp[:, np.newaxis]
        + t_lo[np.newaxis, :] * hi_disp[:, np.newaxis]
    )   # (n_time, 5)

    # Bottom of top / top of core: amp20pct @ 2885 µm
    lo_top_d    = mid_bins[-1]                      # 2885.0
    lo_top_disp = mid_disp[:, -1]                   # (n_time,)

    # Top anchor: d3200um+ @ 3635 µm
    hi_top_d    = m_high.spatial_bins_um[-1]        # 3635.0
    hi_top_disp = m_high.displacement[0][:, -1]    # (n_time,)

    # Interpolated bins 3035, 3185, 3335, 3485 (between 2885 and 3635)
    interp_hi_bins = np.arange(lo_top_d + 150, hi_top_d, 150)
    t_hi = (interp_hi_bins - lo_top_d) / (hi_top_d - lo_top_d)
    interp_hi_disp = (
        (1 - t_hi)[np.newaxis, :] * lo_top_disp[:, np.newaxis]
        + t_hi[np.newaxis, :] * hi_top_disp[:, np.newaxis]
    )   # (n_time, 4)

    all_bins = np.concatenate([
        [lo_d],             # 35
        interp_lo_bins,     # 185, 335, 485, 635, 785
        mid_bins,           # 935 … 2885
        interp_hi_bins,     # 3035, 3185, 3335, 3485
        [hi_top_d],         # 3635
    ])
    all_disp = np.concatenate([
        lo_disp[:, np.newaxis],
        interp_lo_disp,
        mid_disp,
        interp_hi_disp,
        hi_top_disp[:, np.newaxis],
    ], axis=1)

    motion = Motion(
        displacement=all_disp,
        temporal_bins_s=m_amp.temporal_bins_s[0],
        spatial_bins_um=all_bins,
        direction='y',
        interpolation_method='linear',
    )

    def _lerp_entries(bins, t_arr, src_lo, d_lo, src_hi, d_hi):
        return [
            {
                'depth_um': float(d),
                'weight_lo': round(float(1 - t), 6),
                'weight_hi': round(float(t), 6),
                'formula': (
                    f'(1-{t:.4f}) * {src_lo}@{d_lo:.0f}µm '
                    f'+ {t:.4f} * {src_hi}@{d_hi:.0f}µm'
                ),
            }
            for d, t in zip(bins, t_arr)
        ]

    amp_name  = Path(amp_dir).name
    low_name  = Path(low_dir).name
    high_name = Path(high_dir).name

    provenance = {
        'strategy': 'linear_interpolation_at_boundaries',
        'description': (
            'The reliable central range (935–2885 µm) from amp20pct is kept verbatim. '
            'All other spatial bins are filled by linear interpolation between two '
            'anchor motions: d0-700um at 35 µm (shallow) and d3200um+ at 3635 µm (deep). '
            'This produces smooth displacement transitions without hard depth boundaries.'
        ),
        'amp20pct_depth_range_um': [AMP_DEPTH_MIN, AMP_DEPTH_MAX],
        'note_on_range': (
            'The 935–2885 µm range was chosen based on the window reliability '
            'analysis (test_motion_overlay_reliability_combined plots).'
        ),
        'output_spatial_bins_um': all_bins.tolist(),
        'sources': [
            {
                'role': 'shallow_anchor',
                'source_folder': str(Path(low_dir).relative_to(BASE)),
                'peak_selection': f'peaks localised to y < {DEPTH_LOW_CUT:.0f} µm',
                'bins_um': [float(lo_d)],
                'method': 'direct — displacement copied verbatim',
            },
            {
                'role': 'shallow_interpolated',
                'bins_um': interp_lo_bins.tolist(),
                'method': (
                    f'linear interpolation between '
                    f'{low_name}@{lo_d:.0f}µm (w=1-t) and '
                    f'{amp_name}@{hi_d:.0f}µm (w=t)'
                ),
                'per_bin': _lerp_entries(
                    interp_lo_bins, t_lo, low_name, lo_d, amp_name, hi_d
                ),
            },
            {
                'role': 'middle',
                'source_folder': str(Path(amp_dir).relative_to(BASE)),
                'peak_selection': 'all peaks above amplitude percentile threshold (amp20pct)',
                'bins_um': mid_bins.tolist(),
                'bin_selection_criterion': (
                    f'{AMP_DEPTH_MIN:.0f}–{AMP_DEPTH_MAX:.0f} µm '
                    '(reliable range from window reliability analysis)'
                ),
                'method': 'direct — displacement copied verbatim',
            },
            {
                'role': 'deep_interpolated',
                'bins_um': interp_hi_bins.tolist(),
                'method': (
                    f'linear interpolation between '
                    f'{amp_name}@{lo_top_d:.0f}µm (w=1-t) and '
                    f'{high_name}@{hi_top_d:.0f}µm (w=t)'
                ),
                'per_bin': _lerp_entries(
                    interp_hi_bins, t_hi, amp_name, lo_top_d, high_name, hi_top_d
                ),
            },
            {
                'role': 'deep_anchor',
                'source_folder': str(Path(high_dir).relative_to(BASE)),
                'peak_selection': f'peaks localised to y > {DEPTH_HIGH_CUT:.0f} µm',
                'bins_um': [float(hi_top_d)],
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
    out2 = BASE / 'chunks_combined_linear'
    out1.mkdir(exist_ok=True)
    out2.mkdir(exist_ok=True)

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

            save2 = out2 / name
            if save2.exists():
                print(f'EXISTS  {save2.name}')
            else:
                m2, prov2 = combine_linear(m_low, m_amp, m_high, low_dir, amp_dir, high_dir)
                m2.save(str(save2))
                print(f'SAVED   {save2.name}  bins={m2.spatial_bins_um[[0,-1]]}')
            prov2_path = save2 / 'provenance.json'
            if not prov2_path.exists():
                prov2_data = {
                    'created_utc': created,
                    'script': str(Path(__file__).resolve()),
                    'spikeinterface_version': si.__version__,
                    **prov2,
                }
                prov2_path.write_text(json.dumps(prov2_data, indent=2))
                print(f'WROTE   {prov2_path.relative_to(BASE)}')

    print('\nDone.')


if __name__ == '__main__':
    main()
