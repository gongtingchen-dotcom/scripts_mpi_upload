#!/usr/bin/env python

import os
import argparse
import yaml
import logging
from copy import deepcopy

import spikeinterface as si
import spikeinterface.preprocessing as spre
import mountainsort5 as ms5
from mountainsort5.util import create_cached_recording
from spikeinterface.exporters import export_to_phy
from tempfile import TemporaryDirectory


# =========================
# Parameter abbreviations
# =========================
PARAM_ABBREV = {
    "phase1_detect_channel_radius": "p1r",
    "detect_channel_radius": "r",
    "phase1_detect_threshold": "p1thr",
    "detect_threshold": "thr",
    "detect_sign": "sgn",
    "phase1_detect_time_radius_msec": "p1t",
    "detect_time_radius_msec": "t",
    "phase1_npca_per_channel": "p1pc",
    "phase1_npca_per_subdivision": "p1ps",
    "classifier_npca": "clspc",
    "snippet_T1": "T1",
    "snippet_T2": "T2",
    "snippet_mask_radius": "smr",
    "max_num_snippets_per_training_batch": "mb",
    "training_duration_sec": "td",
    "training_recording_sampling_mode": "tsm",
    "classification_chunk_sec": "cc",
    "phase1_pairwise_merge_step": "p1merge",
}


# =========================
# Utilities
# =========================
def build_run_name(defaults, current):
    """Generate folder name from changed parameters"""
    tags = []
    for k, v in current.items():
        if defaults[k] != v:
            abbr = PARAM_ABBREV.get(k, k)
            tags.append(f"{abbr}-{v}")
    return "_".join(tags) if tags else "default"


def save_params(folder, params_dict, scheme_obj):
    """Save params in multiple formats for reproducibility"""
    os.makedirs(folder, exist_ok=True)

    with open(os.path.join(folder, "params.yaml"), "w") as f:
        yaml.dump(params_dict, f, sort_keys=False)

def nullable_float(val):
    if val is None or str(val).lower() in ["none", "null", ""]:
        return None
    return float(val)

# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # --- Paths ---
    parser.add_argument("--root_folder", required=True)
    parser.add_argument("--data_folder", required=True)

    # --- Detection geometry ---
    parser.add_argument("--phase1_detect_channel_radius", type=nullable_float, default=200)
    parser.add_argument("--detect_channel_radius", type=nullable_float, default=100)

    # --- Thresholds ---
    parser.add_argument("--phase1_detect_threshold", type=float, default=5.5)
    parser.add_argument("--detect_threshold", type=float, default=5.5)
    parser.add_argument("--detect_sign", type=int, default=-1)

    # --- Temporal radii (ms) ---
    parser.add_argument("--phase1_detect_time_radius_msec", type=float, default=1.5)
    parser.add_argument("--detect_time_radius_msec", type=float, default=0.5)

    # --- PCA ---
    parser.add_argument("--phase1_npca_per_channel", type=int, default=3)
    parser.add_argument("--phase1_npca_per_subdivision", type=int, default=10)
    parser.add_argument("--classifier_npca", type=int, default=None)

    # --- Snippets ---
    parser.add_argument("--snippet_T1", type=int, default=20)
    parser.add_argument("--snippet_T2", type=int, default=20)
    parser.add_argument("--snippet_mask_radius", type=float, default=None)

    # --- Training ---
    parser.add_argument("--max_num_snippets_per_training_batch", type=int, default=200)
    parser.add_argument("--training_duration_sec", type=float, default=None)
    parser.add_argument("--training_recording_sampling_mode", type=str, default="initial")
    parser.add_argument("--classification_chunk_sec", type=float, default=None)

    # --- Merging ---
    parser.add_argument("--phase1_pairwise_merge_step", action="store_true")

    args = parser.parse_args()
    args_dict = vars(args)

    # ==========================================
    # Build scheme and defaults (Robust Version)
    # ==========================================
    # We define the defaults manually to mirror argparse.
    # This avoids calling .dict() or .model_dump() on the ms5 object.
    default_dict = {
        "phase1_detect_channel_radius": 200.0,
        "detect_channel_radius": 100.0,
        "phase1_detect_threshold": 5.5,
        "detect_threshold": 5.5,
        "detect_sign": -1,
        "phase1_detect_time_radius_msec": 1.5,
        "detect_time_radius_msec": 0.5,
        "phase1_npca_per_channel": 3,
        "phase1_npca_per_subdivision": 10,
        "classifier_npca": None,
        "snippet_T1": 20,
        "snippet_T2": 20,
        "snippet_mask_radius": None,
        "max_num_snippets_per_training_batch": 200,
        "training_duration_sec": None,
        "training_recording_sampling_mode": "initial",
        "classification_chunk_sec": None,
        "phase1_pairwise_merge_step": False,
    }

    # Extract only the valid parameters for MS5 from the command line args
    scheme_params = {
        k: args_dict[k]
        for k in default_dict.keys()
        if k in args_dict
    }

    sorting_parameters = ms5.Scheme2SortingParameters(**scheme_params)

    # =========================
    # Run naming
    # =========================
    run_name = build_run_name(default_dict, scheme_params)
    exp_folder = os.path.join(args.root_folder, f"ms5_{run_name}")
    os.makedirs(exp_folder, exist_ok=True)

    # =========================
    # Logging
    # =========================
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(exp_folder, "ms5.log"),
        filemode="w",
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

    logging.info("Run name: %s", run_name)
    logging.info("Parameters: %s", scheme_params)

    # =========================
    # Save parameters
    # =========================
    save_params(exp_folder, scheme_params, sorting_parameters)

    # =========================
    # Load + preprocess
    # =========================
    rec = si.load_extractor(args.data_folder)
    #rec = spre.whiten(rec)
    ptmp_base = "/ptmp/gongt" if os.path.exists("/ptmp") else None

    # Use a context manager to handle creation and deletion
    with TemporaryDirectory(dir=ptmp_base, prefix="si_cache_") as tmpdir:
        logging.info(f"Caching recording to temporary directory: {tmpdir}")
        
        # Cache the recording into the temp folder
        rec_cached = create_cached_recording(rec, folder=tmpdir, n_jobs = -1, chunk_duration="1s")

        # =========================
        # Run MS5
        # =========================
        
        sorting = ms5.sorting_scheme2(
            recording=rec_cached,
            sorting_parameters=sorting_parameters,
        )
        
        logging.info("Sorting finished. Temporary cache will now be deleted.")

    # =========================
    # Analyzer
    # =========================
    analyzer_folder = os.path.join(exp_folder, "sorting_analyzer")
    analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=rec,
        folder=analyzer_folder,
        format="binary_folder",
    )


    analyzer.compute(
        {
            'random_spikes': {}, 
            'waveforms': {},
            'templates': {},
            'noise_levels': {},
            'spike_amplitudes': {},
            'unit_locations': {'method': 'monopolar_triangulation'},
            'spike_locations': {'method': 'center_of_mass'},
            'template_similarity': {}, 
            'quality_metrics': {
                'metric_names': ['snr', 'isi_violation', 'presence_ratio', 'amplitude_cutoff']
            },
            'principal_components': {'n_components': 6, 'whiten': False},
        },
        save=True,
        progress_bar=True,
        n_jobs=-1,
    )

    # =========================
    # Export to Phy
    # =========================
    export_to_phy(
        sorting_analyzer=analyzer,
        output_folder=os.path.join(exp_folder, "phy"),
        copy_binary=False,
        remove_if_exists=True,
        n_jobs=-1,
    )

    logging.info("Exported to Phy")

    import shutil
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        slurm_log = f"./slurm/{slurm_job_id}/logs/training.log"
        if os.path.exists(slurm_log):
            shutil.copy(slurm_log, os.path.join(exp_folder, "slurm.log"))


if __name__ == "__main__":
    main()