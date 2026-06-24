#!/usr/bin/env python3
from __future__ import annotations

"""
Entry point for value-bin-by-delay MVPA.

For a given delay tertile level (--delay-bin 1|2|3) this script:
  1. Loads behavioral data for all subjects to compute global tertile edges.
  2. For each subject, extracts beta-series patterns and averages within
     delay × value tertile bins to produce one pattern per value bin.
  3. Runs leave-one-subject-out logistic regression (L2, OVR, LBFGS) to
     classify which value tertile bin (A1/A2/A3) each pattern belongs to.
  4. Writes roi_summary.csv, subject_preds.csv, confusion_matrix.csv,
     meta.json, and _SUCCESS to the output directory.

Typical usage (see submit_value_bins_by_delay.sh):
  python run_mvpa_value_bins_by_delay.py --delay-bin 1 --atlas pauli_rois
"""

import argparse
import json
from pathlib import Path

import numpy as np

from dd_kable_analysis.config_loader import load_config
from dd_kable_analysis.mvpa import (
    build_subject_behav_bold_df,
    decode_value_bins_by_delay,
)
from dd_kable_analysis.mvpa.atlas import get_roi_labels_from_atlas_img, resolve_atlas
from dd_kable_analysis.mvpa.data_binned import (
    build_subject_binned_roi_patterns,
    compute_global_bin_edges,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Value-bin-by-delay MVPA: LOSO multiclass decoding (one delay level).'
    )
    p.add_argument(
        '--delay-bin', type=int, required=True, choices=[1, 2, 3],
        help='Which delay tertile level to analyse (1=shortest, 3=longest).',
    )
    p.add_argument(
        '--atlas', type=str, required=True,
        choices=['josh_orig', 'schaefer200', 'schaefer400', 'harvard_oxford_subcort', 'pauli_rois'],
    )
    p.add_argument('--atlas-path', type=str, default=None)
    p.add_argument(
        '--analysis-tag', type=str, default=None,
        help='Output subdirectory tag; default = value_bins_by_delay_<atlas>.',
    )
    p.add_argument('--sub-list', type=str, default=None,
                   help='Path to subject list file (one ID per line). '
                        'Defaults to the standard mvpa_subject_list.txt.')
    p.add_argument('--min-voxels-per-roi', type=int, default=50)
    p.add_argument('--min-trials-per-bin', type=int, default=3)
    p.add_argument('--min-subjects-per-roi', type=int, default=10)
    p.add_argument('--min-voxels-per-subject', type=int, default=20,
                   help='Drop a subject from an ROI if their own QC left fewer than '
                        'this many valid voxels, preventing dropout from one subject '
                        'zeroing the cross-subject voxel intersection.')
    p.add_argument('--beta-series-dir', type=str, default='beta_series')
    p.add_argument('--verbose', action='store_true', default=False)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    atlas_img, atlas_source = resolve_atlas(cfg, args.atlas, args.atlas_path)
    roi_labels = get_roi_labels_from_atlas_img(atlas_img)

    analysis_tag = args.analysis_tag or f'value_bins_by_delay_{args.atlas}'
    out_dir = Path(cfg.output_root) / f'mvpa_{analysis_tag}' / f'delay_bin_{args.delay_bin}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load subject list
    if args.sub_list:
        sub_list_path = Path(args.sub_list)
    else:
        sub_list_path = (
            Path(cfg.data_root)
            / 'subject_lists'
            / 'mvpa_subject_list.txt'
        )
    sub_ids = [s.strip() for s in sub_list_path.read_text().splitlines() if s.strip()]
    print(f'Loaded {len(sub_ids)} subjects from {sub_list_path}')
    print(f'Atlas: {args.atlas}  source: {atlas_source}')
    print(f'Delay bin: {args.delay_bin}  output: {out_dir}')

    # ------------------------------------------------------------------ #
    # Step 1: Load behavioral data for all subjects (no betas yet).       #
    # Used to compute global tertile edges and then reused for feature    #
    # extraction to avoid hitting the filesystem twice per subject.       #
    # ------------------------------------------------------------------ #
    print('\n--- Step 1: loading behavioral data for all subjects ---')
    beh_results = {}
    failed_beh: list[str] = []
    for sub_id in sub_ids:
        try:
            beh_results[sub_id] = build_subject_behav_bold_df(
                cfg,
                sub_id=sub_id,
                y_col='amount',
                beta_series_subdir=args.beta_series_dir,
                verbose=False,
                strict=False,
            )
        except Exception as exc:
            print(f'  [{sub_id}] behavioral load failed: {exc}')
            failed_beh.append(sub_id)

    good_sub_ids = [s for s in sub_ids if s not in failed_beh]
    print(f'{len(good_sub_ids)}/{len(sub_ids)} subjects passed behavioral loading')

    trial_dfs = [beh_results[s].behav_bold_df for s in good_sub_ids]
    delay_edges, value_edges = compute_global_bin_edges(trial_dfs)
    print(f'Global delay edges (tertiles): {delay_edges}')
    print(f'Global value edges (tertiles): {value_edges}')

    # ------------------------------------------------------------------ #
    # Step 2: Build per-subject binned ROI patterns.                      #
    # ------------------------------------------------------------------ #
    print('\n--- Step 2: building per-subject binned ROI patterns ---')
    all_subject_data = []
    failed_binned: list[str] = []
    for sub_id in good_sub_ids:
        try:
            sd = build_subject_binned_roi_patterns(
                cfg,
                sub_id=sub_id,
                atlas_img=atlas_img,
                delay_edges=delay_edges,
                value_edges=value_edges,
                behav_bold_result=beh_results[sub_id],
                min_trials_per_bin=args.min_trials_per_bin,
                min_voxels_per_roi=args.min_voxels_per_roi,
                beta_series_subdir=args.beta_series_dir,
                verbose=args.verbose,
            )
            all_subject_data.append(sd)
            if args.verbose:
                valid_dbins = sorted(sd.roi_patterns.keys())
                print(f'  [{sub_id}] valid delay bins: {valid_dbins}  '
                      f'excluded: {sd.excluded_delay_bins}')
        except Exception as exc:
            print(f'  [{sub_id}] binned patterns failed: {exc}')
            failed_binned.append(sub_id)

    print(f'{len(all_subject_data)} subjects ready for LOSO decoding')

    # ------------------------------------------------------------------ #
    # Step 3: LOSO multiclass decoding for the specified delay bin.       #
    # ------------------------------------------------------------------ #
    print(f'\n--- Step 3: LOSO decoding for delay_bin={args.delay_bin} ---')
    roi_summary_df, subject_preds_df, confusion_df = decode_value_bins_by_delay(
        all_subject_data,
        delay_bin=args.delay_bin,
        min_subjects_per_roi=args.min_subjects_per_roi,
        min_voxels_per_subject=args.min_voxels_per_subject,
        verbose=True,
    )

    # ------------------------------------------------------------------ #
    # Step 4: Save outputs.                                               #
    # ------------------------------------------------------------------ #
    roi_summary_path = out_dir / 'roi_summary.csv'
    subject_preds_path = out_dir / 'subject_preds.csv'
    confusion_path = out_dir / 'confusion_matrix.csv'

    roi_summary_df.to_csv(roi_summary_path, index=False)
    subject_preds_df.to_csv(subject_preds_path, index=False)
    confusion_df.to_csv(confusion_path, index=False)

    n_subs_with_delay_bin = sum(
        1 for sd in all_subject_data if args.delay_bin in sd.roi_patterns
    )

    meta = {
        'delay_bin': int(args.delay_bin),
        'atlas': args.atlas,
        'atlas_source': str(atlas_source),
        'roi_labels': sorted(list(roi_labels)),
        'analysis_tag': str(analysis_tag),
        'params': {
            'min_voxels_per_roi': int(args.min_voxels_per_roi),
            'min_trials_per_bin': int(args.min_trials_per_bin),
            'min_subjects_per_roi': int(args.min_subjects_per_roi),
            'beta_series_dir': str(args.beta_series_dir),
        },
        'bin_edges': {
            'delay_edges': delay_edges.tolist(),
            'value_edges': value_edges.tolist(),
        },
        'subject_counts': {
            'n_subjects_requested': int(len(sub_ids)),
            'n_subjects_beh_ok': int(len(good_sub_ids)),
            'n_subjects_binned_ok': int(len(all_subject_data)),
            'n_subjects_with_delay_bin': int(n_subs_with_delay_bin),
            'failed_beh': failed_beh,
            'failed_binned': failed_binned,
        },
        'outputs': {
            'roi_summary_csv': str(roi_summary_path),
            'subject_preds_csv': str(subject_preds_path),
            'confusion_matrix_csv': str(confusion_path),
            'n_rois_saved': int(len(roi_summary_df)),
            'n_subject_pred_rows': int(len(subject_preds_df)),
        },
    }
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    (out_dir / '_SUCCESS').write_text('ok\n')

    print(f'\n[done] {len(roi_summary_df)} ROIs saved to {out_dir}')


if __name__ == '__main__':
    main()
