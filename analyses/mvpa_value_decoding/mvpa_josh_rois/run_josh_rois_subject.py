#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from dd_kable_analysis.config_loader import load_config
from dd_kable_analysis.mvpa import (
    build_subject_behav_bold_df,
    decode_subject_atlas_rois,
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Run MVPA decoding for Josh ROI label atlas (one subject).'
    )
    p.add_argument('--sub-id', type=str, required=True, help='Subject ID like dmp0330')
    p.add_argument(
        '--y-col', type=str, default='amount', help='Target column (default: amount)'
    )
    p.add_argument('--min-voxels-per-roi', type=int, default=50)
    p.add_argument('--small-thr', type=float, default=1e-4)
    p.add_argument('--max-small-frac', type=float, default=0.05)
    p.add_argument('--require-all-runs', action='store_true', default=True)
    p.add_argument(
        '--no-require-all-runs', dest='require_all_runs', action='store_false'
    )
    p.add_argument('--verbose', action='store_true', default=False)
    return p.parse_args()


def get_roi_labels_from_atlas(atlas_path: Path, drop_label: int = 0) -> set[int]:
    img = nib.load(str(atlas_path))
    labs = np.unique(img.get_fdata().astype(int))
    return {int(l) for l in labs if int(l) != drop_label}


def main():
    args = parse_args()
    cfg = load_config()
    sub_id = args.sub_id

    atlas_path = Path(cfg.masks_dir) / 'josh_orig_rois.nii.gz'
    if not atlas_path.exists():
        raise FileNotFoundError(f'Missing atlas image: {atlas_path}')

    roi_labels = get_roi_labels_from_atlas(atlas_path)

    out_dir = Path(cfg.output_root) / 'mvpa_josh_rois'
    sub_dir = out_dir / f'sub-{sub_id}'
    sub_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f'[{sub_id}] atlas={atlas_path}')
        print(f'[{sub_id}] roi labels={sorted(roi_labels)}')
        print(f'[{sub_id}] output={sub_dir}')

    roi_summary_df, trialwise_df = decode_subject_atlas_rois(
        cfg,
        sub_id=sub_id,
        atlas_img=str(atlas_path),
        y_col=args.y_col,
        min_voxels_per_roi=args.min_voxels_per_roi,
        small_thr=args.small_thr,
        max_small_frac=args.max_small_frac,
        require_all_runs=args.require_all_runs,
        alphas=None,
        verbose=args.verbose,
        return_trialwise=True,
        trialwise_rois=roi_labels,  # all ROIs in atlas
    )

    roi_summary_path = sub_dir / 'roi_summary.csv'
    trialwise_path = sub_dir / 'trialwise_preds.csv'
    roi_summary_df.to_csv(roi_summary_path, index=False)
    trialwise_df.to_csv(trialwise_path, index=False)

    # Save basic QC/meta
    out = build_subject_behav_bold_df(cfg, sub_id=sub_id, verbose=False, strict=True)
    meta = {
        'sub_id': str(sub_id),
        'atlas_path': str(atlas_path),
        'roi_labels': sorted(list(roi_labels)),
        'y_col': args.y_col,
        'params': {
            'min_voxels_per_roi': int(args.min_voxels_per_roi),
            'small_thr': float(args.small_thr),
            'max_small_frac': float(args.max_small_frac),
            'require_all_runs': bool(args.require_all_runs),
        },
        'input_qc': {
            'good_runs': out.good_runs,
            'trials_kept_by_run': out.trials_kept_by_run,
            'runs_passing_trial_threshold': out.runs_passing_trial_threshold,
            'n_trials_before_vif': out.n_trials_before_vif,
            'n_trials_after_vif_and_missing': out.n_trials_after_vif_and_missing,
            'n_missing_betas': out.n_missing_betas,
            'n_high_vif_omitted': out.n_high_vif_omitted,
        },
        'outputs': {
            'roi_summary_csv': str(roi_summary_path),
            'trialwise_preds_csv': str(trialwise_path),
            'n_rois_saved': int(len(roi_summary_df)),
            'n_trialwise_rows': int(len(trialwise_df)),
        },
    }
    (sub_dir / 'meta.json').write_text(json.dumps(meta, indent=2))
    (sub_dir / '_SUCCESS').write_text('ok\n')

    if args.verbose:
        print(f'[done] wrote {roi_summary_path}')
        print(f'[done] wrote {trialwise_path}')


if __name__ == '__main__':
    main()
