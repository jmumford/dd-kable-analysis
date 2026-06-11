#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dd_kable_analysis.config_loader import load_config
from dd_kable_analysis.mvpa import (
    build_subject_behav_bold_df,
    decode_subject_atlas_rois,
)
from dd_kable_analysis.mvpa.atlas import get_roi_labels_from_atlas_img, resolve_atlas


def parse_args():
    p = argparse.ArgumentParser(
        description='Run ridge-regression MVPA decoding for an atlas ROI set (one subject).'
    )
    p.add_argument('--sub-id', type=str, required=True)
    p.add_argument('--y-col', type=str, default='SV_LL')
    p.add_argument(
        '--atlas',
        type=str,
        required=True,
        choices=['josh_orig', 'schaefer200', 'schaefer400', 'harvard_oxford_subcort', 'pauli_rois'],
    )
    p.add_argument('--atlas-path', type=str, default=None)
    p.add_argument('--analysis-tag', type=str, default=None,
                   help='Output subdirectory tag; default = <atlas>_<y-col>.')
    p.add_argument('--min-voxels-per-roi', type=int, default=50)
    p.add_argument('--small-thr', type=float, default=1e-4)
    p.add_argument('--max-small-frac', type=float, default=0.05)
    p.add_argument('--require-all-runs', action='store_true', default=True)
    p.add_argument('--no-require-all-runs', dest='require_all_runs', action='store_false')
    p.add_argument('--beta-series-dir', type=str, default='beta_series',
                   help='Subdirectory under output_root containing beta series images.')
    p.add_argument('--verbose', action='store_true', default=False)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    sub_id = args.sub_id

    atlas_img, atlas_source = resolve_atlas(cfg, args.atlas, args.atlas_path)
    roi_labels = get_roi_labels_from_atlas_img(atlas_img)

    analysis_tag = args.analysis_tag or f'{args.atlas}_{args.y_col}'
    out_dir = Path(cfg.output_root) / f'mvpa_{analysis_tag}'
    sub_dir = out_dir / f'sub-{sub_id}'
    sub_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print(f'[{sub_id}] atlas={args.atlas} source={atlas_source}')
        print(f'[{sub_id}] y_col={args.y_col} n_rois={len(roi_labels)} output={sub_dir}')

    roi_summary_df, trialwise_df = decode_subject_atlas_rois(
        cfg,
        sub_id=sub_id,
        atlas_img=atlas_img,
        y_col=args.y_col,
        min_voxels_per_roi=args.min_voxels_per_roi,
        small_thr=args.small_thr,
        max_small_frac=args.max_small_frac,
        require_all_runs=args.require_all_runs,
        alphas=None,
        verbose=args.verbose,
        return_trialwise=True,
        trialwise_rois=roi_labels,
        beta_series_subdir=args.beta_series_dir,
    )

    roi_summary_path = sub_dir / 'roi_summary.csv'
    trialwise_path = sub_dir / 'trialwise_preds.csv'
    roi_summary_df.to_csv(roi_summary_path, index=False)
    trialwise_df.to_csv(trialwise_path, index=False)

    qc = build_subject_behav_bold_df(cfg, sub_id=sub_id, y_col=args.y_col,
                                     beta_series_subdir=args.beta_series_dir,
                                     verbose=False, strict=True)
    meta = {
        'sub_id': str(sub_id),
        'atlas': args.atlas,
        'atlas_source': atlas_source,
        'atlas_img': str(atlas_img),
        'roi_labels': sorted(list(roi_labels)),
        'y_col': args.y_col,
        'analysis_tag': analysis_tag,
        'params': {
            'min_voxels_per_roi': int(args.min_voxels_per_roi),
            'small_thr': float(args.small_thr),
            'max_small_frac': float(args.max_small_frac),
            'require_all_runs': bool(args.require_all_runs),
        },
        'input_qc': {
            'good_runs': qc.good_runs,
            'trials_kept_by_run': qc.trials_kept_by_run,
            'runs_passing_trial_threshold': qc.runs_passing_trial_threshold,
            'n_trials_before_vif': qc.n_trials_before_vif,
            'n_trials_after_vif_and_missing': qc.n_trials_after_vif_and_missing,
            'n_missing_betas': qc.n_missing_betas,
            'n_high_vif_omitted': qc.n_high_vif_omitted,
            'n_missing_sv_rows': qc.n_missing_sv_rows,
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
