#!/usr/bin/env python3
"""
Beta Series Analysis — no RT regressor.

Identical to run_beta_series.py except RT is not included as a separate
regressor in the design matrix. Use this variant when decoding variables
that may correlate with RT (e.g., subjective value).

Output goes to: output_root/beta_series_no_rt/first_level/sub-<id>/contrast_estimates/
"""

import argparse
import sys
from pathlib import Path

from nilearn.glm.first_level import FirstLevelModel

from dd_kable_analysis.config_loader import load_config
from dd_kable_analysis.data_io import resolve_file
from dd_kable_analysis.tseries_model.contrast_model import (
    make_beta_series_constrast_set,
)
from dd_kable_analysis.tseries_model.design_matrix import make_design_matrix_no_rt


def run_beta_series_analysis_no_rt(sub_id: str, run: int) -> None:
    print('Loading configuration...')
    cfg = load_config()

    run_str = str(run)
    ses = 'scan1'

    print(f'\n{"=" * 70}')
    print('Running Beta Series Analysis (no RT regressor)')
    print(f'Subject: {sub_id}')
    print(f'Run: {run}')
    print(f'{"=" * 70}\n')

    print('Step 1/5: Fetching BOLD data...')
    try:
        bold_file = resolve_file(cfg, sub_id, ses, run_str, 'bold')
        print(f'  ✓ BOLD file: {bold_file.name}')
    except Exception as e:
        print(f'  ✗ Error fetching BOLD data: {e}')
        sys.exit(1)

    print('\nStep 2/5: Creating design matrix (no RT)...')
    try:
        behav_data, events_data, desmat = make_design_matrix_no_rt(cfg, sub_id, run_str)
        print(f'  ✓ Design matrix shape: {desmat.shape}')
        print(
            f'  ✓ Number of trials: {len([col for col in desmat.columns if "trial" in col])}'
        )
    except Exception as e:
        print(f'  ✗ Error creating design matrix: {e}')
        sys.exit(1)

    print('\nStep 3/5: Creating contrast dictionary...')
    try:
        contrasts = make_beta_series_constrast_set(desmat, behav_data)
        print(f'  ✓ Number of contrasts: {len(contrasts)}')
    except Exception as e:
        print(f'  ✗ Error creating contrasts: {e}')
        sys.exit(1)

    print('\nStep 4/5: Fitting FirstLevelModel...')
    try:
        fmri_glm = FirstLevelModel(smoothing_fwhm=0, verbose=0)
        fmri_glm = fmri_glm.fit(bold_file, design_matrices=desmat)
        print('  ✓ Model fitting complete!')
    except Exception as e:
        print(f'  ✗ Error fitting model: {e}')
        sys.exit(1)

    print('\nStep 5/5: Computing and saving contrasts...')
    output_dir = (
        Path(cfg.output_root)
        / 'beta_series_no_rt'
        / 'first_level'
        / f'sub-{sub_id}'
        / 'contrast_estimates'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'  Output directory: {output_dir}')

    saved_files = []
    for contrast_name, contrast_def in contrasts.items():
        try:
            contrast_map = fmri_glm.compute_contrast(
                contrast_def, output_type='effect_size'
            )
            filename = (
                f'sub-{sub_id}_'
                f'ses-{ses}_'
                f'task-{cfg.task_name}_'
                f'run-{run}_'
                f'contrast-{contrast_name}_'
                f'output-effectsize.nii.gz'
            )
            contrast_map.to_filename(output_dir / filename)
            saved_files.append(filename)
        except Exception as e:
            print(f"  ✗ Error computing contrast '{contrast_name}': {e}")
            continue

    print(f'\n  ✓ Successfully saved {len(saved_files)} contrast maps')
    print(f'\n{"=" * 70}')
    print(f'Analysis complete for sub-{sub_id}, run {run}')
    print(f'{"=" * 70}\n')


def main():
    parser = argparse.ArgumentParser(
        description='Run beta series analysis (no RT regressor) for a single subject and run'
    )
    parser.add_argument('subject_id', type=str)
    parser.add_argument('run', type=int)
    args = parser.parse_args()
    run_beta_series_analysis_no_rt(args.subject_id, args.run)


if __name__ == '__main__':
    main()
