#!/usr/bin/env python3
"""
Traditional Model Analysis Script

This script runs a traditional first-level model for a single subject and run.
It fetches BOLD data, creates the traditional design matrix (ll/ss/RT),
computes contrasts, and saves effect size maps with BIDS-style naming conventions.

Usage:
    python run_traditional_model.py <subject_id> <run_number>

Example:
    python run_traditional_model.py dmp0005 1
"""

import argparse
import sys
from pathlib import Path

from nilearn.glm.first_level import FirstLevelModel

from dd_kable_analysis.config_loader import load_config
from dd_kable_analysis.data_io import resolve_file
from dd_kable_analysis.tseries_model.contrast_model import make_traditional_contrast_set
from dd_kable_analysis.tseries_model.design_matrix import make_design_matrix_traditional


def run_traditional_model(sub_id: str, run: int) -> None:
    """
    Run traditional model analysis for a single subject and run.

    Parameters
    ----------
    sub_id : str
        Subject identifier (e.g., 'dmp0005')
    run : int
        Run number (e.g., 1, 2)
    """
    # Load configuration
    print('Loading configuration...')
    cfg = load_config()

    # Convert run to string for file resolution
    run_str = str(run)
    ses = 'scan1'

    print(f'\n{"=" * 70}')
    print('Running Traditional Model Analysis')
    print(f'Subject: {sub_id}')
    print(f'Run: {run}')
    print(f'{"=" * 70}\n')

    # Step 1: Fetch BOLD data
    print('Step 1/5: Fetching BOLD data...')
    try:
        bold_file = resolve_file(cfg, sub_id, ses, run_str, 'bold')
        print(f'  ✓ BOLD file: {bold_file.name}')
    except Exception as e:
        print(f'  ✗ Error fetching BOLD data: {e}')
        sys.exit(1)

    # Step 2: Create design matrix
    print('\nStep 2/5: Creating design matrix...')
    try:
        behav_data, events_data, desmat = make_design_matrix_traditional(
            cfg, sub_id, run_str
        )
        print(f'  ✓ Design matrix shape: {desmat.shape}')
    except Exception as e:
        print(f'  ✗ Error creating design matrix: {e}')
        sys.exit(1)

    # Step 3: Create contrast dictionary
    print('\nStep 3/5: Creating contrast dictionary...')
    try:
        contrasts = make_traditional_contrast_set(desmat)
        print(f'  ✓ Number of contrasts: {len(contrasts)}')
    except Exception as e:
        print(f'  ✗ Error creating contrasts: {e}')
        sys.exit(1)

    # Step 4: Fit GLM model
    print('\nStep 4/5: Fitting FirstLevelModel...')
    try:
        smoothing_fwhm = 0

        fmri_glm = FirstLevelModel(smoothing_fwhm=smoothing_fwhm, verbose=0)
        fmri_glm = fmri_glm.fit(bold_file, design_matrices=desmat)
        print('  ✓ Model fitting complete!')
    except Exception as e:
        print(f'  ✗ Error fitting model: {e}')
        sys.exit(1)

    # Step 5: Compute and save contrasts
    print('\nStep 5/5: Computing and saving contrasts...')

    # Create output directory
    output_dir = (
        Path(cfg.output_root)
        / 'traditional_model'
        / 'first_level'
        / f'sub-{sub_id}'
        / 'contrast_estimates'
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'  Output directory: {output_dir}')

    # Compute and save each contrast
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

            output_path = output_dir / filename
            contrast_map.to_filename(output_path)
            saved_files.append(filename)

        except Exception as e:
            print(f"  ✗ Error computing contrast '{contrast_name}': {e}")
            continue

    print(f'\n  ✓ Successfully saved {len(saved_files)} contrast maps')
    print(f'\n{"=" * 70}')
    print(f'Analysis complete for sub-{sub_id}, run {run}')
    print(f'{"=" * 70}\n')


def main():
    """Parse command-line arguments and run analysis."""
    parser = argparse.ArgumentParser(
        description='Run traditional model analysis for a single subject and run',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_traditional_model.py dmp0005 1
  python run_traditional_model.py dmp0010 2
        """,
    )
    parser.add_argument(
        'subject_id', type=str, help='Subject identifier (e.g., dmp0005)'
    )
    parser.add_argument('run', type=int, help='Run number (e.g., 1, 2, 3, etc.)')

    args = parser.parse_args()

    run_traditional_model(args.subject_id, args.run)


if __name__ == '__main__':
    main()
