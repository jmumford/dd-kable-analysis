from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from nilearn import image


def _compute_z_index(img: nib.spatialimages.SpatialImage, z_world: float) -> int:
    affine = img.affine
    ijk = np.linalg.inv(affine) @ np.array([0, 0, z_world, 1])
    z_idx = int(np.round(ijk[2]))
    return int(np.clip(z_idx, 0, img.shape[2] - 1))


def _load_slices_and_range(
    trial_files: Iterable[Path],
    z_idx: int,
    vmin_pct: float,
    vmax_pct: float,
    fwhm: Optional[float] = None,  # NEW
) -> Tuple[list[np.ndarray], float, float]:
    slices: list[np.ndarray] = []
    all_vals: list[np.ndarray] = []

    for f in trial_files:
        if fwhm is None or fwhm == 0:
            img = nib.load(str(f))
        else:
            img = image.smooth_img(str(f), fwhm=fwhm)

        data = img.get_fdata()
        sl = data[:, :, z_idx]
        slices.append(sl)
        all_vals.append(sl.ravel())

    all_vals_concat = np.concatenate(all_vals)
    vmin = float(np.nanpercentile(all_vals_concat, vmin_pct))
    vmax = float(np.nanpercentile(all_vals_concat, vmax_pct))
    if vmin == vmax:
        vmax = vmin + 1.0

    return slices, vmin, vmax


def _make_grid_qa_figure(
    trial_files: list[Path],
    sub_id: str,
    run: int,
    z_world: float,
    grid_shape: Tuple[int, int],
    figsize: Tuple[float, float],
    vmin_pct: float,
    vmax_pct: float,
    fwhm: Optional[float] = None,
) -> plt.Figure:
    img0 = nib.load(str(trial_files[0]))
    z_idx = _compute_z_index(img0, z_world)

    slices, vmin, vmax = _load_slices_and_range(
        trial_files=trial_files,
        z_idx=z_idx,
        vmin_pct=vmin_pct,
        vmax_pct=vmax_pct,
        fwhm=fwhm,
    )

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    n_rows, n_cols = grid_shape
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        left=0.05,
        right=0.88,
        top=0.92,
        bottom=0.05,
        wspace=0.02,
        hspace=0.15,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(n_rows) for j in range(n_cols)]

    for ax, sl, f in zip(axes, slices, trial_files):
        im = ax.imshow(np.rot90(sl), cmap='coolwarm', norm=norm)
        ax.set_title(Path(f).name.split('_contrast-')[1].split('_')[0], fontsize=8)
        ax.axis('off')

    for ax in axes[len(slices) :]:
        ax.axis('off')

    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Effect size', rotation=90)
    title = f'sub={sub_id}, run={run} (z={z_world})'
    if fwhm and fwhm != 0:
        title += f', smoothed={fwhm}mm'
    _ = fig.suptitle(title, y=0.98)

    return fig


def build_beta_map_qa_pdf(
    cfg,
    good_subs: Optional[pd.DataFrame] = None,
    output_pdf: Optional[Path] = None,
    max_runs: Optional[int] = None,
    z_world: float = 22,
    grid_shape: Tuple[int, int] = (5, 6),
    figsize: Tuple[float, float] = (12, 10),
    vmin_pct: float = 1,
    vmax_pct: float = 99,
    fwhm: Optional[float] = None,
    gc_every: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Create a PDF with one page per subject/run grid QA figure.

    Notes:
    - If there are more trial files than grid slots, extra files are ignored.
    - Figures are closed after saving to keep memory usage low.
    - Set max_runs to an integer to test on a subset.
    """
    if good_subs is None:
        good_subs = pd.read_csv(cfg.subject_lists / 'initial_qa_pass_subjects_runs.csv')

    if max_runs is not None:
        good_subs = good_subs.head(int(max_runs))

    if output_pdf is None:
        figures_dir = Path(cfg.output_root) / 'beta_series' / 'figures'
        output_pdf = figures_dir / 'beta_map_qa.pdf'

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    was_interactive = plt.isinteractive()
    plt.ioff()

    n_written = 0
    n_missing = 0
    n_total = len(good_subs)

    with PdfPages(output_pdf) as pdf:
        for idx, row in good_subs.iterrows():
            sub_id = row['sub_id']
            run = int(row['run'])

            output_dir = (
                Path(cfg.output_root)
                / 'beta_series'
                / 'first_level'
                / f'sub-{sub_id}'
                / 'contrast_estimates'
            )
            trial_files = sorted(
                output_dir.glob(
                    f'sub-{sub_id}_ses-*_task-*_run-{run}_contrast-trial*_output-effectsize.nii.gz'
                )
            )

            if not trial_files:
                n_missing += 1
                if verbose:
                    print(f'No trial beta maps found for sub={sub_id}, run={run}')
                continue

            fig = _make_grid_qa_figure(
                trial_files=trial_files,
                sub_id=sub_id,
                run=run,
                z_world=z_world,
                grid_shape=grid_shape,
                figsize=figsize,
                vmin_pct=vmin_pct,
                vmax_pct=vmax_pct,
                fwhm=fwhm,
            )
            pdf.savefig(fig)
            plt.close(fig)
            n_written += 1

            if gc_every and (n_written % gc_every == 0):
                gc.collect()

            if verbose and ((idx + 1) % 25 == 0):
                print(f'Processed {idx + 1}/{n_total} runs')

    if was_interactive:
        plt.ion()

    return {
        'output_pdf': str(output_pdf),
        'n_total_runs': n_total,
        'n_written': n_written,
        'n_missing': n_missing,
    }


def _load_mask(mask: Union[Path, str, nib.spatialimages.SpatialImage]) -> np.ndarray:
    if isinstance(mask, (str, Path)):
        mask_img = nib.load(str(mask))
    else:
        mask_img = mask

    mask_data = mask_img.get_fdata()
    return mask_data > 0


def _extract_contrast_name(path: Path) -> str:
    name = path.name
    if '_contrast-' not in name:
        return name
    return name.split('_contrast-')[1].split('_output-')[0]


def _compute_metrics(values: np.ndarray, k: float) -> dict:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {'rms': np.nan, 'mad': np.nan, 'tail_fraction': np.nan, 'median': np.nan}

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    rms = float(np.sqrt(np.mean(values**2)))
    eps = 1e-6
    tail_fraction = float(np.mean(np.abs(values - median) > max(k * mad, eps)))

    return {'rms': rms, 'mad': mad, 'tail_fraction': tail_fraction, 'median': median}


def build_beta_map_metric_summary(
    cfg,
    mask: Union[Path, str, nib.spatialimages.SpatialImage],
    good_subs: Optional[pd.DataFrame] = None,
    max_runs: Optional[int] = None,
    output_csv: Optional[Path] = None,
    k: float = 6,
    gc_every: int = 25,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute per-map metrics across all contrasts for each subject/run.

    Metrics:
    - RMS amplitude
    - MAD (median absolute deviation)
    - Tail fraction: mean(|x - median(x)| > k * MAD)
    """
    if good_subs is None:
        good_subs = pd.read_csv(cfg.subject_lists / 'initial_qa_pass_subjects_runs.csv')

    if max_runs is not None:
        good_subs = good_subs.head(int(max_runs))

    mask_bool = _load_mask(mask)

    rows = []
    n_total = len(good_subs)

    for idx, row in good_subs.iterrows():
        sub_id = row['sub_id']
        run = int(row['run'])

        output_dir = (
            Path(cfg.output_root)
            / 'beta_series'
            / 'first_level'
            / f'sub-{sub_id}'
            / 'contrast_estimates'
        )
        contrast_files = sorted(
            output_dir.glob(
                f'sub-{sub_id}_ses-*_task-*_run-{run}_contrast-*_output-effectsize.nii.gz'
            )
        )

        if not contrast_files:
            if verbose:
                print(f'No contrast maps found for sub={sub_id}, run={run}')
            continue

        for f in contrast_files:
            img = nib.load(str(f))
            data = img.get_fdata()

            if data.shape != mask_bool.shape:
                raise ValueError(
                    f'Mask shape {mask_bool.shape} does not match data shape {data.shape} for {f}'
                )

            vals = data[mask_bool]
            metrics = _compute_metrics(vals, k=k)

            rows.append(
                {
                    'sub_id': sub_id,
                    'run': run,
                    'contrast': _extract_contrast_name(f),
                    'path': str(f),
                    **metrics,
                }
            )

        if gc_every and ((idx + 1) % gc_every == 0):
            gc.collect()

        if verbose and ((idx + 1) % 25 == 0):
            print(f'Processed {idx + 1}/{n_total} runs')

    df = pd.DataFrame(rows)

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)

    return df
