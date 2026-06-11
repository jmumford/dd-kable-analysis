from __future__ import annotations

from pathlib import Path
from typing import Any, Set, Tuple

import nibabel as nib
import numpy as np
from nilearn import datasets


def get_roi_labels_from_atlas_img(atlas_img: Any, drop_label: int = 0) -> Set[int]:
    img = (
        nib.load(str(atlas_img))
        if not hasattr(atlas_img, 'get_fdata')
        else atlas_img
    )
    labs = np.unique(img.get_fdata().astype(int))
    return {int(label) for label in labs if int(label) != drop_label}


def resolve_atlas(cfg: Any, atlas: str, atlas_path: str | None) -> Tuple[Any, str]:
    if atlas_path is not None:
        ap = Path(atlas_path)
        if not ap.exists():
            raise FileNotFoundError(f'Missing atlas_path: {ap}')
        return str(ap), str(ap)

    if atlas == 'josh_orig':
        ap = Path(cfg.masks_dir) / 'josh_orig_rois.nii.gz'
        if not ap.exists():
            raise FileNotFoundError(f'Missing atlas image: {ap}')
        return str(ap), str(ap)

    if atlas in ('schaefer200', 'schaefer400'):
        n_rois = 200 if atlas == 'schaefer200' else 400
        sch = datasets.fetch_atlas_schaefer_2018(
            n_rois=n_rois, yeo_networks=7, resolution_mm=2
        )
        return (
            sch.maps,
            f'nilearn.fetch_atlas_schaefer_2018(n_rois={n_rois}, networks=7, res=2mm)',
        )

    if atlas == 'harvard_oxford_subcort':
        ho = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
        return ho.maps, "nilearn.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')"

    if atlas == 'pauli_rois':
        pauli_dir = (
            Path(cfg.masks_dir) / 'pauli_subcort_rois_neurovault_collection_3145'
        )
        ap = pauli_dir / 'pauli_roi_atlas_thr0.50_to-groupmask.nii.gz'
        if not ap.exists():
            raise FileNotFoundError(f'Missing Pauli atlas: {ap}')
        return str(ap), str(ap)

    raise ValueError(f'Unknown atlas: {atlas}')
