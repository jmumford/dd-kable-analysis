import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from nilearn import image
from nilearn.image import new_img_like
from nilearn.maskers import NiftiMasker
from nilearn.plotting import plot_epi, plot_roi, plot_stat_map

from dd_kable_analysis.config_loader import load_config
from dd_kable_analysis.data_io import resolve_file

# Create a colormap with a single bright yellow
yellow_cmap = ListedColormap(['yellow'])


# --------------------------
# Inputs
# --------------------------
cfg = load_config()
good_subs_csv = cfg.subject_lists / 'initial_qa_pass_subjects_runs.csv'
good_subs = pd.read_csv(good_subs_csv)
output_pdf = cfg.masks_dir / 'assess_subject_bold_dropout/mask_and_mean_QA.pdf'

mask_output_dir = cfg.masks_dir / 'assess_subject_bold_dropout/nifti_masker_masks'
os.makedirs(mask_output_dir, exist_ok=True)

# --------------------------
# Create output PDF
# --------------------------
with PdfPages(output_pdf) as pdf:
    for row in good_subs.itertuples(index=False):
        sub_id = row.sub_id
        run = row.run
        bold = resolve_file(cfg, sub_id, 'scan1', run, 'bold')
        print(f'Processing: {bold}')

        # --------------------------
        # Load 4D → compute mean across time
        # --------------------------
        img_4d = image.load_img(bold)
        mean_img = image.mean_img(img_4d)

        # --------------------------
        # Fit masker on the mean image
        # (FAST + equivalent to masking whole 4D)
        # --------------------------
        masker = NiftiMasker(
            smoothing_fwhm=None, standardize=False, mask_strategy='epi'
        )
        masker.fit(mean_img)
        mask_img = masker.mask_img_

        mask_img.to_filename(f'{mask_output_dir}/sub-{sub_id}_run-{run}.nii.gz')

        # --------------------------
        # Prepare figure
        # --------------------------
        z_slices = np.arange(-50, 51, 10)
        fig, axes = plt.subplots(3, 1, figsize=(25, 10))

        fig.suptitle(os.path.basename(str(bold)), fontsize=14)

        # --------------------------
        # Top panel → mean image
        # --------------------------
        plot_epi(
            mean_img,
            display_mode='z',
            cut_coords=z_slices,
            colorbar=False,
            axes=axes[0],
            title='Mean BOLD (time-averaged)',
        )

        # --------------------------
        # middle panel → mask
        # --------------------------
        plot_roi(
            mask_img,
            display_mode='z',
            cut_coords=z_slices,
            colorbar=False,
            cmap=yellow_cmap,
            axes=axes[1],
            title='Mask from NiftiMasker (on MNI)',
            black_bg=True,
        )

        # bg_data = mean_img.get_fdata()

        # # Linear stretch with amplification
        # bg_data_scaled = (bg_data - bg_data.min()) / (bg_data.max() - bg_data.min())
        # bg_data_scaled = bg_data_scaled**0.5  # gamma >1 brightens midrange values
        # bg_data_scaled = np.clip(bg_data_scaled, 0, 1)

        # mean_img_scaled = new_img_like(mean_img, bg_data_scaled)
        # --------------------------
        # bottom panel → mask on mean bold
        # --------------------------
        plot_roi(
            mask_img,
            bg_img=mean_img,
            alpha=0.8,
            cmap=yellow_cmap,
            display_mode='z',
            cut_coords=z_slices,
            colorbar=False,
            title='Mask from NiftiMasker (on Mean BOLD)',
            axes=axes[2],
        )

        pdf.savefig(fig)
        plt.close(fig)

print(f'\nSaved PDF: {output_pdf}')
