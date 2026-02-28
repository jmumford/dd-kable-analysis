# Master Analysis Registry

# Summary Table

<table>
<tr><th colspan="5"><strong>Preprocessing</strong></th></tr>
<tr><td>ID</td><td>Description</td><td>Status</td><td>Results Files</td><td>Notes</td></tr>
<tr><td>behavioral data exploration</td><td>This is where any exploration of the behavioral data is done.  I started it with RT distributions (within-subject and within-runs across subjects).</td><td>completed</td><td><a href="../analyses/behavioral_data_exploration/response_time_exploration.ipynb">response_time_exploration.ipynb</a></td><td>Let me know if there are other ways I should assess RT or the other measures.</td></tr>
<tr><td>bold data exploration</td><td>Initiated to verify that there did not appear to be nonsteady state scans in the BOLD data.</td><td>complete</td><td><a href="../analyses/bold_data_exploration/check_scanner_warmup_volumes.ipynb">check_scanner_warmup_volumes.ipynb</a></td><td>Using the non_steady_state_outlier columns of the fmriprep confound output files, I checked how many nonstead state scans occurred for each subject/run and found that it wasn't common and typically if a scan was flagged it was only the first TR.   Of the ~420 runs only around 50-60 had any TRs flagged.  Figures summarize findings in notebook.</td></tr>
<tr><td>define inclusion criteria</td><td>Working from inclusion_subjects_120426.csv, the notebook in this directory generates an inclusion (DD_Kable/subject_lists/initial_qa_pass_subjects_runs.csv) and exclusion file (DD_Kable/subject_lists/initial_qa_fail_subjects_runs_reasons.csv). The inclusion criteria are defined in the notebook and these are the subjects and runs that should be included in time series analysis.</td><td>completed</td><td><a href="../analyses/define_inclusion_criteria/define_inclusion_criteria.ipynb">define_inclusion_criteria.ipynb</a></td><td>Omit a run that has >10% missingness, <10% or >90% choseAccept and/or no data in csv.   Omit a subject if they have fewer than 2 good runs out of 4.</td></tr>
<tr><td>group mask</td><td>Generate a group mask and create individual mask PDF summaries for QC.</td><td>completed</td><td><a href="../analyses/group_mask/make_group_mask.ipynb">make_group_mask.ipynb</a>, <a href="../analyses/group_mask/.../DD_Kable/subject_lists/initial_qa_pass_and_mask_pass_subjects_runs.csv">.../DD_Kable/subject_lists/initial_qa_pass_and_mask_pass_subjects_runs.csv</a>, <a href="../analyses/group_mask/.../DD_Kable/derivatives/masks/assess_subject_bold_dropout/group_mask_intersection_30pct.nii.gz">.../DD_Kable/derivatives/masks/assess_subject_bold_dropout/group_mask_intersection_30pct.nii.gz</a>, <a href="../analyses/group_mask/.../DD_Kable/derivatives/masks/assess_subject_bold_dropout/mask_and_mean_QA.pdf">.../DD_Kable/derivatives/masks/assess_subject_bold_dropout/mask_and_mean_QA.pdf</a></td><td>Use the group_mask_intersection_30pct.nii.gz for any group analyses.</td></tr>
<tr><th colspan="5"><strong>Time Series</strong></th></tr>
<tr><td>ID</td><td>Description</td><td>Status</td><td>Results Files</td><td>Notes</td></tr>
<tr><td>beta series analysis</td><td>Set up design matrices and contrasts for beta series analysis, and run the analysis on the included subjects and runs.  Assess the results for outliers.</td><td>completed</td><td><a href="../analyses/beta_series_analysis/figures/design_matrix_assessment.pdf">/figures/design_matrix_assessment.pdf</a>, <a href="../analyses/beta_series_analysis/subject_lists/vif_gt_5.csv">subject_lists/vif_gt_5.csv</a>, <a href="../analyses/beta_series_analysis/beta_series_analysis_code_dev.ipynb">beta_series_analysis_code_dev.ipynb</a>, <a href="../analyses/beta_series_analysis/qa_design_matrices.ipynb">qa_design_matrices.ipynb</a>, <a href="../analyses/beta_series_analysis/qa_analysis_output.ipynb">qa_analysis_output.ipynb</a></td><td>TBD</td></tr>
<tr><td>traditional model analysis</td><td>Set up design matrices and contrasts for the traditional model, and run the analysis on the included subjects and runs. Assess the results for outliers.</td><td>in progress</td><td><a href="../analyses/traditional_model_analysis/model_code_dev.ipynb">model_code_dev.ipynb</a>, <a href="../analyses/traditional_model_analysis/qa_analysis_output.ipynb">qa_analysis_output.ipynb</a></td><td>TBD</td></tr>
<tr><th colspan="5"><strong>Time Series And Group</strong></th></tr>
<tr><td>ID</td><td>Description</td><td>Status</td><td>Results Files</td><td>Notes</td></tr>
<tr><td>mvpa value decoding josh rois</td><td>MVPA value decoding using Josh ROIs, including data prep and aggregated results.</td><td>in progress</td><td><a href="../analyses/mvpa_value_decoding/mvpa_josh_rois/mvpa_josh_rois_prep.ipynb">mvpa_josh_rois_prep.ipynb</a>, <a href="../analyses/mvpa_value_decoding/mvpa_josh_rois/mvpa_josh_rois_aggregate_results.ipynb">mvpa_josh_rois_aggregate_results.ipynb</a></td><td>TBD</td></tr>
<tr><th colspan="5"><strong>Group</strong></th></tr>
<tr><td>ID</td><td>Description</td><td>Status</td><td>Results Files</td><td>Notes</td></tr>
<tr><td>ll ss group model comparison</td><td>Group-level comparisons for LL vs SS effects using summary outputs and QA notebooks.</td><td>in progress</td><td><a href="../analyses/ll_ss_group_model_comparison/accept_minus_reject_individual_run_comparison.ipynb">accept_minus_reject_individual_run_comparison.ipynb</a>, <a href="../analyses/ll_ss_group_model_comparison/accept_reject_group_comparison.ipynb">accept_reject_group_comparison.ipynb</a></td><td>TBD</td></tr>
</table>




# Detailed Reports


## Preprocessing

### behavioral_data_exploration
**Name:** RT distributions and other explorations<br>
**Description:** This is where any exploration of the behavioral data is done.  I started it with RT distributions (within-subject and within-runs across subjects).<br>
**Code Directory:** analyses/behavioral_data_exploration<br>
**Dependencies:** None<br>
**Script Entry:** None<br>
**Notebook Entry:**
- response_time_exploration.ipynb

**Other Files:** None<br>
**Output Directory:** None<br>
**Results Files:** <a href="../analyses/behavioral_data_exploration/response_time_exploration.ipynb">response_time_exploration.ipynb</a><br>
**Hypothesis:** None<br>
**Conclusion:** Nothing unusual found (that I'm aware of)<br>
**Notes:** Let me know if there are other ways I should assess RT or the other measures.<br>
**Status:** completed<br>
**Last Updated:** 2026-02-05<br>
**Authors:** Jeanette Mumford<br>

---

### bold_data_exploration
**Name:** Explore BOLD data for oddities<br>
**Description:** Initiated to verify that there did not appear to be nonsteady state scans in the BOLD data.<br>
**Code Directory:** analyses/bold_data_exploration<br>
**Dependencies:** None<br>
**Script Entry:** None<br>
**Notebook Entry:**
- check_scanner_warmup_volumes.ipynb

**Other Files:** None<br>
**Output Directory:** None<br>
**Results Files:** <a href="../analyses/bold_data_exploration/check_scanner_warmup_volumes.ipynb">check_scanner_warmup_volumes.ipynb</a><br>
**Hypothesis:** None<br>
**Conclusion:** No nonsteady state scans detected in the BOLD data.<br>
**Notes:** Using the non_steady_state_outlier columns of the fmriprep confound output files, I checked how many nonstead state scans occurred for each subject/run and found that it wasn't common and typically if a scan was flagged it was only the first TR.   Of the ~420 runs only around 50-60 had any TRs flagged.  Figures summarize findings in notebook.<br>
**Status:** complete<br>
**Last Updated:** 2026-02-11<br>
**Authors:** Jeanette Mumford<br>

---

### define_inclusion_criteria
**Name:** Generates inclusion/exclusion files for inputs in time series analysis.<br>
**Description:** Working from inclusion_subjects_120426.csv, the notebook in this directory generates an inclusion (DD_Kable/subject_lists/initial_qa_pass_subjects_runs.csv) and exclusion file (DD_Kable/subject_lists/initial_qa_fail_subjects_runs_reasons.csv). The inclusion criteria are defined in the notebook and these are the subjects and runs that should be included in time series analysis.<br>
**Code Directory:** analyses/define_inclusion_criteria<br>
**Dependencies:** None<br>
**Script Entry:**
- define_inclusion_criteria_utils.py

**Notebook Entry:**
- define_inclusion_criteria.ipynb

**Other Files:** None<br>
**Output Directory:** DD_Kable/subject_lists/<br>
**Results Files:** <a href="../analyses/define_inclusion_criteria/define_inclusion_criteria.ipynb">define_inclusion_criteria.ipynb</a><br>
**Hypothesis:** None<br>
**Conclusion:** 125 subjects total (6 subjects with 2 runs, 17 with 3 runs, 102 with 4 runs)<br>
**Notes:** Omit a run that has >10% missingness, <10% or >90% choseAccept and/or no data in csv.   Omit a subject if they have fewer than 2 good runs out of 4.<br>
**Status:** completed<br>
**Last Updated:** 2026-02-05<br>
**Authors:** Jeanette Mumford<br>

---

### group_mask
**Name:** Create group mask and individual mask QC<br>
**Description:** Generate a group mask and create individual mask PDF summaries for QC.<br>
**Code Directory:** analyses/group_mask<br>
**Dependencies:** None<br>
**Script Entry:**
- make_individual_masks_pdf.py
- run_make_individual_masks_pdf.batch

**Notebook Entry:**
- make_group_mask.ipynb

**Other Files:** None<br>
**Output Directory:** .../DD_Kable/subject_lists, .../DD_Kable/derivatives/masks/assess_subject_bold_dropout<br>
**Results Files:** <a href="../analyses/group_mask/make_group_mask.ipynb">make_group_mask.ipynb</a>, <a href="../analyses/group_mask/.../DD_Kable/subject_lists/initial_qa_pass_and_mask_pass_subjects_runs.csv">.../DD_Kable/subject_lists/initial_qa_pass_and_mask_pass_subjects_runs.csv</a>, <a href="../analyses/group_mask/.../DD_Kable/derivatives/masks/assess_subject_bold_dropout/group_mask_intersection_30pct.nii.gz">.../DD_Kable/derivatives/masks/assess_subject_bold_dropout/group_mask_intersection_30pct.nii.gz</a>, <a href="../analyses/group_mask/.../DD_Kable/derivatives/masks/assess_subject_bold_dropout/mask_and_mean_QA.pdf">.../DD_Kable/derivatives/masks/assess_subject_bold_dropout/mask_and_mean_QA.pdf</a><br>
**Hypothesis:** None<br>
**Conclusion:** None<br>
**Notes:** Use the group_mask_intersection_30pct.nii.gz for any group analyses.<br>
**Status:** completed<br>
**Last Updated:** 2026-02-28<br>
**Authors:** Jeanette Mumford<br>

---


## Time Series

### beta_series_analysis
**Name:** Develop and assess beta series analysis pipeline for time series data.<br>
**Description:** Set up design matrices and contrasts for beta series analysis, and run the analysis on the included subjects and runs.  Assess the results for outliers.<br>
**Code Directory:** analyses/beta_series_analysis<br>
**Dependencies:** None<br>
**Script Entry:**
- run_beta_series.py
- submit_beta_series.sh
- qa_output.py

**Notebook Entry:**
- beta_series_analysis_code_dev.ipynb
- qa_design_matrices.ipynb
- qa_analysis_output.ipynb

**Other Files:** None<br>
**Output Directory:** None<br>
**Results Files:** <a href="../analyses/beta_series_analysis/figures/design_matrix_assessment.pdf">/figures/design_matrix_assessment.pdf</a>, <a href="../analyses/beta_series_analysis/subject_lists/vif_gt_5.csv">subject_lists/vif_gt_5.csv</a>, <a href="../analyses/beta_series_analysis/beta_series_analysis_code_dev.ipynb">beta_series_analysis_code_dev.ipynb</a>, <a href="../analyses/beta_series_analysis/qa_design_matrices.ipynb">qa_design_matrices.ipynb</a>, <a href="../analyses/beta_series_analysis/qa_analysis_output.ipynb">qa_analysis_output.ipynb</a><br>
**Hypothesis:** None<br>
**Conclusion:** Beta series model has stable VIFs across trial estimates.<br>
**Notes:** TBD<br>
**Status:** completed<br>
**Last Updated:** 2026-02-28<br>
**Authors:** Jeanette Mumford<br>

---

### traditional_model_analysis
**Name:** Develop and assess traditional model pipeline for time series data.<br>
**Description:** Set up design matrices and contrasts for the traditional model, and run the analysis on the included subjects and runs. Assess the results for outliers.<br>
**Code Directory:** analyses/traditional_model_analysis<br>
**Dependencies:** None<br>
**Script Entry:**
- run_traditional_model.py
- submit_traditional_model.sh

**Notebook Entry:**
- model_code_dev.ipynb
- qa_analysis_output.ipynb

**Other Files:** None<br>
**Output Directory:** None<br>
**Results Files:** <a href="../analyses/traditional_model_analysis/model_code_dev.ipynb">model_code_dev.ipynb</a>, <a href="../analyses/traditional_model_analysis/qa_analysis_output.ipynb">qa_analysis_output.ipynb</a><br>
**Hypothesis:** None<br>
**Conclusion:** None<br>
**Notes:** TBD<br>
**Status:** in progress<br>
**Last Updated:** 2026-02-13<br>
**Authors:** Jeanette Mumford<br>

---


## Time Series And Group

### mvpa_value_decoding_josh_rois
**Name:** MVPA value decoding (Josh ROIs)<br>
**Description:** MVPA value decoding using Josh ROIs, including data prep and aggregated results.<br>
**Code Directory:** analyses/mvpa_value_decoding/mvpa_josh_rois<br>
**Dependencies:** None<br>
**Script Entry:**
- run_josh_rois_subject.py
- submit_run_josh_rois.sh

**Notebook Entry:**
- mvpa_josh_rois_prep.ipynb
- mvpa_josh_rois_aggregate_results.ipynb

**Other Files:** None<br>
**Output Directory:** None<br>
**Results Files:** <a href="../analyses/mvpa_value_decoding/mvpa_josh_rois/mvpa_josh_rois_prep.ipynb">mvpa_josh_rois_prep.ipynb</a>, <a href="../analyses/mvpa_value_decoding/mvpa_josh_rois/mvpa_josh_rois_aggregate_results.ipynb">mvpa_josh_rois_aggregate_results.ipynb</a><br>
**Hypothesis:** None<br>
**Conclusion:** None<br>
**Notes:** TBD<br>
**Status:** in progress<br>
**Last Updated:** 2026-02-28<br>
**Authors:** Jeanette Mumford<br>

---


## Group

### ll_ss_group_model_comparison
**Name:** LL-SS group model comparison<br>
**Description:** Group-level comparisons for LL vs SS effects using summary outputs and QA notebooks.<br>
**Code Directory:** analyses/ll_ss_group_model_comparison<br>
**Dependencies:** None<br>
**Script Entry:** None<br>
**Notebook Entry:**
- accept_minus_reject_individual_run_comparison.ipynb
- accept_reject_group_comparison.ipynb

**Other Files:** None<br>
**Output Directory:** None<br>
**Results Files:** <a href="../analyses/ll_ss_group_model_comparison/accept_minus_reject_individual_run_comparison.ipynb">accept_minus_reject_individual_run_comparison.ipynb</a>, <a href="../analyses/ll_ss_group_model_comparison/accept_reject_group_comparison.ipynb">accept_reject_group_comparison.ipynb</a><br>
**Hypothesis:** None<br>
**Conclusion:** None<br>
**Notes:** TBD<br>
**Status:** in progress<br>
**Last Updated:** 2026-02-13<br>
**Authors:** Jeanette Mumford<br>

---
