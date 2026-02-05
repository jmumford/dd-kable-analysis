# Master Analysis Registry

# Summary Table

<table>
<tr><th colspan="5"><strong>Preprocessing</strong></th></tr>
<tr><td>ID</td><td>Description</td><td>Status</td><td>Results Files</td><td>Notes</td></tr>
<tr><td>behavioral data exploration</td><td>This is where any exploration of the behavioral data is done.  I started it with RT distributions (within-subject and within-runs across subjects).</td><td>completed</td><td><a href="../analyses/behavioral_data_exploration/response_time_exploration.ipynb">response_time_exploration.ipynb</a></td><td>Let me know if there are other ways I should assess RT or the other measures.</td></tr>
<tr><td>define inclusion criteria</td><td>Working from inclusion_subjects_120426.csv, the notebook in this directory generates an inclusion (DD_Kable/subject_lists/initial_qa_pass_subjects_runs.csv) and exclusion file (DD_Kable/subject_lists/initial_qa_fail_subjects_runs_reasons.csv). The inclusion criteria are defined in the notebook and these are the subjects and runs that should be included in time series analysis.</td><td>completed</td><td><a href="../analyses/define_inclusion_criteria/define_inclusion_criteria.ipynb">define_inclusion_criteria.ipynb</a></td><td>Omit a run that has >10% missingness, <10% or >90% choseAccept and/or no data in csv.   Omit a subject if they have fewer than 2 good runs out of 4.</td></tr>
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

### define_inclusion_criteria
**Name:** Generates inclusion/exclusion files for inputs in time series analysis.<br>
**Description:** Working from inclusion_subjects_120426.csv, the notebook in this directory generates an inclusion (DD_Kable/subject_lists/initial_qa_pass_subjects_runs.csv) and exclusion file (DD_Kable/subject_lists/initial_qa_fail_subjects_runs_reasons.csv). The inclusion criteria are defined in the notebook and these are the subjects and runs that should be included in time series analysis.<br>
**Code Directory:** analyses/define_inclusion_criteria<br>
**Dependencies:** None<br>
**Script Entry:** None<br>
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
