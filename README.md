## Delay Discounting (aka ITC) Data Analysis
This respository contains code for the analysis of the Kable Delay Discounting data.  Analyses were carried out by Jeanette Mumford under the direction of Josh Buckholtz.


## Repository Structure

> **Note:** This code is configured for a Sherlock-based directory structure. Paths and file locations are specific to that environment and may need adjustment to run elsewhere.

**Highlights:**

- **analyses/** contains all step-specific scripts and notebooks. Some examples:  
  - `behavioral_data_exploration/` – Scripts for basic exploration of behavioral data 
  - `define_inclusion_criteria/` – How subjects were selected for time series analysis

- **analysis_registry/** contains the Master Analysis Registry, designed for internal navigation and quick reference — it makes it easy to locate results, notebooks, and processed outputs without manually searching directories.  
  - `README.md`: [Master Analysis Registry](analysis_registry/README.md)  
    - Quickly summarizes results  
    - Helps navigate the analysis code  
    - Points to output files  
  - Each analysis step is documented in a YAML file (`analysis_registry/analyses/*.yaml`)  
  - `build_master_registry.py`: Assembles individual `.yaml` files into the main `.yaml` and `README.md`

- **src/** contains utility code and core processing functions used across analyses.



## Running the Code

- **Python 3.12+** is recommended.  
- All scripts assume paths and files are structured according to the Sherlock environment.  
- Most analyses are run via batch scripts on Sherlock; notebooks are primarily for exploration, visualization, and QA.

---

## License & Citation

- MIT License  
- If this code is used for research purposes, please cite the corresponding study or repository.