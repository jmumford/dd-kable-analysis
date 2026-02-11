"""
Minimal Config Loader for ddkable
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Pattern, Union

import yaml


class ConfigError(Exception):
    pass


def _to_path(v: Union[str, Path]) -> Path:
    """Convert string-like path values to Path objects."""
    return v if isinstance(v, Path) else Path(v)


# -------------------------
#  CONFIG DATACLASS
# -------------------------
@dataclass
class ConfoundsConfig:
    patterns: list[str]
    compiled_patterns: list[Pattern[str]] = field(init=False)

    def __post_init__(self) -> None:
        if not self.patterns:
            raise ConfigError('ConfoundsConfig.patterns cannot be empty')
        self.compiled_patterns = [re.compile(p) for p in self.patterns]


@dataclass
class Config:
    # Required fields from YAML
    data_root: Path
    fmriprep_dir: Path
    bids_dir: Path
    subject_lists: Path
    output_root: Path
    task_name: str
    bold_func_glob: str
    bold_data_suffix: str
    mask_data_suffix: str
    behav_func_glob: str
    behav_data_suffix: str
    masks_dir: Path

    confounds: ConfoundsConfig

    tr: float = 3
    smoothing_fwhm: float = 6

    # Derived fields
    bold_file_glob: str = field(init=False)
    bold_mask_file_glob: str = field(init=False)
    behav_file_glob: str = field(init=False)
    confounds_file_glob: str = field(init=False)

    def __post_init__(self):
        # --- Convert path-like fields to Path objects ---
        path_fields = [
            'data_root',
            'fmriprep_dir',
            'bids_dir',
            'subject_lists',
            'output_root',
            'masks_dir',
        ]
        for f in path_fields:
            setattr(self, f, _to_path(getattr(self, f)))

        # --- Normalize globs ---
        if not self.bold_func_glob.endswith('/'):
            self.bold_func_glob += '/'
        if not self.behav_func_glob.endswith('/'):
            self.behav_func_glob += '/'

        # --- Derived file globs ---
        self.bold_file_glob = (
            f'{self.bold_func_glob}'
            f'sub-{{subject}}*_ses-{{ses}}_task-{self.task_name}_run-{{run}}_'
            f'{self.bold_data_suffix}'
        )
        self.bold_mask_file_glob = (
            f'{self.bold_func_glob}'
            f'sub-{{subject}}*_ses-{{ses}}_task-{self.task_name}_run-{{run}}_'
            f'{self.mask_data_suffix}'
        )
        self.confounds_file_glob = (
            f'{self.bold_func_glob}'
            f'sub-{{subject}}*_ses-{{ses}}_task-{self.task_name}_run-{{run}}_'
            f'desc-confounds_timeseries.tsv'
        )
        self.behav_file_glob = (
            f'{self.behav_func_glob}'
            f'sub-{{subject}}_ses-{{ses}}_task-{self.task_name}_run-{{run}}_events.tsv'
        )


# -------------------------
#  LOADER
# -------------------------


def load_config(config_file: Union[str, Path] = None) -> Config:
    """
    Load YAML from either:
      - the provided config_file, OR
      - the package-local default configs/config.yaml
    """

    if config_file is None:
        config_file = Path(__file__).parent / 'configs' / 'config.yaml'

    config_path = _to_path(config_file)

    if not config_path.exists():
        raise ConfigError(f'Config file not found: {config_file}')

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    # Convert known path fields
    path_keys = {
        'data_root',
        'fmriprep_dir',
        'bids_dir',
        'subject_lists',
        'output_root',
        'masks_dir',
    }
    for key in path_keys:
        if key in data and isinstance(data[key], str):
            data[key] = Path(data[key])

    # Ensure numeric fields are floats
    for float_key in ['tr', 'smoothing_fwhm']:
        if float_key in data:
            data[float_key] = float(data[float_key])

    if 'confounds' in data:
        data['confounds'] = ConfoundsConfig(**data['confounds'])

    return Config(**data)


# -------------------------
# Standalone test
# -------------------------
if __name__ == '__main__':
    cfg = load_config()  # Uses internal configs/config.yaml
    print('Config loaded successfully!')
    print('BOLD glob:', cfg.bold_file_glob)
    print('TR:', cfg.tr)
