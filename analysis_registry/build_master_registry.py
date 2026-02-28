#!/usr/bin/env python3
"""
Build the master analysis registry by combining all YAML files in analyses/
and generating a human-readable Markdown summary (block format).
"""

from pathlib import Path

import yaml

ANALYSIS_DIR = Path(__file__).parent / 'analyses'
MASTER_YAML = Path(__file__).parent / 'master_registry.yaml'
MASTER_MD = Path(__file__).parent / 'README.md'

# Preferred stage order for grouping in the markdown output
STAGE_ORDER = [
    'preprocessing',
    'time series',
    'time series and group',
    'group',
    'other',
]


def load_yaml_files(yaml_dir: Path):
    """Load all YAML files in a directory and return a list of dicts."""
    all_entries = []
    for yaml_file in sorted(yaml_dir.glob('*.yaml')):
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            all_entries.append(data)
    return all_entries


def write_master_yaml(entries, output_file: Path):
    with open(output_file, 'w') as f:
        yaml.safe_dump(entries, f, sort_keys=False)


def stringify_entry(entry):
    """Convert YAML lists, None, or scalars into strings."""
    if entry is None:
        return 'None'
    if isinstance(entry, list):
        if not entry:
            return 'None'
        return ', '.join(str(x) for x in entry)
    return str(entry)


def clean_text_field(value):
    """
    Normalize text fields for Markdown:
    - Convert None → 'None'
    - Convert lists → single string via stringify_entry
    - Remove extra whitespace
    - Replace newlines with spaces
    """
    s = stringify_entry(value)
    return s.strip().replace('\n', ' ')


def sort_by_stage(entries):
    """Sort entries according to STAGE_ORDER; unknown stages go last."""
    stage_rank = {stage: i for i, stage in enumerate(STAGE_ORDER)}
    return sorted(
        entries, key=lambda e: stage_rank.get(e.get('stage', ''), len(STAGE_ORDER))
    )


def render_md_list(label, value):
    """Render scalar or list as markdown with bullets for lists."""
    if isinstance(value, list):
        lines = [f'**{label}:**']
        for item in value:
            lines.append(f'- {item}')
        return '\n'.join(lines) + '\n'
    return f'**{label}:** {value}<br>'


def build_results_links(entry):
    """Build HTML links for results files using code_dir when available."""
    result_files = entry.get('results_files') or []
    if not result_files:
        return 'None'

    code_dir = entry.get('code_dir')
    fallback_id = (entry.get('id') or '').replace(' ', '_')

    links = []
    for rf in result_files:
        rf_str = str(rf)
        if rf_str.startswith('http://') or rf_str.startswith('https://'):
            href = rf_str
        else:
            rel_path = rf_str.lstrip('/')
            if code_dir:
                href = f'../{code_dir}/{rel_path}'
            else:
                href = f'../analyses/{fallback_id}/{rel_path}'
        links.append(f'<a href="{href}">{rf_str}</a>')

    return ', '.join(links)


def write_markdown_summary(entries, output_file: Path):
    md_lines = ['# Master Analysis Registry\n']

    # --- Summary Table ---
    md_lines.append('# Summary Table\n')
    md_lines.append('<table>')

    for stage in STAGE_ORDER:
        stage_entries = [e for e in entries if e.get('stage') == stage]
        if not stage_entries:
            continue

        # Stage header row
        md_lines.append(
            f'<tr><th colspan="5"><strong>{stage.title()}</strong></th></tr>'
        )
        md_lines.append(
            '<tr><td>ID</td><td>Description</td><td>Status</td><td>Results Files</td><td>Notes</td></tr>'
        )

        for e in stage_entries:
            desc = clean_text_field(e.get('description'))
            notes = clean_text_field(e.get('notes'))
            status = clean_text_field(e.get('status'))
            pretty_id = e.get('id', 'None').replace('_', ' ')

            # Results files
            links = build_results_links(e)

            md_lines.append(
                f'<tr><td>{pretty_id}</td><td>{desc}</td><td>{status}</td><td>{links}</td><td>{notes}</td></tr>'
            )

    md_lines.append('</table>\n\n')

    # Divider before detailed section
    md_lines.append('\n')
    md_lines.append('# Detailed Reports\n')

    for stage in STAGE_ORDER:
        stage_entries = [e for e in entries if e.get('stage') == stage]
        if not stage_entries:
            continue

        md_lines.append(f'\n## {stage.title()}\n')
        for e in stage_entries:
            pretty_id = e.get('id', 'Unknown ID')
            md_lines.append(f'### {pretty_id}')

            md_lines.append(f'**Name:** {clean_text_field(e.get("name"))}<br>')
            md_lines.append(
                f'**Description:** {clean_text_field(e.get("description"))}<br>'
            )
            md_lines.append(
                f'**Code Directory:** {clean_text_field(e.get("code_dir"))}<br>'
            )
            md_lines.append(
                f'**Dependencies:** {stringify_entry(e.get("dependencies"))}<br>'
            )
            md_lines.append(render_md_list('Script Entry', e.get('script_entry')))
            md_lines.append(render_md_list('Notebook Entry', e.get('notebook_entry')))
            md_lines.append(render_md_list('Other Files', e.get('other_files')))
            md_lines.append(
                f'**Output Directory:** {stringify_entry(e.get("output_dir"))}<br>'
            )
            md_lines.append(f'**Results Files:** {build_results_links(e)}<br>')

            md_lines.append(
                f'**Hypothesis:** {clean_text_field(e.get("hypothesis"))}<br>'
            )
            md_lines.append(
                f'**Conclusion:** {clean_text_field(e.get("conclusion"))}<br>'
            )
            md_lines.append(f'**Notes:** {clean_text_field(e.get("notes"))}<br>')
            md_lines.append(f'**Status:** {clean_text_field(e.get("status"))}<br>')
            md_lines.append(
                f'**Last Updated:** {clean_text_field(e.get("last_updated"))}<br>'
            )
            md_lines.append(f'**Authors:** {stringify_entry(e.get("authors"))}<br>')
            md_lines.append('\n---\n')

    with open(output_file, 'w') as f:
        f.write('\n'.join(md_lines))


def main():
    entries = load_yaml_files(ANALYSIS_DIR)
    # Preserve raw YAML output as-is
    write_master_yaml(entries, MASTER_YAML)
    # Create Markdown with grouping by stage
    write_markdown_summary(entries, MASTER_MD)
    print(f'Master YAML written to {MASTER_YAML}')
    print(f'Markdown summary written to {MASTER_MD}')


if __name__ == '__main__':
    main()
