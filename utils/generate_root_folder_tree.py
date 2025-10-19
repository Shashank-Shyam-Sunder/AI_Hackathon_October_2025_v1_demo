# utils/generate_root_folder_tree.py
import os
from pathlib import Path

# --- config ---
IGNORED_DIRS = {
    '.git', '.idea', '.venv', '__pycache__', '.mypy_cache', '.pytest_cache',
    '.ruff_cache', 'node_modules', '.tox', '.nox', '.history'
}
IGNORED_FILES = {
    '.DS_Store', 'Thumbs.db'
}
OUTPUT_FILENAME = "clean_folder_structure.txt"
# --------------

def iter_entries(dir_path: Path):
    """Return (dirs, files) sorted, excluding ignored and temp files."""
    dirs, files = [], []
    for entry in dir_path.iterdir():
        name = entry.name
        if entry.is_dir():
            if name in IGNORED_DIRS:
                continue
            dirs.append(entry)
        else:
            if name in IGNORED_FILES:
                continue
            if name.startswith('~$'):  # Office temp files
                continue
            files.append(entry)
    dirs.sort(key=lambda p: p.name.lower())
    files.sort(key=lambda p: p.name.lower())
    return dirs, files

def tree(dir_path: Path, prefix: str, out_lines: list[str]):
    dirs, files = iter_entries(dir_path)
    entries = dirs + files
    count = len(entries)
    for idx, entry in enumerate(entries):
        connector = '└── ' if idx == count - 1 else '├── '
        out_lines.append(f"{prefix}{connector}{entry.name}")
        if entry.is_dir():
            extension = '    ' if connector == '└── ' else '│   '
            tree(entry, prefix + extension, out_lines)

def main():
    script_dir = Path(__file__).resolve().parent
    # Repo root = parent of utils/
    repo_root = script_dir.parent
    output_path = script_dir / OUTPUT_FILENAME

    lines = ['.']
    tree(repo_root, prefix='', out_lines=lines)

    # Print to console
    print('\n'.join(lines))

    # Save into utils/
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\n✅ Saved to {output_path.relative_to(repo_root)}")

if __name__ == "__main__":
    main()
