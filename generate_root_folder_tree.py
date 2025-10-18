import os

IGNORED_DIRS = {'.git', '.idea', '.venv', '__pycache__', '.mypy_cache', '.pytest_cache'}
OUTPUT_FILE = "clean_folder_structure.txt"


def tree(dir_path: str, prefix: str = '', out_lines=None):
    files = sorted(os.listdir(dir_path))
    files = [f for f in files if f not in IGNORED_DIRS]

    pointers = ['├── '] * (len(files) - 1) + ['└── ']

    for pointer, name in zip(pointers, files):
        path = os.path.join(dir_path, name)
        out_lines.append(prefix + pointer + name)
        if os.path.isdir(path):
            extension = '│   ' if pointer == '├── ' else '    '
            tree(path, prefix + extension, out_lines)


if __name__ == '__main__':
    lines = ['.']
    tree('.', out_lines=lines)

    # Print to console
    for line in lines:
        print(line)

    # Save to file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n✅ Saved to {OUTPUT_FILE}")
