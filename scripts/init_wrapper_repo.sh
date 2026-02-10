# scripts/init_wrapper_repo.sh
# What: Bootstraps per-repo SLURM launcher config files and ignore rules for wrapper usage.
# Why: Lets each project repo keep private MN5 credentials local while committing a sanitized example.
# RELEVANT FILES: README.md, examples/remote_launcher_config.mn5.example.py, launcher/templates/config.py.template
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/init_wrapper_repo.sh [TARGET_REPO] [--force]

Behavior:
  - Creates TARGET_REPO/.slurm/
  - Copies MN5 example config into:
      .slurm/remote_launcher_config.mn5.example.py
      .slurm/remote_launcher_config.mn5.py
  - Appends gitignore rules (if missing):
      .slurm/*.py
      !.slurm/*.example.py

Examples:
  scripts/init_wrapper_repo.sh .
  scripts/init_wrapper_repo.sh /path/to/project-repo --force
EOF
}

target_repo="."
force="false"

for arg in "$@"; do
  case "$arg" in
    -h|--help)
      usage
      exit 0
      ;;
    --force)
      force="true"
      ;;
    *)
      target_repo="$arg"
      ;;
  esac
done

if [[ ! -d "$target_repo" ]]; then
  echo "ERROR: target repo not found: $target_repo" >&2
  exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
launcher_root="$(cd "$script_dir/.." && pwd)"
source_config="$launcher_root/examples/remote_launcher_config.mn5.example.py"

if [[ ! -f "$source_config" ]]; then
  echo "ERROR: source config not found: $source_config" >&2
  exit 1
fi

slurm_dir="$target_repo/.slurm"
private_cfg="$slurm_dir/remote_launcher_config.mn5.py"
example_cfg="$slurm_dir/remote_launcher_config.mn5.example.py"
gitignore_path="$target_repo/.gitignore"

mkdir -p "$slurm_dir"

if [[ "$force" == "true" || ! -f "$example_cfg" ]]; then
  cp "$source_config" "$example_cfg"
  echo "Wrote $example_cfg"
else
  echo "Kept existing $example_cfg"
fi

if [[ "$force" == "true" || ! -f "$private_cfg" ]]; then
  cp "$source_config" "$private_cfg"
  echo "Wrote $private_cfg"
else
  echo "Kept existing $private_cfg"
fi

touch "$gitignore_path"
if ! grep -qxF ".slurm/*.py" "$gitignore_path"; then
  echo ".slurm/*.py" >> "$gitignore_path"
  echo "Added .slurm/*.py to $gitignore_path"
fi
if ! grep -qxF "!.slurm/*.example.py" "$gitignore_path"; then
  echo "!.slurm/*.example.py" >> "$gitignore_path"
  echo "Added !.slurm/*.example.py to $gitignore_path"
fi

cat <<EOF

Next steps:
1) Install launcher in this repo (once):
   uv add --editable "$launcher_root"
2) Edit: $private_cfg
3) Dry-run:
   uv run slurm-launcher --dry-run
4) Submit:
   uv run slurm-launcher
EOF
