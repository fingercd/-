#!/usr/bin/env bash
set -euo pipefail

project_root="${VAD_PROJECT_ROOT:-/users/fotile/VAD}"
dataset_target="${1:-}"
link_path="$project_root/data/ucf_crime"

if [[ -z "$dataset_target" ]]; then
  echo "usage: $0 /absolute/path/to/UCF-Crime" >&2
  exit 2
fi
if [[ "$dataset_target" != /users/* ]] || [[ ! -d "$dataset_target" ]]; then
  echo "dataset target must be an existing directory below /users: $dataset_target" >&2
  exit 2
fi
if [[ -e "$link_path" || -L "$link_path" ]]; then
  echo "refusing to replace existing path: $link_path" >&2
  exit 3
fi

mkdir -p "$project_root/data"
ln -s "$dataset_target" "$link_path"
readlink -f "$link_path"

