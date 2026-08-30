#!/usr/bin/env bash
set -euo pipefail

project_root="${VAD_PROJECT_ROOT:-/users/fotile/VAD}"
incoming="$project_root/.incoming"
bundle="$incoming/vadbench.bundle"
branch="feat/video-encoder-benchmark-framework"

if [[ ! -f "$bundle" ]]; then
  echo "missing git bundle: $bundle" >&2
  exit 2
fi

cd "$project_root"
if [[ ! -d .git ]]; then
  git init
fi
git bundle verify "$bundle"
git fetch "$bundle" "$branch"
git checkout -B "$branch" FETCH_HEAD

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin 'https://github.com/fingercd/Abnormal-Video-Detection..git'
else
  git remote add origin 'https://github.com/fingercd/Abnormal-Video-Detection..git'
fi

mkdir -p external wheels data/smoke
tar -xzf "$incoming/vadbench-external.tar.gz" -C external
tar -xzf "$incoming/vadbench-wheels.tar.gz" -C wheels
install -m 0644 "$incoming/surveillance-smoke.mp4" data/smoke/surveillance-smoke.mp4

git rev-parse HEAD
git status --short --branch
