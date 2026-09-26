#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_root/dist"
mojo build --emit shared-lib -I "$repo_root/src" \
  "$repo_root/src/kernels.mojo" -o "$repo_root/dist/libmojo-bayesian-optimization.so"
