#!/bin/bash
# Commits the files of this directory to the branch gpu-presolve-notes and pushes it to the
# fork (origin). The branch holds nothing else. Usage: snapshot.sh "<message>"
# The owner allowed the push on 2026-11-27. The snapshots before that are on the branch
# local/gpu-presolve-notes, which stays local.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
ref=refs/heads/gpu-presolve-notes
idx=$(git rev-parse --git-dir)/notes-snapshot.index
rm -f "$idx"
export GIT_INDEX_FILE=$idx
git add -f design_summaries/gpu_mip_presolve
tree=$(git write-tree)
rm -f "$idx"
unset GIT_INDEX_FILE
parent=$(git rev-parse -q --verify "$ref" || true)
if [ -n "$parent" ] && [ "$(git rev-parse "$parent^{tree}")" = "$tree" ]; then
  echo "no change since $parent"
else
  commit=$(git commit-tree "$tree" ${parent:+-p "$parent"} -m "${1:-Notes snapshot}")
  git update-ref "$ref" "$commit"
  echo "$ref -> $commit"
fi
git push -q origin "$ref" && echo "pushed to origin" || echo "NOT pushed: git push failed"
