#!/bin/bash
# Verification suite for the norm -> radial -> gram reordering, the new radial
# spectral/radiality metrics, and the norm_helper cost controls.
#
#   cd <repo root>  &&  source llm_env/activate.sh
#   bash resources/torchtitan/tests/unit_tests/disco_metrics/run_all.sh
#
# Must be run from the repository root (the scripts add resources/torchtitan to
# sys.path and the disco_ordering check reads disco.py by relative path).
set -u
D="$(cd "$(dirname "$0")" && pwd)"
rc=0
for t in verify_power_iteration verify_norm_helper verify_radial_helper verify_disco_ordering verify_disco_construct verify_shard_predicate verify_radial_batched verify_gram_helper; do
  printf '%-28s ' "$t"
  if python "$D/$t.py" > "/tmp/$t.out" 2>&1; then echo PASS; else echo "FAIL  (see /tmp/$t.out)"; rc=1; fi
done
exit $rc
