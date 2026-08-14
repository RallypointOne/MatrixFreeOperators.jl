#!/usr/bin/env bash
# Full comparison: both packages × {1 thread, all cores} × size sweep, then the
# congruency check. Results land in results/timings.csv.
#
# Usage: ./run.sh [comma-separated n list, default 128,...,16384]
set -euo pipefail
cd "$(dirname "$0")"
SIZES="${1:-128,512,1024,2048,4096,8192,16384}"

julia --project=. --startup-file=no -e 'using Pkg; Pkg.instantiate()'

for t in 1 auto; do
    echo "=== threads: $t ==="
    julia --project=. --startup-file=no -t "$t" chmy.jl "$SIZES"
    julia --project=. --startup-file=no -t "$t" mfo.jl "$SIZES"
done

julia --project=. --startup-file=no compare.jl "$SIZES"
echo "timings: results/timings.csv"
