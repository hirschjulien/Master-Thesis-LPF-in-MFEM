# Bash script to run the weak scaling experiment

#!/usr/bin/env bash
set -euo pipefail

EXE=./ws
OUT=data/weak-scaling.txt

# Number of ranks
NPS=(1 2 4 8)

# Repeat variable if needed
REPEATS=1

# Prevent oversubscription on macOS
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1

echo "== Weak scaling run =="
echo "Make sure your C++ has: const int mode = 1;"
echo "Executable: $EXE"
echo "Output:     $OUT"
echo "Ranks:      ${NPS[*]}"
echo "Repeats:    $REPEATS"
echo

for np in "${NPS[@]}"; do
  for rep in $(seq 1 "$REPEATS"); do
    echo "[np=$np rep=$rep] running..."
    mpirun -np "$np" "$EXE"
  done
done

echo
echo "Done. Results appended to: $OUT"
