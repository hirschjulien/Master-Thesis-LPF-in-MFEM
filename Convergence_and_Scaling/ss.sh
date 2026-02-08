# Bash script to run the strong scaling experiment

#!/usr/bin/env bash
set -euo pipefail


EXE=./ss  
OUT=data/strong-scaling.txt  

# avoid oversubscription
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1

# Strong-scaling ranks to test
NPS=(1 2 4 6 8)

# Repeat variable if wanted
REPEATS=1

echo "== Strong scaling run =="
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
