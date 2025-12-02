set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

DATA_TYPE="${1:-test}"
NUM_RUNS="${2:-1}"

echo "========================================"
echo "Generation Pipeline"
echo "Data type: $DATA_TYPE"
echo "Number of runs: $NUM_RUNS"
echo "========================================"

for RUN_ID in $(seq 1 $NUM_RUNS); do
    echo ""
    echo "========================================"
    echo "Run $RUN_ID / $NUM_RUNS"
    echo "========================================"

    python generate_responses.py --data_type "$DATA_TYPE" --run_id "$RUN_ID"

    echo "Run $RUN_ID completed!"
done

echo ""
echo "========================================"
echo "All $NUM_RUNS runs completed!"
echo "========================================"
