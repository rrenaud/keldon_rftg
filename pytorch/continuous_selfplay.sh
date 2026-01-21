#!/bin/bash
#
# Continuous self-play data generation for RFTG training.
# Runs in background, generating training data until stopped.
#
# Usage:
#   ./continuous_selfplay.sh          # Start generation
#   ./continuous_selfplay.sh stop     # Stop generation
#   ./continuous_selfplay.sh status   # Check status
#   ./continuous_selfplay.sh tail     # Follow log output
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/training_data"
LOG_FILE="${OUTPUT_DIR}/continuous.log"
PID_FILE="${OUTPUT_DIR}/continuous.pid"

# Configuration
WORKERS=8
GAMES_PER_WORKER=500
PLAYERS=2
EXPANSION=0

start() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        echo "Already running (PID $(cat "$PID_FILE"))"
        echo "Use '$0 stop' to stop it first"
        exit 1
    fi

    mkdir -p "$OUTPUT_DIR"

    echo "Starting continuous self-play..."
    echo "  Workers: $WORKERS"
    echo "  Games per batch: $((WORKERS * GAMES_PER_WORKER))"
    echo "  Output: $OUTPUT_DIR"
    echo "  Log: $LOG_FILE"

    cd "$SCRIPT_DIR" && \
    nohup python3 parallel_selfplay.py \
        --continuous \
        --workers "$WORKERS" \
        --games-per-worker "$GAMES_PER_WORKER" \
        -p "$PLAYERS" \
        -e "$EXPANSION" \
        --output-dir "$OUTPUT_DIR" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "Started with PID $!"
    echo ""
    echo "Monitor with: $0 tail"
    echo "Stop with:    $0 stop"
}

stop() {
    if [ ! -f "$PID_FILE" ]; then
        echo "Not running (no PID file)"
        exit 0
    fi

    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping PID $PID..."
        kill "$PID"
        sleep 2
        if kill -0 "$PID" 2>/dev/null; then
            echo "Force killing..."
            kill -9 "$PID"
        fi
        echo "Stopped"
    else
        echo "Process not running"
    fi
    rm -f "$PID_FILE"
}

status() {
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        PID=$(cat "$PID_FILE")
        echo "Running (PID $PID)"
        echo ""

        # Show data stats
        if [ -d "$OUTPUT_DIR" ]; then
            TOTAL_SIZE=$(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1)
            NUM_FILES=$(find "$OUTPUT_DIR" -name "*.jsonl" 2>/dev/null | wc -l)
            NUM_BATCHES=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "batch_*" 2>/dev/null | wc -l)
            echo "Data: $TOTAL_SIZE in $NUM_FILES files ($NUM_BATCHES batches)"
        fi

        # Show last log line
        if [ -f "$LOG_FILE" ]; then
            echo ""
            echo "Latest:"
            tail -1 "$LOG_FILE"
        fi
    else
        echo "Not running"
        rm -f "$PID_FILE" 2>/dev/null
    fi
}

tail_log() {
    if [ -f "$LOG_FILE" ]; then
        tail -f "$LOG_FILE"
    else
        echo "No log file found at $LOG_FILE"
        exit 1
    fi
}

case "${1:-start}" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    status)
        status
        ;;
    tail)
        tail_log
        ;;
    *)
        echo "Usage: $0 {start|stop|status|tail}"
        exit 1
        ;;
esac
