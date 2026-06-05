#!/usr/bin/env bash
# Launch the Phase-1 recompute in the BACKGROUND so it survives logout / closed SSH.
# Usage:
#   ./phase1_launch_bg.sh            # 20 seeds (default)
#   SEEDS=2 ./phase1_launch_bg.sh    # quick validation pass
#
# It detaches via setsid+nohup (no need to keep the terminal open).
# Monitor / manage with the commands printed at the end.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SEEDS="${SEEDS:-20}"
RUN="$SCRIPT_DIR/phase1_run_recompute.sh"
OUT="$SCRIPT_DIR/phase1_console.log"          # full console output of the whole sweep
PIDF="$SCRIPT_DIR/phase1.pid"

[ -f "$RUN" ] || { echo "FATAL: $RUN not found"; exit 1; }

# already running?
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "Already running (PID $(cat "$PIDF")). Tail: tail -f $OUT"; exit 0
fi

# detach: setsid makes it a new session leader; nohup ignores SIGHUP on logout
SEEDS="$SEEDS" setsid nohup bash "$RUN" > "$OUT" 2>&1 &
echo $! > "$PIDF"
sleep 1
echo "Launched in background. PID $(cat "$PIDF"), SEEDS=$SEEDS"
echo
echo "  watch progress :  tail -f $OUT"
echo "  is it alive    :  kill -0 \$(cat $PIDF) && echo running || echo done"
echo "  count results  :  find $SCRIPT_DIR/results_phase1 -name '*.json' | wc -l"
echo "  failures so far:  cat $SCRIPT_DIR/results_phase1/_status.txt"
echo "  stop it        :  kill \$(cat $PIDF)"
echo
echo "You can safely close this SSH session now."
