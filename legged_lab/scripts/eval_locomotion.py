"""Evaluate g1_locomotion_v2 velocity-tracking quality from tensorboard logs and
emit an extend/stop decision for the overnight orchestrator.

Aggregates Episode_Reward/track_lin_vel_xy across ALL run dirs (resume makes a new
timestamped dir but rsl_rl keeps the global iteration count, so steps stay
continuous). The per-iter reward is noisy (~+/-0.04), so we compare BLOCK MEANS
(last 500 iters vs the 500-iter block ~2000 iters earlier) to judge real progress.

Prints one line:  ITER=<n> TRACK=<x> DELTA=<d> DECISION=<tok>
  GOOD      -> track >= GOOD_BAR (deployable-quality tracking, stop)
  CONVERGED -> below bar but plateaued/degrading (more rounds won't help, stop)
  CAP       -> iter >= CAP (safety bound hit, stop)
  EXTEND    -> below bar and still improving (undertrained, train more)
"""
import glob, sys
from tensorboard.backend.event_processing import event_accumulator

GOOD_BAR = 0.85   # block-mean track reward considered deployable-good -> stop
IMPROVE = 0.010   # min block-mean gain over WINDOW iters to count as "still learning"
WINDOW = 2000     # iters between the two comparison blocks
BLOCK = 500       # block width (iters) for smoothing out reward noise
CAP = 35000       # never train past this many iters

TAG = "Episode_Reward/track_lin_vel_xy"


def block_mean(pts, lo, hi):
    vs = [v for s, v in pts if lo <= s <= hi]
    return sum(vs) / len(vs) if vs else None


pts = []
for d in glob.glob("logs/g1_locomotion_v2/2026-*"):
    try:
        ea = event_accumulator.EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        if TAG in ea.Tags().get("scalars", []):
            pts += [(x.step, x.value) for x in ea.Scalars(TAG)]
    except Exception:
        pass

if not pts:
    print("ITER=0 TRACK=0 DELTA=0 DECISION=EXTEND")
    sys.exit(0)

pts.sort()
last = pts[-1][0]
now = block_mean(pts, last - BLOCK, last)
prev = block_mean(pts, last - WINDOW - BLOCK, last - WINDOW)
if prev is None:  # not enough history yet -> treat as still improving
    prev = pts[0][1]
delta = now - prev

if now >= GOOD_BAR:
    dec = "GOOD"
elif last >= CAP:
    dec = "CAP"
elif delta >= IMPROVE:
    dec = "EXTEND"
else:
    dec = "CONVERGED"

print(f"ITER={last} TRACK={now:.4f} DELTA={delta:.4f} DECISION={dec}")
