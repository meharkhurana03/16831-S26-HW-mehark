import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = 'data'
METRIC = 'Eval_AverageReturn'

fig, ax = plt.subplots(figsize=(8, 5))

seen = {}
for d in sorted(os.listdir(DATA_DIR)):
    if not os.path.isdir(os.path.join(DATA_DIR, d)):
        continue
    if not re.match(r'q4.*_b30000_', d):
        continue
    # require lr/r = 0.02
    if not re.search(r'[_r]0\.02', d):
        continue

    rtg = '_rtg' in d
    nn = '_nnbaseline' in d
    label = 'no_rtg, no_baseline'
    if rtg and nn:
        label = 'rtg, nn_baseline'
    elif rtg:
        label = 'rtg'
    elif nn:
        label = 'nn_baseline'

    ea = EventAccumulator(os.path.join(DATA_DIR, d))
    ea.Reload()
    if METRIC not in ea.Tags().get('scalars', []):
        continue

    evs = ea.Scalars(METRIC)
    steps = np.array([e.step for e in evs])
    values = np.array([e.value for e in evs])

    if label in seen:
        s0, v0 = seen[label]
        n = min(len(v0), len(values))
        seen[label] = (s0[:n], (v0[:n] + values[:n]) / 2)
    else:
        seen[label] = (steps, values)

for label, (steps, values) in sorted(seen.items()):
    ax.plot(steps, values, label=label)

ax.set_xlabel('Iteration')
ax.set_ylabel(METRIC)
ax.set_title('Q4b: HalfCheetah-v4 (b=30000, lr=0.02)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_q4b.png', dpi=150, bbox_inches='tight')
print('Saved fig_q4b.png')
plt.show()
