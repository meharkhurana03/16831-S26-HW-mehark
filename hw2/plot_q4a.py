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
    if not re.match(r'q4.*_b10000_lr0\.02', d) or not os.path.isdir(os.path.join(DATA_DIR, d)):
        continue

    m = re.search(r'_lr([\d.]+)', d)
    lr = m.group(1) if m else '?'
    rtg = '_rtg' in d
    nn = '_nnbaseline' in d
    label = 'lr=%s' % lr
    if rtg:
        label += ', rtg'
    if nn:
        label += ', nn_baseline'

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
ax.set_title('Q4a: HalfCheetah-v4 (b=10000)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_q4a.png', dpi=150, bbox_inches='tight')
print('Saved fig_q4a.png')
plt.show()
