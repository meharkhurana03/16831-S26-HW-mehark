import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = 'data'
METRIC = 'Eval_AverageReturn'

fig, ax = plt.subplots(figsize=(8, 5))

for d in sorted(os.listdir(DATA_DIR)):
    if not d.startswith('q2_'):
        continue
    m = re.match(r'q2_b(\d+)_r([\d.]+)', d)
    if not m:
        continue
    b, lr = int(m.group(1)), float(m.group(2))

    ea = EventAccumulator(os.path.join(DATA_DIR, d))
    ea.Reload()
    if METRIC not in ea.Tags().get('scalars', []):
        continue

    evs = ea.Scalars(METRIC)
    steps = np.array([e.step for e in evs])
    values = np.array([e.value for e in evs])

    ax.plot(steps, values, label='b=%d, lr=%s' % (b, lr))

ax.axhline(1000, color='gray', linestyle='--', linewidth=1, label='Optimum (1000)')
ax.set_xlabel('Iteration')
ax.set_ylabel(METRIC)
ax.set_title('Q2: InvertedPendulum-v4 Sweep')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('q2_curves.png', dpi=150, bbox_inches='tight')
print('Saved q2_curves.png')
plt.show()
