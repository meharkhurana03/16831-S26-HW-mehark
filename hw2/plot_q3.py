import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = 'data'
METRIC = 'Eval_AverageReturn'

fig, ax = plt.subplots(figsize=(8, 5))

for d in sorted(os.listdir(DATA_DIR)):
    if not d.startswith('q3_'):
        continue
    m = re.search(r'_b(\d+)_r([\d.]+)', d)
    label = ('b=%s, lr=%s' % (m.group(1), m.group(2))) if m else d

    ea = EventAccumulator(os.path.join(DATA_DIR, d))
    ea.Reload()
    if METRIC not in ea.Tags().get('scalars', []):
        continue

    evs = ea.Scalars(METRIC)
    steps = np.array([e.step for e in evs])
    values = np.array([e.value for e in evs])

    ax.plot(steps, values, label=label)

ax.set_xlabel('Iteration')
ax.set_ylabel(METRIC)
ax.set_title('Q3: LunarLanderContinuous-v2')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('fig_q3.png', dpi=150, bbox_inches='tight')
print('Saved fig_q3.png')
plt.show()
