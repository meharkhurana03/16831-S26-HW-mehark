import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DATA_DIR = 'data'
METRIC = 'Eval_AverageReturn'

def load(path):
    ea = EventAccumulator(path)
    ea.Reload()
    if METRIC not in ea.Tags().get('scalars', []):
        return None, None
    evs = ea.Scalars(METRIC)
    return np.array([e.step for e in evs]), np.array([e.value for e in evs])

def strip_timestamp(name):
    m = re.match(r'^(.+?)_\d{2}-\d{2}-\d{4}_\d{2}-\d{2}-\d{2}$', name)
    return m.group(1) if m else name

def plot_group(ax, dirs, title):
    seen = {}
    for d in dirs:
        label = strip_timestamp(os.path.basename(d))
        steps, values = load(d)
        if steps is None:
            continue
        if label in seen:
            # average duplicate runs
            s0, v0 = seen[label]
            n = min(len(v0), len(values))
            seen[label] = (s0[:n], (v0[:n] + values[:n]) / 2)
        else:
            seen[label] = (steps, values)

    for label, (steps, values) in sorted(seen.items()):
        ax.plot(steps, values, label=label)

    ax.set_xlabel('Iteration')
    ax.set_ylabel(METRIC)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

all_dirs = [os.path.join(DATA_DIR, d) for d in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, d))]
sb_dirs = [d for d in all_dirs if os.path.basename(d).startswith('q1_sb_')]
lb_dirs = [d for d in all_dirs if os.path.basename(d).startswith('q1_lb_')]

for dirs, title, fname in [
    (sb_dirs, 'Small Batch (q1_sb_)', 'fig_q1_sb.png'),
    (lb_dirs, 'Large Batch (q1_lb_)', 'fig_q1_lb.png'),
]:
    if not dirs:
        continue
    fig, ax = plt.subplots(figsize=(7, 5))
    plot_group(ax, dirs, title)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print('Saved', fname)

plt.show()
