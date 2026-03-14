import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_run(exp_dir, y_tag='Eval_AverageReturn'):
    ea = EventAccumulator(exp_dir)
    ea.Reload()
    Y = np.array([e.value for e in ea.Scalars(y_tag)])
    X = np.arange(len(Y))
    return X, Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument('--out', type=str, default='q2_cartpole_ac.png')
    args = parser.parse_args()

    matches = sorted(glob.glob(os.path.join(args.datadir, 'q2_10_10*')))
    if not matches:
        raise FileNotFoundError(f'No directory matching q2_10_10* in {args.datadir}')

    X, Y = load_run(matches[0])
    print(f'Loaded {len(Y)} eval points from {matches[0]}')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(X, Y, color='steelblue', linewidth=1.5)
    ax.axhline(200, color='gray', linestyle='--', linewidth=1, label='Target (200)')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Eval Average Return', fontsize=12)
    ax.set_title('Q2: Actor-Critic Sanity Check – CartPole-v0\n(ntu=10, ngsptu=10)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'Saved at {args.out}')
