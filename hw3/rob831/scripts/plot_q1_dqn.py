import argparse
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_run(exp_dir, x_tag='Train_EnvstepsSoFar', y_tag='Train_AverageReturn'):
    ea = EventAccumulator(exp_dir)
    ea.Reload()

    # align by step to handle off-by-one mismatches between logged tags
    step_to_x = {e.step: e.value for e in ea.Scalars(x_tag)}
    y_events = ea.Scalars(y_tag)
    matched = [(step_to_x[e.step], e.value) for e in y_events if e.step in step_to_x]
    X, Y = zip(*matched)

    return np.array(X), np.array(Y)


def load_runs(datadir, exp_prefix):
    """Load the latest run for each seed (e.g. q1_dqn_1, q1_dqn_2, q1_dqn_3)."""
    # Group directories by seed number (the digit immediately after exp_prefix)
    from collections import defaultdict
    seed_dirs = defaultdict(list)
    for d in glob.glob(os.path.join(datadir, exp_prefix + '*')):
        basename = os.path.basename(d)
        # seed id is the character right after the prefix
        seed_id = basename[len(exp_prefix)]
        seed_dirs[seed_id].append(d)

    runs = []
    for seed_id in sorted(seed_dirs):
        # take the most recently modified directory for this seed
        latest = max(seed_dirs[seed_id], key=os.path.getmtime)
        try:
            X, Y = load_run(latest)
            if len(X) == 0:
                print(f'  [warn] no data in {latest}')
                continue
            print(f'  loaded seed={seed_id}: {latest}  ({len(X)} steps)')
            runs.append((X, Y))
        except Exception as exc:
            print(f'  [warn] {latest}: {exc}')
    return runs


def interpolate_and_stack(runs, n_points=200):
    x_min = max(r[0].min() for r in runs)
    x_max = min(r[0].max() for r in runs)
    x_common = np.linspace(x_min, x_max, n_points)
    ys = [np.interp(x_common, X, Y) for X, Y in runs]
    return x_common, np.array(ys)


def plot_with_errorbars(ax, runs, label, color):
    if not runs:
        print(f'  [warn] no runs found for {label}')
        return
    
    x, ys = interpolate_and_stack(runs)
    mean, std = ys.mean(axis=0), ys.std(axis=0)
    ax.plot(x, mean, label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument('--out', type=str, default='q1_dqn_vs_ddqn.png')
    args = parser.parse_args()

    print('Loading DQN runs...')
    dqn_runs = load_runs(args.datadir, 'q1_dqn_')
    print('Loading DDQN runs...')
    ddqn_runs = load_runs(args.datadir, 'q1_doubledqn_')

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_with_errorbars(ax, dqn_runs,  label='DQN',        color='steelblue')
    plot_with_errorbars(ax, ddqn_runs, label='Double DQN', color='darkorange')

    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Average Return per Epoch', fontsize=12)
    ax.set_title('Q1: DQN vs Double DQN on LunarLander-v3\n(mean ± std over 3 seeds)', fontsize=13)

    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)

    print(f'Saved → {args.out}')
