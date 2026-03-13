# Instructions to run the commands

First, run the setup code from the README-assn.md, and install `rob831/` as a module in the conda environment.

For Experiment 1
```bash
# switch to the correct conda env
bash q1.sh
```

To plot Experiment 1 curves:
```bash
# switch to the correct conda env
python plot_q1.py
```

For Experiment 2
```bash
bash q2.sh  # runs batch/lr sweep in parallel
```

To plot Experiment 2 curves:
```bash
python plot_q2.py
```

For Experiment 3
```bash
bash q3.sh  # runs in parallel
```

To plot Experiment 3 curves:
```bash
python plot_q3.py
```

For Experiment 4
```bash
bash q4.sh   # batch/lr sweep, runs in parallel
bash q4a.sh  # additional b=10000 runs, parallel
bash q4b.sh  # 4 variants (no-rtg, rtg, nnbaseline, rtg+nnbaseline) of the best b and r values, run in parallel across 2 GPUs
```

To plot Experiment 4 curves:
```bash
python plot_q4.py   # all sweep runs (RTG + NN baseline only, one per b/lr)
python plot_q4a.py  # b=10000, lr=0.02 runs only
python plot_q4b.py
```


For Experiment 5
```bash
# Split across two scripts to fit on available GPUs
bash q5a.sh  # lambda=0 and lambda=0.95
bash q5b.sh  # lambda=0.99 and lambda=1
```

To plot Experiment 5 curves:
```bash
python plot_q5.py
```

For Bonus 1 (parallel trajectory collection)
```bash
bash qb1.sh  # HalfCheetah with 4 parallel workers
```

Read the logs for TimeSinceStart.

For Bonus 2 (multi-step PG)
```bash
bash qb2.sh  # CartPole with ngspb=1, 5, 25 in parallel
```

To plot Bonus 2 curves:
```bash
python plot_qb2.py
```