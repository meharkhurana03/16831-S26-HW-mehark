# Environment Setup

Running this code requires the reuse of the conda environment from the previous homework (hw2).

Please uninstall the old rob831 module, and install the one in this folder:

```
pip uninstall rob831
pip install -e .
```

All experiments (as listed in the submission pdf) should run now.

Here are the commands to generate the plots:

```
# for q1
python rob831/scripts/plot_q1_dqn.py

# for q2
python rob831/scripts/plot_q2_cartpole.py

# for q3
python rob831/scripts/plot_q3_invpendulum.py
```