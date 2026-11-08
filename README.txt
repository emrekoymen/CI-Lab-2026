Evolution of cooperation
========================

Exercise 8 of the Computational Intelligence lab, using Python, DEAP, NumPy,
SciPy, and Matplotlib.

Hypothesis A: Increasing action noise reduces cooperation.

Each population has 100 individuals. Each individual follows one of 32 five-bit
strategies and plays 150 rounds against every other individual. We evolve the
population for 500 generations at noise levels 0, 0.01, and 0.05.


Setup
-----

Use Python 3.12. Run these commands from the folder that contains this README.

    python3.12 -m venv .venv
    source .venv/bin/activate        (Windows: .venv\Scripts\activate)
    python -m pip install -r requirements.txt


Notebook
--------

Open exercise_8.ipynb, and select the Python kernel from .venv.

Run the cells in order. The notebook calculates the matrices, runs one population
of 100 individuals at each noise level, and displays the results. It also reads
the saved 90-run study to show how results vary between runs.


Repeat the full study
---------------------

    python run_experiment.py --runs 30

This runs 30 populations per noise level, 90 runs in total. It replaces the CSV
files and figures directly in results/. The settings below are fixed in ipd.py
and are not rewritten by the script.


Files
-----

- ipd.py: strategy rules, matrix calculations, and DEAP evolution.
- run_experiment.py: runs, summaries, statistics, plotting, and CSV export.
- exercise_8.ipynb: explanations and live examples.
- results/run_summary.csv: one row per run, including all 32 final strategy counts.
- results/histories/: payoff, cooperation, and counts at each generation.
- results/A_noise_*.csv: expected payoff matrices.
- results/H_noise_*.csv: expected cooperation matrices.
- results/strategies.csv: strategy IDs, bits, and names.
- results/comparisons.csv: differences between every pair of noise levels, with
  95% bootstrap intervals, Mann-Whitney U and p, Holm-adjusted p, and
  Vargha-Delaney A12.
- results/pilot_spread.csv: spread of the first 10 runs (pilot) and of all 30
  runs per condition.
- results/pilot_comparisons.csv: the comparisons using only the 10 pilot runs.
- results/figures/: cooperation, payoff, strategy frequencies, and matrix plots.

Bits are [first action, after CC, after CD, after DC, after DD].
Use 0 = cooperate, 1 = defect.


Settings for the saved study
----------------------------

Runs per noise level: 30. Total: 90.

    Setting                      Value
    population                     100
    generations                    500
    rounds                         150
    crossover probability          0.7
    crossover bit probability      0.5
    mutation bit probability      0.01
    tournament size                  3
    elites                           2
    final window                   100

Base seed: 21261112. Each condition adds 10000; each run adds its index, starting at 0.
Bootstrap seed: 22161112. Bootstrap samples: 5000.

Python: 3.12.14.
numpy: 2.5.4.
deap: 1.4.4.
matplotlib: 3.11.2.
scipy: 1.18.1.

Each individual's fitness is its expected average payoff against the other 99
individuals. The matrices include action noise and all 150 rounds. Cooperation
is measured as the fraction of both players' actions that are cooperative.
