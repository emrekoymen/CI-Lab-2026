"""Run repeated experiments and save their results. Use: python run_experiment.py"""

import argparse
import csv
import importlib.metadata
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ipd import DEFAULT_SETTINGS, BITS, LABELS, NAMES, STRATEGIES, build_matrices, run_evolution


NOISE_LEVELS = (0.0, 0.01, 0.05)
BASE_SEED = 21261112  # reproduces the saved 90-run study
BOOTSTRAP_SAMPLES = 5000
RESULTS_FOLDER = Path(__file__).resolve().parent / "results"
COLORS = {0.0: "#0072B2", 0.01: "#D55E00", 0.05: "#009E73"}


def summarize_run(history, noise, run, seed, settings):
    """Reduce one run to its final counts and last-100-generation averages."""

    final = history[-1]
    window = history[-settings["final_window"]:]

    summary = {
        "noise": noise,
        "run": run,
        "seed": seed,
        **settings,
        "final_payoff": final["payoff"],
        "final_cooperation": final["cooperation"],
        "window_payoff": float(np.mean([row["payoff"] for row in window])),
        "window_cooperation": float(np.mean([row["cooperation"] for row in window])),
    }

    for bits in BITS:

        summary[f"final_count_{bits}"] = final[f"count_{bits}"]
        counts = [row[f"count_{bits}"] for row in window]
        summary[f"window_mean_count_{bits}"] = float(np.mean(counts))

    return summary


def run_study(runs=30, base_seed=BASE_SEED, settings=None, matrices=None):
    """Run the populations and return results in memory. This function saves no files."""

    if settings is None:
        settings = DEFAULT_SETTINGS.copy()

    if not 1 <= runs <= 10000:
        raise ValueError("Choose between 1 and 10000 runs per noise level.")

    if matrices is None:
        matrices = {}

    histories = {}
    summaries = []

    for condition, noise in enumerate(NOISE_LEVELS):

        if noise not in matrices:
            matrices[noise] = build_matrices(noise, settings["rounds"])

        A, H = matrices[noise]
        histories[noise] = []

        for run in range(runs):

            seed = base_seed + condition * 10000 + run  # a different seed for every run
            history = run_evolution(A, H, seed, settings)

            histories[noise].append(history)
            summaries.append(summarize_run(history, noise, run + 1, seed, settings))

        print(f"Noise {noise:g}: finished {runs} run(s) of {settings['population']} individuals.", flush=True)

    return histories, summaries, matrices


def bootstrap_difference(reference, treatment, rng, samples=BOOTSTRAP_SAMPLES):
    """Estimate how uncertain the difference between two medians is."""

    reference = np.array(reference)
    treatment = np.array(treatment)

    if len(reference) < 2 or len(treatment) < 2:
        raise ValueError("Confidence intervals need repeated runs in each group.")

    # Make new groups by randomly picking existing runs, allowing repeats.

    reference_groups = rng.choice(reference, size=(samples, len(reference)), replace=True)
    treatment_groups = rng.choice(treatment, size=(samples, len(treatment)), replace=True)

    differences = np.median(treatment_groups, axis=1) - np.median(reference_groups, axis=1)

    # The middle 95% of these differences gives the confidence interval.

    low, high = np.quantile(differences, [0.025, 0.975])
    effect = np.median(treatment) - np.median(reference)

    return float(effect), float(low), float(high)


def make_comparisons(summaries, bootstrap_seed=BASE_SEED + 900000):
    """Compare cooperation and payoff against the zero-noise condition."""

    rng = np.random.default_rng(bootstrap_seed)
    comparisons = []

    for measure in ("window_cooperation", "window_payoff"):

        reference = [row[measure] for row in summaries if row["noise"] == 0]

        for noise in NOISE_LEVELS[1:]:

            treatment = [row[measure] for row in summaries if row["noise"] == noise]
            effect, low, high = bootstrap_difference(reference, treatment, rng)

            comparisons.append({
                "measure": measure,
                "reference_noise": 0.0,
                "treatment_noise": noise,
                "runs_per_group": len(treatment),
                "median_difference": effect,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
            })

    return comparisons


def read_matrix(path):
    """Read labels and numbers from one matrix CSV."""

    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))  # each CSV row becomes a list of text values

    labels = rows[0][1:]  # first row, except the corner cell
    values = []

    for row in rows[1:]:  # skip the header

        numbers = []

        for value in row[1:]:  # skip the row label
            numbers.append(float(value))  # convert text to a number

        values.append(numbers)

    return labels, np.array(values)


def read_numeric_rows(path):
    """Read a results CSV, using its column names as dictionary keys."""

    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for row in rows:

        for column in row:
            row[column] = float(row[column])  # CSV values start as text

    return rows


def load_results(folder=RESULTS_FOLDER):
    """Read saved runs without running evolution again."""

    folder = Path(folder)
    summaries = read_numeric_rows(folder / "run_summary.csv")
    histories = {noise: [] for noise in NOISE_LEVELS}
    matrices = {}

    # Use the summary to identify the saved runs.

    for summary in summaries:

        noise = summary["noise"]
        run = int(summary["run"])
        path = folder / "histories" / f"noise_{noise:g}_run_{run:03d}.csv"

        histories[noise].append(read_numeric_rows(path))

    for noise in NOISE_LEVELS:

        _, A = read_matrix(folder / f"A_noise_{noise:g}.csv")
        _, H = read_matrix(folder / f"H_noise_{noise:g}.csv")

        matrices[noise] = (A, H)

    return histories, summaries, matrices


def plot_outcomes(histories, population_size=100):
    """Show cooperation and payoff over generations. Return a figure without saving."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    number_of_runs = len(histories[0.0])

    for noise in NOISE_LEVELS:

        for ax, measure in zip(axes, ("cooperation", "payoff")):

            values = []

            for history in histories[noise]:
                values.append([row[measure] for row in history])

            values = np.array(values)

            if measure == "cooperation":
                values = values * 100  # show cooperation as a percentage

            generations = np.arange(values.shape[1])
            middle = np.median(values, axis=0)

            ax.plot(generations, middle, color=COLORS[noise], label=f"noise {noise:g}")

            if number_of_runs > 1:
                ax.fill_between(generations, values.min(axis=0), values.max(axis=0),
                                color=COLORS[noise], alpha=0.15)

            ax.set_xlabel("Generation")

    axes[0].set(ylabel="Cooperative actions (%)", ylim=(0, 100))
    axes[1].set(ylabel="Payoff per round", ylim=(0, 5))
    axes[0].legend()

    if number_of_runs == 1:
        title = f"Live experiment | 1 run per noise level | {population_size} individuals per run"
    else:
        title = f"{number_of_runs} runs per noise level | {population_size} individuals per run | median and full range"

    fig.suptitle(title)

    return fig


def plot_strategy_frequencies(histories, population_size=100):
    """Show which strategies become common in the populations."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, layout="constrained")
    tracked = [0, 5, 6, 7, 31]  # type 00111 keeps its bit-string label

    for ax, noise in zip(axes, NOISE_LEVELS):

        runs = []

        for history in histories[noise]:
            runs.append([[row[f"count_{bits}"] for bits in BITS] for row in history])

        percentages = np.mean(runs, axis=0) * 100 / population_size
        generations = np.arange(len(percentages))

        for strategy_id in tracked:
            ax.plot(generations, percentages[:, strategy_id], label=LABELS[strategy_id])

        other_ids = [i for i in range(32) if i not in tracked]
        other = percentages[:, other_ids].sum(axis=1)

        ax.plot(generations, other, color="gray", ls="--", label="Other strategies")
        ax.set(title=f"Noise {noise:g}", xlabel="Generation", ylim=(0, 100))

    axes[0].set_ylabel("Population (%)")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))

    return fig


def plot_matrix(matrix, noise, kind):
    """Show all 32 strategies in A or H."""

    maximum = 5 if kind == "A" else 1
    title = "Expected payoff A" if kind == "A" else "Cooperation H"

    fig, ax = plt.subplots(figsize=(13, 12), layout="constrained")
    image = ax.imshow(matrix, vmin=0, vmax=maximum, cmap="viridis")

    ax.set_xticks(range(32), LABELS, rotation=90, fontsize=7)
    ax.set_yticks(range(32), LABELS, fontsize=7)
    ax.set(title=f"{title} | noise {noise:g}", xlabel="Column opponent", ylabel="Row strategy")

    fig.colorbar(image, ax=ax, shrink=0.75)

    return fig


def plot_classical_payoffs(matrices):
    """Show a smaller selection of A, with the payoffs printed in the cells."""

    selected = [0, 31, 5, 6, 7]
    names = [NAMES.get(i, BITS[i]) for i in selected]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")

    for ax, noise in zip(axes, NOISE_LEVELS):

        A, _ = matrices[noise]
        values = A[np.ix_(selected, selected)]  # choose these rows and columns

        ax.imshow(values, vmin=0, vmax=5, cmap="Blues")
        ax.set_xticks(range(5), names, rotation=45, ha="right")
        ax.set_yticks(range(5), names)
        ax.set(title=f"Payoff A | noise {noise:g}", xlabel="Column opponent")

        for row in range(5):

            for column in range(5):

                value = values[row, column]
                text_color = "white" if value > 2.5 else "black"

                ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=text_color)

    return fig


def plot_final_window(summaries):
    """Show each repeated run's average over its final 100 generations."""

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")

    for ax, measure in zip(axes, ("window_cooperation", "window_payoff")):

        for position, noise in enumerate(NOISE_LEVELS):

            values = np.array([row[measure] for row in summaries if row["noise"] == noise])

            if measure == "window_cooperation":
                values = values * 100

            positions = position + np.linspace(-0.12, 0.12, len(values))

            ax.scatter(positions, values, color=COLORS[noise], alpha=0.65, s=20)
            ax.plot([position - 0.2, position + 0.2], [np.median(values)] * 2, color="black")

        ax.set_xticks(range(3), [str(noise) for noise in NOISE_LEVELS])
        ax.set_xlabel("Action noise")

    axes[0].set(ylabel="Cooperative actions (%)", ylim=(0, 100))
    axes[1].set_ylabel("Payoff per round")
    window = int(summaries[0]["final_window"])
    population_size = int(summaries[0]["population"])
    fig.suptitle(f"Final {window} generations | each dot is one run of {population_size} individuals")

    return fig


def write_csv(path, rows):
    """Save a list of dictionaries as a CSV table."""

    with path.open("w", newline="", encoding="utf-8") as file:

        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(histories, summaries, matrices, comparisons, folder=RESULTS_FOLDER):
    """Save files only when explicitly called by the command-line script."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "histories").mkdir(exist_ok=True)
    (folder / "figures").mkdir(exist_ok=True)

    # Remove old generated histories before replacing the study.

    for old_file in (folder / "histories").glob("noise_*_run_*.csv"):
        old_file.unlink()

    write_csv(folder / "run_summary.csv", summaries)
    write_csv(folder / "comparisons.csv", comparisons)

    strategy_rows = []

    for strategy_id, genes in enumerate(STRATEGIES):

        row = {
            "id": strategy_id,
            "bits": BITS[strategy_id],
            "name": NAMES.get(strategy_id, ""),
            "label": LABELS[strategy_id],
        }

        for name, bit in zip(("first", "after_CC", "after_CD", "after_DC", "after_DD"), genes):
            row[name] = int(bit)

        strategy_rows.append(row)

    write_csv(folder / "strategies.csv", strategy_rows)

    for noise in NOISE_LEVELS:

        for run, history in enumerate(histories[noise], start=1):
            write_csv(folder / "histories" / f"noise_{noise:g}_run_{run:03d}.csv", history)

        for kind, matrix in zip(("A", "H"), matrices[noise]):

            path = folder / f"{kind}_noise_{noise:g}.csv"

            with path.open("w", newline="", encoding="utf-8") as file:

                writer = csv.writer(file)
                writer.writerow(["row_strategy / column_opponent", *LABELS])

                for label, values in zip(LABELS, matrix):
                    writer.writerow([label, *values])

    save_figures(histories, summaries, matrices, folder / "figures")


def save_figures(histories, summaries, matrices, folder):
    """Save the same plots that the notebook can display directly."""

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    population_size = int(summaries[0]["population"])

    figures = {
        "outcomes.png": plot_outcomes(histories, population_size),
        "strategy_frequencies.png": plot_strategy_frequencies(histories, population_size),
        "classical_payoffs.png": plot_classical_payoffs(matrices),
        "final_window.png": plot_final_window(summaries),
    }

    for name, figure in figures.items():

        figure.savefig(folder / name, dpi=180)
        plt.close(figure)

    for noise in NOISE_LEVELS:

        for kind, matrix in zip(("A", "H"), matrices[noise]):

            figure = plot_matrix(matrix, noise, kind)
            figure.savefig(folder / f"matrix_{kind}_noise_{noise:g}.png", dpi=150)
            plt.close(figure)


def update_readme_settings(runs, base_seed, settings):
    """Record the latest saved study's settings in the README."""

    path = Path(__file__).resolve().parent / "README.md"
    text = path.read_text(encoding="utf-8")
    section_heading = "## Settings for the saved study"
    next_paragraph = "Each individual's fitness"

    # Replace the settings section while keeping the rest of the README.

    before, rest = text.split(section_heading, 1)
    _, after = rest.split(next_paragraph, 1)

    lines = [f"Runs per noise level: **{runs}**. Total: **{runs * 3}**.", "",
             "| Setting | Value |", "|---|---:|"]

    for name, value in settings.items():
        lines.append(f"| {name.replace('_', ' ')} | {value} |")

    lines += ["", f"Base seed: `{base_seed}`. Each condition adds 10000; each run adds its index, starting at 0.",
              f"Bootstrap seed: `{base_seed + 900000}`. Bootstrap samples: `{BOOTSTRAP_SAMPLES}`.",
              "", f"Python: `{platform.python_version()}`."]

    for package in ("numpy", "deap", "matplotlib"):
        lines.append(f"{package}: `{importlib.metadata.version(package)}`.")

    path.write_text(before + section_heading + "\n\n" + "\n".join(lines) + "\n\n" + next_paragraph + after,
                    encoding="utf-8")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30, help="Runs per noise level")
    parser.add_argument("--seed", type=int, default=BASE_SEED, help="Base random seed")
    arguments = parser.parse_args()

    if arguments.runs < 2:
        parser.error("Use at least two runs per condition for confidence intervals.")

    plt.switch_backend("Agg")  # save figures without opening windows
    settings = DEFAULT_SETTINGS.copy()

    histories, summaries, matrices = run_study(arguments.runs, arguments.seed, settings)
    comparisons = make_comparisons(summaries, arguments.seed + 900000)

    save_results(histories, summaries, matrices, comparisons)
    update_readme_settings(arguments.runs, arguments.seed, settings)

    print(f"Saved results to {RESULTS_FOLDER}")
