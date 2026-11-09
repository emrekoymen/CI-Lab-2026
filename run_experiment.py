"""Run the repeated study and save results. Use: python run_experiment.py --runs 30"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

from ipd import SETTINGS, BITS, LABELS, NAMES, STRATEGIES, build_matrices, run_evolution


NOISE_LEVELS = [0.0, 0.01, 0.05]
PAIRS = [(0.0, 0.01), (0.0, 0.05), (0.01, 0.05)]
MEASURES = ["window_cooperation", "window_payoff"]
BASE_SEED = 21261112
BOOTSTRAP_SEED = BASE_SEED + 900000
BOOTSTRAP_SAMPLES = 5000
PILOT_RUNS = 10
RESULTS_FOLDER = Path(__file__).resolve().parent / "results"
COLORS = {0.0: "tab:blue", 0.01: "tab:orange", 0.05: "tab:green"}


def summarize_run(history, noise, run, seed):
    final = history[-1]
    window = history[-SETTINGS["final_window"]:]

    summary = {"noise": noise, "run": run, "seed": seed}
    summary.update(SETTINGS)
    summary["final_payoff"] = final["payoff"]
    summary["final_cooperation"] = final["cooperation"]
    summary["window_payoff"] = float(np.mean([row["payoff"] for row in window]))
    summary["window_cooperation"] = float(np.mean([row["cooperation"] for row in window]))

    for bits in BITS:
        summary[f"final_count_{bits}"] = final[f"count_{bits}"]
        summary[f"window_mean_count_{bits}"] = float(np.mean([row[f"count_{bits}"] for row in window]))

    return summary


def run_study(runs=30, matrices=None):
    if matrices is None:
        matrices = {}
    histories = {}
    summaries = []

    for condition, noise in enumerate(NOISE_LEVELS):
        if noise not in matrices:
            matrices[noise] = build_matrices(noise, SETTINGS["rounds"])
        A, H = matrices[noise]
        histories[noise] = []

        for run in range(runs):
            seed = BASE_SEED + condition * 10000 + run
            history = run_evolution(A, H, seed)
            histories[noise].append(history)
            summaries.append(summarize_run(history, noise, run + 1, seed))

        print(f"Noise {noise:g}: finished {runs} run(s).")

    return histories, summaries, matrices


def get_values(summaries, noise, column):
    values = []
    for row in summaries:
        if row["noise"] == noise:
            values.append(row[column])
    return values


def bootstrap_difference(reference, treatment, rng):
    """Median difference (treatment - reference) with a 95% bootstrap interval"""
    reference = np.array(reference)
    treatment = np.array(treatment)

    reference_samples = rng.choice(reference, size=(BOOTSTRAP_SAMPLES, len(reference)))
    treatment_samples = rng.choice(treatment, size=(BOOTSTRAP_SAMPLES, len(treatment)))
    reference_medians = np.median(reference_samples, axis=1)
    treatment_medians = np.median(treatment_samples, axis=1)
    differences = treatment_medians - reference_medians

    low, high = np.quantile(differences, [0.025, 0.975])
    effect = np.median(treatment) - np.median(reference)
    return float(effect), float(low), float(high)


def vargha_delaney_a12(reference, treatment):
    """Chance that a treatment run beats a reference run"""
    wins = 0.0
    for treatment_value in treatment:
        for reference_value in reference:
            if treatment_value > reference_value:
                wins += 1
            elif treatment_value == reference_value:
                wins += 0.5
    return wins / (len(reference) * len(treatment))


def add_holm(comparisons):
    comparisons_by_p = sorted(comparisons, key=lambda row: row["mann_whitney_p"])
    number_of_tests = len(comparisons)
    largest = 0.0
    for rank, row in enumerate(comparisons_by_p):
        adjusted = min(1.0, (number_of_tests - rank) * row["mann_whitney_p"])
        largest = max(largest, adjusted)
        row["holm_p"] = largest


def make_comparisons(summaries, max_runs=None):
    """Compare each pair of noise levels. max_runs keeps only the first runs"""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    comparisons = []

    for measure in MEASURES:
        for reference_noise, treatment_noise in PAIRS:
            reference = get_values(summaries, reference_noise, measure)
            treatment = get_values(summaries, treatment_noise, measure)
            if max_runs is not None:
                reference = reference[:max_runs]
                treatment = treatment[:max_runs]

            effect, low, high = bootstrap_difference(reference, treatment, rng)
            test = mannwhitneyu(treatment, reference, alternative="two-sided")

            comparisons.append({
                "measure": measure,
                "reference_noise": reference_noise,
                "treatment_noise": treatment_noise,
                "runs_per_group": len(treatment),
                "median_difference": effect,
                "ci95_low": low,
                "ci95_high": high,
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
                "mann_whitney_U": float(test.statistic),
                "mann_whitney_p": float(test.pvalue),
                "vargha_delaney_A12": vargha_delaney_a12(reference, treatment),
            })

    add_holm(comparisons)
    return comparisons


def pilot_spread(summaries):
    rows = []
    for noise in NOISE_LEVELS:
        for measure in MEASURES:
            values = np.array(get_values(summaries, noise, measure))

            for label, sample in [("pilot", values[:PILOT_RUNS]), ("full", values)]:
                q1, median, q3 = np.quantile(sample, [0.25, 0.5, 0.75])
                rows.append({
                    "noise": noise,
                    "measure": measure,
                    "sample": label,
                    "runs": len(sample),
                    "median": float(median),
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(q3 - q1),
                    "std": float(sample.std(ddof=1)),
                    "min": float(sample.min()),
                    "max": float(sample.max()),
                })
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        for key in row:
            row[key] = float(row[key])
    return rows


def write_matrix(path, matrix):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["row_strategy / column_opponent"] + LABELS)
        for label, values in zip(LABELS, matrix):
            writer.writerow([label] + list(values))


def read_matrix(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.reader(file))
    labels = rows[0][1:]
    values = []
    for row in rows[1:]:
        values.append([float(value) for value in row[1:]])
    return labels, np.array(values)


def load_results(folder=RESULTS_FOLDER):
    folder = Path(folder)
    summaries = read_csv(folder / "run_summary.csv")

    histories = {noise: [] for noise in NOISE_LEVELS}
    for row in summaries:
        path = folder / "histories" / f"noise_{row['noise']:g}_run_{int(row['run']):03d}.csv"
        histories[row["noise"]].append(read_csv(path))

    matrices = {}
    for noise in NOISE_LEVELS:
        labels, A = read_matrix(folder / f"A_noise_{noise:g}.csv")
        labels, H = read_matrix(folder / f"H_noise_{noise:g}.csv")
        matrices[noise] = (A, H)

    return histories, summaries, matrices


def save_results(histories, summaries, matrices, comparisons, pilot, pilot_comparisons):
    folder = RESULTS_FOLDER
    (folder / "histories").mkdir(parents=True, exist_ok=True)
    (folder / "figures").mkdir(exist_ok=True)

    for old_file in (folder / "histories").glob("noise_*_run_*.csv"):
        old_file.unlink()

    write_csv(folder / "run_summary.csv", summaries)
    write_csv(folder / "comparisons.csv", comparisons)
    write_csv(folder / "pilot_spread.csv", pilot)
    write_csv(folder / "pilot_comparisons.csv", pilot_comparisons)

    strategy_rows = []
    for i, genes in enumerate(STRATEGIES):
        strategy_rows.append({
            "id": i,
            "bits": BITS[i],
            "name": NAMES.get(i, ""),
            "label": LABELS[i],
            "first": int(genes[0]),
            "after_CC": int(genes[1]),
            "after_CD": int(genes[2]),
            "after_DC": int(genes[3]),
            "after_DD": int(genes[4]),
        })
    write_csv(folder / "strategies.csv", strategy_rows)

    for noise in NOISE_LEVELS:
        for run, history in enumerate(histories[noise], start=1):
            write_csv(folder / "histories" / f"noise_{noise:g}_run_{run:03d}.csv", history)
        A, H = matrices[noise]
        write_matrix(folder / f"A_noise_{noise:g}.csv", A)
        write_matrix(folder / f"H_noise_{noise:g}.csv", H)

    save_figures(histories, summaries, matrices, folder / "figures")


def plot_outcomes(histories):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    number_of_runs = len(histories[0.0])

    for noise in NOISE_LEVELS:
        for ax, measure in zip(axes, ["cooperation", "payoff"]):
            values = []
            for history in histories[noise]:
                values.append([row[measure] for row in history])
            values = np.array(values)
            if measure == "cooperation":
                values = values * 100

            generations = np.arange(values.shape[1])
            ax.plot(generations, np.median(values, axis=0), color=COLORS[noise], label=f"noise {noise:g}")
            if number_of_runs > 1:
                ax.fill_between(generations, values.min(axis=0), values.max(axis=0),
                                color=COLORS[noise], alpha=0.15)
            ax.set_xlabel("Generation")

    axes[0].set(ylabel="Cooperative actions (%)", ylim=(0, 100))
    axes[1].set(ylabel="Payoff per round", ylim=(0, 5))
    axes[0].legend()

    population_size = SETTINGS["population"]
    if number_of_runs == 1:
        fig.suptitle(f"Live experiment | 1 run per noise level | {population_size} individuals per run")
    else:
        fig.suptitle(f"{number_of_runs} runs per noise level | {population_size} individuals per run"
                     " | median and full range")
    return fig


def plot_strategy_frequencies(histories):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True, layout="constrained")
    tracked = [0, 5, 6, 7, 31]
    others = [i for i in range(32) if i not in tracked]

    for ax, noise in zip(axes, NOISE_LEVELS):
        counts = []
        for history in histories[noise]:
            run_counts = []
            for row in history:
                run_counts.append([row[f"count_{bits}"] for bits in BITS])
            counts.append(run_counts)
        percentages = np.mean(counts, axis=0) * 100 / SETTINGS["population"]
        generations = np.arange(len(percentages))

        for i in tracked:
            ax.plot(generations, percentages[:, i], label=LABELS[i])
        ax.plot(generations, percentages[:, others].sum(axis=1), color="gray", ls="--", label="Other strategies")
        ax.set(title=f"Noise {noise:g}", xlabel="Generation", ylim=(0, 100))

    axes[0].set_ylabel("Population (%)")
    axes[-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1))
    return fig


def plot_matrix(matrix, noise, kind):
    if kind == "A":
        title, maximum = "Expected payoff A", 5
    else:
        title, maximum = "Cooperation H", 1

    fig, ax = plt.subplots(figsize=(13, 12), layout="constrained")
    image = ax.imshow(matrix, vmin=0, vmax=maximum, cmap="viridis")
    ax.set_xticks(range(32), LABELS, rotation=90, fontsize=7)
    ax.set_yticks(range(32), LABELS, fontsize=7)
    ax.set(title=f"{title} | noise {noise:g}", xlabel="Column opponent", ylabel="Row strategy")
    fig.colorbar(image, ax=ax, shrink=0.75)
    return fig


def plot_classical_payoffs(matrices):
    selected = [0, 31, 5, 6, 7]
    names = [NAMES[i] for i in selected]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout="constrained")

    for ax, noise in zip(axes, NOISE_LEVELS):
        A = matrices[noise][0]
        values = np.zeros((5, 5))
        for row in range(5):
            for column in range(5):
                values[row, column] = A[selected[row], selected[column]]

        ax.imshow(values, vmin=0, vmax=5, cmap="Blues")
        ax.set_xticks(range(5), names, rotation=45, ha="right")
        ax.set_yticks(range(5), names)
        ax.set(title=f"Payoff A | noise {noise:g}", xlabel="Column opponent")

        for row in range(5):
            for column in range(5):
                value = values[row, column]
                color = "white" if value > 2.5 else "black"
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=color)

    return fig


def plot_final_window(summaries):
    """One dot per run, black bar = median."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")

    for ax, measure in zip(axes, MEASURES):
        for position, noise in enumerate(NOISE_LEVELS):
            values = np.array(get_values(summaries, noise, measure))
            if measure == "window_cooperation":
                values = values * 100

            x_positions = position + np.linspace(-0.12, 0.12, len(values))
            ax.scatter(x_positions, values, color=COLORS[noise], alpha=0.65, s=20)
            ax.plot([position - 0.2, position + 0.2], [np.median(values)] * 2, color="black")

        ax.set_xticks(range(3), [str(noise) for noise in NOISE_LEVELS])
        ax.set_xlabel("Action noise")

    axes[0].set(ylabel="Cooperative actions (%)", ylim=(0, 100))
    axes[1].set_ylabel("Payoff per round")
    fig.suptitle(f"Final {SETTINGS['final_window']} generations"
                 f" | each dot is one run of {SETTINGS['population']} individuals")
    return fig


def save_figures(histories, summaries, matrices, folder):
    figures = {
        "outcomes.png": plot_outcomes(histories),
        "strategy_frequencies.png": plot_strategy_frequencies(histories),
        "classical_payoffs.png": plot_classical_payoffs(matrices),
        "final_window.png": plot_final_window(summaries),
    }
    for name, fig in figures.items():
        fig.savefig(folder / name, dpi=180)
        plt.close(fig)

    for noise in NOISE_LEVELS:
        A, H = matrices[noise]
        for kind, matrix in [("A", A), ("H", H)]:
            fig = plot_matrix(matrix, noise, kind)
            fig.savefig(folder / f"matrix_{kind}_noise_{noise:g}.png", dpi=150)
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30, help="Runs per noise level (at least 2)")
    args = parser.parse_args()

    plt.switch_backend("Agg")

    histories, summaries, matrices = run_study(args.runs)
    comparisons = make_comparisons(summaries)
    pilot = pilot_spread(summaries)
    pilot_comparisons = make_comparisons(summaries, max_runs=PILOT_RUNS)

    save_results(histories, summaries, matrices, comparisons, pilot, pilot_comparisons)
    print(f"Saved results to {RESULTS_FOLDER}")
