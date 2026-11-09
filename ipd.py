"""Game rules and DEAP evolution for exercise 8."""

import random
from itertools import product

import numpy as np
from deap import algorithms, base, creator, tools


SETTINGS = {
    "population": 100,
    "generations": 500,
    "rounds": 150,
    "crossover_probability": 0.7,
    "crossover_bit_probability": 0.5,
    "mutation_bit_probability": 0.01,
    "tournament_size": 3,
    "elites": 2,
    "final_window": 100,
}

#0 = cooperate, 1 = defect
STRATEGIES = np.array(list(product([0, 1], repeat=5)))

BITS = []
for strategy in STRATEGIES:
    BITS.append("".join(str(bit) for bit in strategy))

NAMES = {0: "ALLC", 5: "TFT", 6: "PAVLOV", 7: "GRIM", 31: "ALLD"}

LABELS = []
for strategy_id in range(32):
    label = f"{strategy_id:02d}: {BITS[strategy_id]}"
    if strategy_id in NAMES:
        label += f' ("{NAMES[strategy_id]}")'
    LABELS.append(label)

PAYOFF = {(0, 0): 3, (0, 1): 0, (1, 0): 5, (1, 1): 1}
OUTCOMES = list(PAYOFF)


def strategy_to_id(strategy):
    strategy_id = 0
    for bit in strategy:
        strategy_id = 2 * strategy_id + int(bit)
    return strategy_id


def outcome_probabilities(intended_a, intended_b, noise):
    probabilities = []
    for actual_a, actual_b in OUTCOMES:
        chance_a = 1 - noise if actual_a == intended_a else noise
        chance_b = 1 - noise if actual_b == intended_b else noise
        probabilities.append(chance_a * chance_b)
    return np.array(probabilities, dtype=float)


def expected_match(strategy_a, strategy_b, noise, rounds=150):
    """Expected payoff (for player A) and cooperation per round."""
    probabilities = outcome_probabilities(strategy_a[0], strategy_b[0], noise)

    #next_round[i] = chances of each outcome after outcome i
    next_round = np.zeros((4, 4))
    for i in range(4):
        action_a, action_b = OUTCOMES[i]
        intended_a = strategy_a[1 + 2 * action_a + action_b]
        intended_b = strategy_b[1 + 2 * action_b + action_a]
        next_round[i] = outcome_probabilities(intended_a, intended_b, noise)

    totals = probabilities.copy()
    for round_number in range(1, rounds):
        probabilities = probabilities @ next_round
        totals += probabilities

    #outcomes in the order CC, CD, DC, DD
    payoff = totals @ np.array([3, 0, 5, 1]) / rounds
    cooperation = totals @ np.array([1, 0.5, 0.5, 0]) / rounds
    return float(payoff), float(cooperation)


def build_matrices(noise, rounds=150):
    A = np.zeros((32, 32))
    H = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            A[i, j], H[i, j] = expected_match(STRATEGIES[i], STRATEGIES[j], noise, rounds)
    return A, H


def simulate_match(strategy_a, strategy_b, noise, rounds=150, seed=1, matches=1):
    #play real random matches, to check expected_match
    rng = np.random.default_rng(seed)

    intended_a = np.full(matches, strategy_a[0])
    intended_b = np.full(matches, strategy_b[0])
    payoffs = np.zeros(matches)
    cooperation = np.zeros(matches)

    for round_number in range(rounds):
        error_a = rng.random(matches) < noise
        error_b = rng.random(matches) < noise

        for match in range(matches):
            actual_a = intended_a[match]
            actual_b = intended_b[match]
            if error_a[match]:
                actual_a = 1 - actual_a
            if error_b[match]:
                actual_b = 1 - actual_b

            payoffs[match] += PAYOFF[(actual_a, actual_b)]
            cooperation[match] += (2 - actual_a - actual_b) / 2

            intended_a[match] = strategy_a[1 + 2 * actual_a + actual_b]
            intended_b[match] = strategy_b[1 + 2 * actual_b + actual_a]

    return payoffs / rounds, cooperation / rounds


def make_toolbox():
    if not hasattr(creator, "IPDFitnessMax"):
        creator.create("IPDFitnessMax", base.Fitness, weights=(1.0,))
        creator.create("IPDIndividual", list, fitness=creator.IPDFitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("bit", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.IPDIndividual, toolbox.bit, n=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selTournament, tournsize=SETTINGS["tournament_size"])
    toolbox.register("mate", tools.cxUniform, indpb=SETTINGS["crossover_bit_probability"])
    toolbox.register("mutate", tools.mutFlipBit, indpb=SETTINGS["mutation_bit_probability"])
    return toolbox


def evaluate_population(population, A, H):
    """Fitness = average payoff against the other individuals (no self-play)."""
    ids = [strategy_to_id(individual) for individual in population]
    counts = np.bincount(ids, minlength=32)
    population_size = len(population)

    fitness_by_type = (A @ counts - A.diagonal()) / (population_size - 1)
    cooperation_by_type = (H @ counts - H.diagonal()) / (population_size - 1)

    for individual, strategy_id in zip(population, ids):
        individual.fitness.values = (float(fitness_by_type[strategy_id]),)

    mean_payoff = float(counts @ fitness_by_type / population_size)
    mean_cooperation = float(counts @ cooperation_by_type / population_size)
    return counts, mean_payoff, mean_cooperation


def run_evolution(A, H, seed):
    random.seed(seed)
    toolbox = make_toolbox()
    population = toolbox.population(n=SETTINGS["population"])
    history = []

    for generation in range(SETTINGS["generations"] + 1):
        counts, payoff, cooperation = evaluate_population(population, A, H)

        row = {"generation": generation, "payoff": payoff, "cooperation": cooperation}
        for bits, count in zip(BITS, counts):
            row[f"count_{bits}"] = int(count)
        history.append(row)

        if generation == SETTINGS["generations"]:
            break

        best = tools.selBest(population, SETTINGS["elites"])
        elites = [toolbox.clone(individual) for individual in best]
        parents = toolbox.select(population, SETTINGS["population"] - SETTINGS["elites"])
        #mutpb=1: every child goes through mutFlipBit, which flips each bit with 1% chance
        children = algorithms.varAnd(parents, toolbox, cxpb=SETTINGS["crossover_probability"], mutpb=1.0)
        population = children + elites

    return history
