"""Game rules and DEAP evolution for exercise 8."""

import random
from itertools import product

import numpy as np
from deap import algorithms, base, creator, tools


# Fixed settings. We change only the action noise in the experiment.

DEFAULT_SETTINGS = {
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

# All five-bit strategies, in binary order: 00000 to 11111.
# Bits: first action, after CC, after CD, after DC, after DD.

STRATEGIES = np.array(list(product((0, 1), repeat=5)), dtype=int)
BITS = []

for strategy in STRATEGIES:
    bits = "".join(str(bit) for bit in strategy)
    BITS.append(bits)

NAMES = {0: "ALLC", 5: "TFT", 6: "PAVLOV", 31: "ALLD"}
LABELS = []

for strategy_id, bits in enumerate(BITS):

    label = f"{strategy_id:02d}: {bits}"

    if strategy_id in NAMES:
        label += f' ("{NAMES[strategy_id]}")'

    LABELS.append(label)

# 0 = cooperate, 1 = defect. Own action comes first.

OUTCOMES = [(0, 0), (0, 1), (1, 0), (1, 1)]  # CC, CD, DC, DD
REWARDS = np.array([3.0, 0.0, 5.0, 1.0])  # points for player A
COOPERATION = np.array([1.0, 0.5, 0.5, 0.0])  # fraction of both actions that are C


def strategy_to_id(strategy):
    """Convert five bits into an ID. Example: 00101 becomes 5."""

    if len(strategy) != 5 or any(bit not in (0, 1) for bit in strategy):
        raise ValueError("A strategy needs five bits, each 0 or 1.")

    strategy_id = 0

    for bit in strategy:
        strategy_id = strategy_id * 2 + int(bit)

    return strategy_id


def outcome_probabilities(intended_a, intended_b, noise):
    """Calculate the chances of CC, CD, DC and DD after action errors."""

    if not 0 <= noise <= 1:
        raise ValueError("Noise must be between 0 and 1.")

    probabilities = []

    for actual_a, actual_b in OUTCOMES:

        # Matching the intention has chance 1-noise. Flipping it has chance noise.

        chance_a = 1 - noise if actual_a == intended_a else noise
        chance_b = 1 - noise if actual_b == intended_b else noise

        probabilities.append(chance_a * chance_b)  # independent errors

    return np.array(probabilities, dtype=float)


def expected_match(strategy_a, strategy_b, noise, rounds=150):
    """Calculate expected payoff and cooperation over one whole match."""

    if rounds < 1:
        raise ValueError("A match needs at least one round.")

    # First round: use the first bit of each strategy.

    probabilities = outcome_probabilities(strategy_a[0], strategy_b[0], noise)

    # Each row tells us the next outcome probabilities after CC, CD, DC or DD.

    next_round = np.zeros((4, 4))

    for outcome_id, (action_a, action_b) in enumerate(OUTCOMES):

        # 1 + 2*own + opponent selects the correct response bit.
        # Player B reads the same outcome from its own point of view.

        intended_a = strategy_a[1 + 2 * action_a + action_b]
        intended_b = strategy_b[1 + 2 * action_b + action_a]

        next_round[outcome_id] = outcome_probabilities(intended_a, intended_b, noise)

    # Count expected outcomes over all rounds, including round one.

    totals = probabilities.copy()

    for _ in range(1, rounds):

        probabilities = probabilities @ next_round  # combine the possible next outcomes
        totals += probabilities

    payoff = float(totals @ REWARDS / rounds)  # expected points per round
    cooperation = float(totals @ COOPERATION / rounds)  # expected fraction of cooperative actions

    return payoff, cooperation


def build_matrices(noise, rounds=150):
    """Calculate A and H for all 32 x 32 pairs at one noise level."""

    A = np.zeros((32, 32))
    H = np.zeros((32, 32))

    for row_id, strategy_a in enumerate(STRATEGIES):

        for column_id, strategy_b in enumerate(STRATEGIES):

            payoff, cooperation = expected_match(strategy_a, strategy_b, noise, rounds)

            A[row_id, column_id] = payoff
            H[row_id, column_id] = cooperation

    return A, H


def simulate_match(strategy_a, strategy_b, noise, rounds=150, seed=1, matches=1):
    """Actually play random matches. Useful for checking the expected values."""

    rng = np.random.default_rng(seed)  # reproducible action errors

    intended_a = np.full(matches, strategy_a[0], dtype=int)
    intended_b = np.full(matches, strategy_b[0], dtype=int)

    payoffs = np.zeros(matches)
    cooperation = np.zeros(matches)

    for _ in range(rounds):

        # Draw an error separately for each player's action.

        error_a = rng.random(matches) < noise
        error_b = rng.random(matches) < noise

        actual_a = np.where(error_a, 1 - intended_a, intended_a)
        actual_b = np.where(error_b, 1 - intended_b, intended_b)

        payoffs += REWARDS[2 * actual_a + actual_b]
        cooperation += (2 - actual_a - actual_b) / 2

        # Next intentions use the actual actions from this round.

        intended_a = strategy_a[1 + 2 * actual_a + actual_b]
        intended_b = strategy_b[1 + 2 * actual_b + actual_a]

    return payoffs / rounds, cooperation / rounds


def make_toolbox(settings):
    """Prepare the DEAP functions we will use."""

    # Create these classes once, so rerunning notebook cells is safe.

    if not hasattr(creator, "IPDFitnessMax"):

        creator.create("IPDFitnessMax", base.Fitness, weights=(1.0,))  # maximize payoff
        creator.create("IPDIndividual", list, fitness=creator.IPDFitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("bit", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.IPDIndividual, toolbox.bit, n=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("select", tools.selTournament, tournsize=settings["tournament_size"])
    toolbox.register("mate", tools.cxUniform, indpb=settings["crossover_bit_probability"])
    toolbox.register("mutate", tools.mutFlipBit, indpb=settings["mutation_bit_probability"])

    return toolbox


def evaluate_population(population, A, H):
    """Calculate everyone's fitness against the current opponents."""

    strategy_ids = []

    for individual in population:
        strategy_ids.append(strategy_to_id(individual))

    counts = np.bincount(strategy_ids, minlength=32)  # number of each type
    population_size = len(population)

    if population_size < 2:
        raise ValueError("At least two individuals are needed to play each other.")

    # Add the payoff against every opponent, remove self-play, then take the average.

    fitness_by_type = (A @ counts - A.diagonal()) / (population_size - 1)

    for individual, strategy_id in zip(population, strategy_ids):
        individual.fitness.values = (float(fitness_by_type[strategy_id]),)

    mean_payoff = float(counts @ fitness_by_type / population_size)

    # Apply the same opponent counting to cooperation.

    cooperation_by_type = (H @ counts - H.diagonal()) / (population_size - 1)
    mean_cooperation = float(counts @ cooperation_by_type / population_size)

    return counts, mean_payoff, mean_cooperation


def run_evolution(A, H, seed, settings=None):
    """Evolve one population and return its results in memory."""

    if settings is None:
        settings = DEFAULT_SETTINGS.copy()

    if not 0 <= settings["elites"] < settings["population"]:
        raise ValueError("Elites must be fewer than the population size.")

    if not 1 <= settings["final_window"] <= settings["generations"]:
        raise ValueError("The final window must fit inside the evolution.")

    random.seed(seed)  # reproducible DEAP choices

    toolbox = make_toolbox(settings)
    population = toolbox.population(n=settings["population"])
    history = []

    for generation in range(settings["generations"] + 1):

        # The opponents change, so everyone needs a fresh fitness value.

        counts, payoff, cooperation = evaluate_population(population, A, H)

        row = {
            "generation": generation,
            "payoff": payoff,
            "cooperation": cooperation,
        }

        for bits, count in zip(BITS, counts):
            row[f"count_{bits}"] = int(count)

        history.append(row)

        if generation == settings["generations"]:
            break

        # Copy the two best individuals into the next population.

        best = tools.selBest(population, settings["elites"])
        elites = [toolbox.clone(individual) for individual in best]

        # Choose the remaining parents, then cross and mutate their copies.

        number_of_children = settings["population"] - settings["elites"]
        parents = toolbox.select(population, number_of_children)

        children = algorithms.varAnd(
            parents,
            toolbox,
            cxpb=settings["crossover_probability"],
            mutpb=1.0,  # apply the 0.01-per-bit mutation procedure to every child
        )

        population = children + elites

    return history
