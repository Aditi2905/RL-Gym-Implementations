import math
import random
from collections import namedtuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

StepMemory = namedtuple("StepMemory", ["state", "action", "reward"])


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

# States: 0, 1, 2, 3
STATES = [0, 1, 2, 3]

# Actions:
#   0 = Left
#   1 = Right
ACTIONS = [0, 1]

STATE_TRANSITIONS = {
    0: {0: 0, 1: 1},
    1: {0: 2, 1: 0},
    2: {0: 1, 1: 3},
    3: {0: 3, 1: 3},
}

TERMINAL_STATE = 3

# Every time step before reaching the terminal state gives a reward of -1.
REWARD_PER_STEP = -1

START_STATE = 0


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

# Policy parameters.
#
# Random initialization:
# theta = [random.uniform(-1, 1), random.uniform(-1, 1)]
#
# Currently using a deliberately skewed initialization.
theta = [-2, 2]


# Feature vectors used to represent an action.
#
# Left  -> [0, 1]
# Right -> [1, 0]
LEFT_FEATURE = [0, 1]
RIGHT_FEATURE = [1, 0]


def get_features(action, state):
    """Return the feature vector for a given action and state."""
    if action == 0:
        return list(LEFT_FEATURE)

    return list(RIGHT_FEATURE)


def policy_probability(action, state, theta):
    """
    Calculate pi(action | state, theta).

    The policy uses a softmax over the preferences of the two actions.
    """
    numerator = 0
    denominator = 0

    for candidate_action in ACTIONS:
        features = get_features(candidate_action, state)

        preference = (
            theta[0] * features[0]
            + theta[1] * features[1]
        )

        probability_weight = math.exp(preference)
        denominator += probability_weight

        if candidate_action == action:
            numerator += probability_weight

    return numerator / denominator


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

TRAINING_EPOCHS = 500
LEARNING_RATE = 2e-4


for epoch in range(TRAINING_EPOCHS):
    print("Epoch:", epoch)

    state = START_STATE
    memory = []

    # Generate one complete episode.
    while state != TERMINAL_STATE:
        action_probabilities = [
            policy_probability(0, state, theta),
            policy_probability(1, state, theta),
        ]

        chosen_action = random.choices(
            ACTIONS,
            weights=action_probabilities,
            k=1,
        )[0]

        memory.append(
            StepMemory(
                state=state,
                action=chosen_action,
                reward=REWARD_PER_STEP,
            )
        )

        state = STATE_TRANSITIONS[state][chosen_action]

    # Calculate the return G backwards through the episode.
    G = 0

    for step in reversed(memory):
        G += step.reward

        # x(s, a) for the action that was actually taken.
        gradient = get_features(step.action, step.state)

        # Subtract the expected feature vector:
        #
        #   x(s, a) - sum_b pi(b | s) x(s, b)
        #
        # This is the policy-gradient term for the softmax policy.
        for action in ACTIONS:
            features = get_features(action, step.state)
            probability = policy_probability(
                action,
                step.state,
                theta,
            )

            expected_features = [
                feature * probability
                for feature in features
            ]

            gradient[0] -= expected_features[0]
            gradient[1] -= expected_features[1]

        # Scale the gradient by the return and learning rate.
        update = [
            value * G * LEARNING_RATE
            for value in gradient
        ]

        theta[0] += update[0]
        theta[1] += update[1]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

EVALUATION_EPISODES = 100

total_score = 0

for _ in range(EVALUATION_EPISODES):
    score = 0
    state = START_STATE

    while state != TERMINAL_STATE:
        action_probabilities = [
            policy_probability(0, state, theta),
            policy_probability(1, state, theta),
        ]

        chosen_action = random.choices(
            ACTIONS,
            weights=action_probabilities,
            k=1,
        )[0]

        state = STATE_TRANSITIONS[state][chosen_action]
        score += REWARD_PER_STEP

    total_score += score


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

average_reward = total_score / EVALUATION_EPISODES

print("Average Reward:", average_reward)
print("Theta:", theta)
print("Left probability:", policy_probability(0, START_STATE, theta))
print("Right probability:", policy_probability(1, START_STATE, theta))