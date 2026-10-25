"""Train a Q-table on MinionRush-v4 and save it to minion_q.npy.

This is the Taxi-v4 Q-learning loop, unchanged except for the gym.make string
and a 1/n step size (see below). Run:  python train.py
"""

import time

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import minion_rush                          # registers MinionRush-v4
from minion_rush import (LANES, N_ACTIONS, N_TRACK, decode_action, exact_q,
                         hits, index_to_track, missed_bananas, wasted_jumps)

N_STATES = LANES * N_TRACK                   # 2976
N_Q_CELLS = N_STATES * N_ACTIONS             # 44640

GAMMA = 0.99
# A 1/n step size is only an exact average once every cell has been visited,
# and a greedy episode revisits the same argmax cells forever. So the floor on
# exploration — not the episode count alone — is what guarantees coverage.
# The rarest-cell visit count is roughly `episodes * MIN_EPSILON / N_Q_CELLS`,
# and v4's pit and chasm took the table from 15390 cells to 44640, a 2.9x
# jump. Holding that estimate at v3's ~49 means scaling the episode budget by
# the same factor: 5M * 2.9 = 14.5M, rounded to 15M. MIN_EPSILON stays at
# 0.15 — raising it instead would buy coverage by making the behaviour policy
# worse, and the episode count is the cheaper knob.
EPSILON, MIN_EPSILON, DECAY = 1.0, 0.15, 0.9999
EPISODES = 15_000_000
OUT = "minion_q.npy"


def train(gamma=GAMMA, episodes=EPISODES, seed=0, verbose=True):
    env = gym.make("MinionRush-v4")
    rng = np.random.default_rng(seed)

    Q = np.zeros((env.observation_space.n, env.action_space.n))   # (2976, 15)
    N = np.zeros_like(Q)                                          # visit counts
    epsilon = EPSILON
    returns = []

    for ep in range(episodes):
        state, _ = env.reset()
        done, total = False, 0.0

        while not done:
            # --- act ---
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            # --- step ---
            next_state, reward, terminated, truncated, _ = env.step(action)

            # --- learn ---
            # A 1/n step size makes the update an exact running average. Valid
            # here only because episodes are one step long, so there is no
            # bootstrapping and the target is a stationary immediate reward.
            N[state, action] += 1
            alpha = 1.0 / N[state, action]
            best_next = 0.0 if terminated else np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
            total += reward
            done = terminated or truncated

        epsilon = max(MIN_EPSILON, epsilon * DECAY)
        returns.append(total)

        if verbose and (ep + 1) % 10_000 == 0:
            window = np.mean(returns[-10_000:])
            print(f"ep {ep + 1:>6}  avg return: {window:>7.2f}  epsilon: {epsilon:.3f}")

    env.close()
    return Q, N, returns


def greedy_policy_is_safe(Q):
    """Does argmax survive every one of the 2976 states?"""
    return not unsafe_states(Q)


def unsafe_states(Q):
    bad = []
    for s in range(Q.shape[0]):
        lane, index = divmod(s, N_TRACK)
        shift, height = decode_action(int(Q[s].argmax()))
        landing = int(np.clip(lane + shift, 0, LANES - 1))
        if hits(index_to_track(index)[landing], height):
            bad.append(s)
    return bad


if __name__ == "__main__":
    # Validate the contract before spending any time on hyperparameters.
    check_env(gym.make("MinionRush-v4").unwrapped)
    print("check_env: OK\n")

    t0 = time.time()
    Q, N, returns = train()
    print(f"\ntrained in {time.time() - t0:.1f}s")

    exact = exact_q()
    print(f"max |Q - exact| = {np.abs(Q - exact).max():.6f}")
    print(f"greedy policy safe in all {N_STATES} states: {greedy_policy_is_safe(Q)}")
    print(f"needless jumps/rolls in greedy policy: {len(wasted_jumps(Q))}")
    print(f"bananas left on the table: {len(missed_bananas(Q, exact))}")
    print(f"least-visited (state, action) cell: {int(N.min())} visits")

    np.save(OUT, Q)
    print(f"\nsaved -> {OUT}")
