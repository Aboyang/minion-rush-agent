"""Train a Q-table on MinionRush-v5 and save it to minion_q.npy. See README.md."""

import time

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import minion_rush                          # registers MinionRush-v5
from minion_rush import (LANES, N_ACTIONS, N_TRACK, decode_action, exact_q,
                         hits, index_to_track, missed_payloads, wasted_jumps)

N_STATES = LANES * N_TRACK
N_Q_CELLS = N_STATES * N_ACTIONS

GAMMA = 0.99
EPSILON, MIN_EPSILON, DECAY = 1.0, 0.15, 0.9999
EPISODES = 33_000_000
OUT = "minion_q.npy"


def train(gamma=GAMMA, episodes=EPISODES, seed=0, verbose=True):
    env = gym.make("MinionRush-v5")
    rng = np.random.default_rng(seed)

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    N = np.zeros_like(Q)
    epsilon = EPSILON
    returns = []

    for ep in range(episodes):
        state, _ = env.reset()
        done, total = False, 0.0

        while not done:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)

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
    return not unsafe_states(Q)


def unsafe_states(Q):
    """States where greedy(Q) crashes."""
    bad = []
    for s in range(Q.shape[0]):
        lane, index = divmod(s, N_TRACK)
        shift, height = decode_action(int(Q[s].argmax()))
        landing = int(np.clip(lane + shift, 0, LANES - 1))
        if hits(index_to_track(index)[landing], height):
            bad.append(s)
    return bad


if __name__ == "__main__":
    check_env(gym.make("MinionRush-v5").unwrapped)
    print("check_env: OK\n")

    t0 = time.time()
    Q, N, returns = train()
    print(f"\ntrained in {time.time() - t0:.1f}s")

    exact = exact_q()
    print(f"max |Q - exact| = {np.abs(Q - exact).max():.6f}")
    print(f"greedy policy safe in all {N_STATES} states: {greedy_policy_is_safe(Q)}")
    print(f"needless jumps/rolls in greedy policy: {len(wasted_jumps(Q))}")
    print(f"payloads left on the table: {len(missed_payloads(Q, exact))}")
    print(f"least-visited (state, action) cell: {int(N.min())} visits")

    np.save(OUT, Q)
    print(f"\nsaved -> {OUT}")
