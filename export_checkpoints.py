"""Snapshot the Q-table at key episodes and export it for the web visualiser.

train.py saves only the converged table. To *show* learning you need the table
mid-flight, so this re-runs the same loop and copies Q at chosen episodes.
The loop body is kept identical to train.py on purpose — if they drift, the
visualiser stops telling the truth about training.

Run:  python export_checkpoints.py   ->  web/checkpoints.json
"""

import json
import os

import gymnasium as gym
import numpy as np

import minion_rush                          # registers MinionRush-v5
from minion_rush import (ACTION_NAMES, CELL_CHARS, LANES, N_ACTIONS,
                         N_TRACK, TRACK_CODES, decode_action, exact_q)
from minion_rush import N_CELLS as N_CELL_TYPES   # local N_CELLS is Q-table size
from train import (DECAY, EPSILON, GAMMA, MIN_EPSILON, greedy_policy_is_safe,
                   unsafe_states)
from minion_rush import missed_payloads, wasted_jumps

# Dense early, sparse later: almost all the interesting change happens in the
# first few thousand episodes, because 98505 cells each need roughly one visit.
#
# v5 trimmed this list from 21 snapshots to 14. Each snapshot is a full
# 6567x15 q matrix AND a 6567x15 visit matrix, so the file grows with the
# table — at v4's 21 snapshots the export would now be ~14 MB for a page that
# loads it all up front. The dropped snapshots were the ones adjacent to a
# kept one (2, 10, 50, 250, 1000, 5000), where the table barely moves between
# them; the shape of the curve is in the decades, not the doublings.
CHECKPOINTS = [0, 1, 5, 25, 100, 500, 2500, 12_000, 50_000, 200_000,
               800_000, 3_000_000, 10_000_000, 33_000_000]

# The payload is one q matrix + one visit matrix per checkpoint. Two decimals
# is what the page actually renders, so storing more is paying bandwidth for
# digits nobody sees.
Q_DECIMALS = 2

OUT = os.path.join("web", "checkpoints.json")
N_STATES = LANES * N_TRACK
N_CELLS = N_STATES * N_ACTIONS
EXACT = exact_q()


def policy_stats(Q):
    """How good is greedy(Q), measured exactly rather than sampled.

    Every state is equally likely under reset(), so sweeping all 6567 states
    is the true expected reward — no need for a Monte-Carlo estimate.
    """
    rewards = [float(EXACT[s, int(np.argmax(Q[s]))]) for s in range(N_STATES)]
    unsafe = unsafe_states(Q)
    return {
        "avg_reward": float(np.mean(rewards)),
        "unsafe_states": unsafe,
        "safe": not unsafe,
        "wasted_jumps": len(wasted_jumps(Q)),
        "missed_payloads": len(missed_payloads(Q, EXACT)),
        "max_abs_err": float(np.abs(Q - EXACT).max()),
    }


def snapshot(Q, N, ep, epsilon):
    return {
        "episode": ep,
        "epsilon": round(float(epsilon), 4),
        "q": [[round(float(v), Q_DECIMALS) for v in row] for row in Q],
        "visits": [[int(v) for v in row] for row in N],
        "cells_seen": int((N > 0).sum()),
        **policy_stats(Q),
    }


def train_with_checkpoints(seed=0):
    env = gym.make("MinionRush-v5")
    rng = np.random.default_rng(seed)

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    N = np.zeros_like(Q)
    epsilon = EPSILON
    snaps, wanted = [], set(CHECKPOINTS)

    if 0 in wanted:                          # the untrained table: all zeros
        snaps.append(snapshot(Q, N, 0, epsilon))

    for ep in range(1, max(CHECKPOINTS) + 1):
        state, _ = env.reset()
        done = False
        while not done:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)

            N[state, action] += 1
            alpha = 1.0 / N[state, action]
            best_next = 0.0 if terminated else np.max(Q[next_state])
            Q[state, action] += alpha * (reward + GAMMA * best_next - Q[state, action])

            state = next_state
            done = terminated or truncated

        epsilon = max(MIN_EPSILON, epsilon * DECAY)

        if ep in wanted:
            snaps.append(snapshot(Q, N, ep, epsilon))
            print(f"  ep {ep:>6}  avg reward {snaps[-1]['avg_reward']:>7.2f}  "
                  f"cells seen {snaps[-1]['cells_seen']:>3}/{N_CELLS}  "
                  f"safe {snaps[-1]['safe']}")

    env.close()
    return Q, snaps


if __name__ == "__main__":
    print("training with checkpoints:")
    Q, snaps = train_with_checkpoints()

    # The final checkpoint must match what train.py produces, or the
    # visualiser is showing a different agent than the one you trained.
    assert greedy_policy_is_safe(Q), "final policy is not safe — refusing to export"
    print(f"\nmax |Q - exact| = {np.abs(Q - EXACT).max():.6f}")

    payload = {
        "action_names": ACTION_NAMES,
        "shift": [decode_action(a)[0] for a in range(N_ACTIONS)],
        "height": [decode_action(a)[1] for a in range(N_ACTIONS)],
        "lanes": LANES,
        "n_track": N_TRACK,
        "n_cells": N_CELL_TYPES,
        # The visualiser decodes a track itself, so it needs the cell alphabet
        # rather than a hardcoded copy of it.
        "cell_chars": [CELL_CHARS[c] for c in range(N_CELL_TYPES)],
        # v4's two impassable cells scattered the 8 unwinnable rows through
        # the code space, so a state's track index is a position in this
        # filtered list, not a base-10 number the page can decode on its own.
        # 2189 small ints — cheap next to the q matrices, and the alternative
        # is a second copy of the filter rule living in JavaScript.
        "track_codes": TRACK_CODES,
        "exact_q": [[float(v) for v in row] for row in EXACT],
        "checkpoints": snaps,
    }

    os.makedirs("web", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"saved {len(snaps)} checkpoints -> {OUT} "
          f"({os.path.getsize(OUT) / 1024:.0f} KB)")
