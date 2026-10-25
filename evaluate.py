"""Evaluate a trained Q-table against the random baseline.

Exploration is off here — epsilon was a training device, not part of the
final policy. Run:  python evaluate.py

Two distributions are reported, and the gap between them is the point:

  uniform    every one of the 2976 states equally likely. This is what the
             agent trained on, so it is the honest measure of the policy.
  clustered  game-shaped rows, with bananas in contiguous runs. Closer to what
             Minion Rush actually puts in front of you, but it cannot produce
             a split run like `b.b`, so it never visits a good share of the
             states. Good for watching, wrong for training.
"""

import gymnasium as gym
import numpy as np

import minion_rush                          # registers MinionRush-v4
from minion_rush import ACTION_NAMES, track_str

EPISODES = 2000
IN = "minion_q.npy"


def evaluate(Q=None, episodes=EPISODES, clustered=False):
    """Q=None plays uniformly at random, giving the baseline to beat."""
    env = gym.make("MinionRush-v4", clustered=clustered)
    rewards, swipes, jumps, rolls = [], [], [], []
    bananas, crashes = 0, 0

    for i in range(episodes):
        state, _ = env.reset(seed=i)        # seed per episode -> reproducible
        if Q is None:
            action = int(env.action_space.sample())
        else:
            action = int(np.argmax(Q[state]))
        _, reward, _, _, info = env.step(action)

        rewards.append(reward)
        swipes.append(info["swiped"])
        jumps.append(info["height"] == 1)
        rolls.append(info["height"] == 2)
        crashes += int(info["crashed"])
        bananas += int(info["banana"])

    env.close()
    return {
        "avg_reward": float(np.mean(rewards)),
        "crashes": crashes,
        "bananas": bananas,
        "episodes": episodes,
        "avg_swipes": float(np.mean(swipes)),
        "jump_rate": float(np.mean(jumps)),
        "roll_rate": float(np.mean(rolls)),
    }


HEADER = (f"{'':<20} {'avg reward':>10} {'crashes':>13} {'bananas':>13} "
          f"{'avg swipes':>11} {'jump rate':>10} {'roll rate':>10}")


def show(label, r):
    n = str(r["episodes"])
    print(f"{label:<20} {r['avg_reward']:>10.2f} "
          f"{r['crashes']:>7}/{n:<5} "
          f"{r['bananas']:>7}/{n:<5} "
          f"{r['avg_swipes']:>11.2f} "
          f"{r['jump_rate']:>10.2f} "
          f"{r['roll_rate']:>10.2f}")


def rollout(Q, seed=22, clustered=True):
    """Print one greedy episode so you can see the decision.

    Seed 22 happens to be the whole point of v4 in one row. The track is
    `^O_` and the minion starts in lane 2, on a crate:

        lane 0  ^  a beam          roll under it
        lane 1  O  a banana dangling over a PIT
        lane 2  _  a crate         <- he is here

    Staying put and jumping his own crate is safe and pays +1.5. Instead he
    swipes left over the hole and jumps:

        +2 clear  +2 banana  -1 swipe  -0.5 jump  =  +2.5

    He crossed a lane to leave the ground over a *hole* — a cell where
    touching down is death — because something was hanging above it. The
    banana axis supplies the motive and the pit supplies the risk, and the
    agent priced the pair correctly without being told the pit was new.
    """
    env = gym.make("MinionRush-v4", render_mode="ansi", clustered=clustered)
    state, _ = env.reset(seed=seed)
    lane, code = env.unwrapped.decode(state)

    print(f"\nstate {state}  (lane {lane}, track {track_str(code)})")
    print("before:")
    print(env.render(), end="")

    action = int(np.argmax(Q[state]))
    _, reward, _, _, info = env.step(action)
    print(f"action: {ACTION_NAMES[action]}   reward {reward:+.1f}   "
          f"swiped {info['swiped']}   height {info['height']}   "
          f"banana {info['banana']}   crashed {info['crashed']}")
    print("after:")
    print(env.render(), end="")
    env.close()


if __name__ == "__main__":
    Q = np.load(IN)

    print(HEADER)
    show("random", evaluate(None))
    show("trained", evaluate(Q))
    print()
    show("random (clustered)", evaluate(None, clustered=True))
    show("trained (clustered)", evaluate(Q, clustered=True))

    print("\nZero crashes is the signal, not the average reward: with a -50 "
          "penalty\na single crash per 100 episodes moves the mean by half a "
          "point while\nthe policy is still 99% correct.")
    print("\nThe banana column is the new one, and it has to be read against "
          "avg\nswipes. A policy that collects every banana is overpaying in "
          "swipes;\none that collects none is leaving +2s on the floor. The "
          "optimum is\nneither — it takes the cheap ones and walks past the "
          "two-lane ones.")

    rollout(Q)
