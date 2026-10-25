"""Evaluate a trained Q-table against the random baseline.

Exploration is off here — epsilon was a training device, not part of the
final policy. Run:  python evaluate.py

Two distributions are reported, and the gap between them is the point:

  uniform    every one of the 6567 states equally likely. This is what the
             agent trained on, so it is the honest measure of the policy.
  clustered  game-shaped rows: bananas in contiguous runs, at most one
             bystander minion standing on his own. Closer to what Minion Rush
             actually puts in front of you, but it cannot produce a split run
             like `b.b`, so it never visits a good share of the states. Good
             for watching, wrong for training.
"""

import gymnasium as gym
import numpy as np

import minion_rush                          # registers MinionRush-v5
from minion_rush import ACTION_NAMES, track_str

EPISODES = 2000
IN = "minion_q.npy"


def evaluate(Q=None, episodes=EPISODES, clustered=False):
    """Q=None plays uniformly at random, giving the baseline to beat."""
    env = gym.make("MinionRush-v5", clustered=clustered)
    rewards, swipes, jumps, rolls = [], [], [], []
    bananas, minions, crashes = 0, 0, 0

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
        minions += int(info["minion"])

    env.close()
    return {
        "avg_reward": float(np.mean(rewards)),
        "crashes": crashes,
        "bananas": bananas,
        "minions": minions,
        "episodes": episodes,
        "avg_swipes": float(np.mean(swipes)),
        "jump_rate": float(np.mean(jumps)),
        "roll_rate": float(np.mean(rolls)),
    }


HEADER = (f"{'':<20} {'avg reward':>10} {'crashes':>13} {'bananas':>13} "
          f"{'minions':>13} {'avg swipes':>11} {'jump rate':>10} {'roll rate':>10}")


def show(label, r):
    n = str(r["episodes"])
    print(f"{label:<20} {r['avg_reward']:>10.2f} "
          f"{r['crashes']:>7}/{n:<5} "
          f"{r['bananas']:>7}/{n:<5} "
          f"{r['minions']:>7}/{n:<5} "
          f"{r['avg_swipes']:>11.2f} "
          f"{r['jump_rate']:>10.2f} "
          f"{r['roll_rate']:>10.2f}")


def rollout(Q, seed=536, clustered=True):
    """Print one greedy episode so you can see the decision.

    Seed 536 is the whole point of v5 in one row. The track is `M^b` and the
    minion starts in lane 2:

        lane 0  M  a BYSTANDER standing on a crate — worth +5
        lane 1  ^  a beam
        lane 2  b  a banana on clear road           <- he is here

    The +5 is the biggest number on the road and it is reachable: swipe two
    lanes and jump the crate he is standing on. The agent does not go.

        go for the minion   +2 clear  +5 minion  -2 swipe  -0.5 jump  = +4.5
        stay on the banana  +2 clear  +3 banana                       = +5.0

    **He walks past a +5 to take a +3, and he is right.** Two lanes of swipe
    plus a posture costs 2.5, and the gap between the payloads is only 2.0.
    This is the row the whole version was built to produce: before v5 there
    was exactly one kind of prize, so "which one" was never a question, and
    greed and optimality never came apart.

    Seed 44 is its mirror — `m=B` from lane 2, where he crosses the entire
    track *past a chasm* for a minion on open ground (+5.0) rather than jump
    the banana-crate under his feet (+4.5). Same arithmetic, opposite answer,
    0.5 apart.
    """
    env = gym.make("MinionRush-v5", render_mode="ansi", clustered=clustered)
    state, _ = env.reset(seed=seed)
    lane, code = env.unwrapped.decode(state)

    print(f"\nstate {state}  (lane {lane}, track {track_str(code)})")
    print("before:")
    print(env.render(), end="")

    action = int(np.argmax(Q[state]))
    _, reward, _, _, info = env.step(action)
    print(f"action: {ACTION_NAMES[action]}   reward {reward:+.1f}   "
          f"swiped {info['swiped']}   height {info['height']}   "
          f"banana {info['banana']}   minion {info['minion']}   "
          f"crashed {info['crashed']}")
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
    print("\nRead the banana and minion columns against avg swipes, and "
          "against each\nother. A minion is worth +5 to a banana's +3, so the "
          "policy should take\nminions at a higher rate — but not at any "
          "price: 2.5 of swipe-and-posture\nis more than the 2.0 that "
          "separates the two payloads, so the furthest,\nawkwardest minions "
          "are correctly left standing.")

    rollout(Q)
