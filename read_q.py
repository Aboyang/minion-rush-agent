"""Print the whole Q-table — all 98505 cells — and check it against the exact
answer. The point of modelling this small is that the right answer is
verifiable by eye. Run:  python read_q.py
"""

import numpy as np

import minion_rush                          # registers MinionRush-v5
from minion_rush import (ACTION_NAMES, AIR_COST, BANANA, CLEAR, CRASH, LANES,
                         MINION, N_ACTIONS, N_TRACK, SWIPE_COST, exact_q,
                         missed_payloads, track_str, wasted_jumps)

IN = "minion_q.npy"
N_STATES = LANES * N_TRACK

CELL_LEGEND = [
    (".", "clear — safe at any height"),
    ("_", "LOW   — hits you at ground level, rolling included, so jump it"),
    ("^", "DUCK  — hits you standing or airborne, so roll under it"),
    ("o", "PIT   — a missing tile; you fall in at ground level, so jump it"),
    ("b", "banana on open ground — free +3, if you are already in the lane"),
    ("B", "banana on a crate — jump to take it"),
    ("v", "banana under a beam — roll to take it (`v` is `^` turned over)"),
    ("O", "banana dangling over a pit — jump to take it (`O` is `o` filled)"),
    ("m", "BYSTANDER on open ground — run him over, +5"),
    ("M", "bystander standing on a crate — jump, and take him with you"),
    ("w", "bystander crouched under a beam — roll (`w` is `m` turned over)"),
    ("=", "CHASM — a pit longer than the jump arc, so swipe around it"),
    ("#", "WALL  — hits you at any height, so swipe around it"),
]


def legend_rows():
    """Every value the table can hold, derived rather than hand-listed.

    v2 had 12 distinct values and they fitted in a comment. The payload axis
    has since tripled the safe half of that, so the legend is computed from
    the same constants the env charges — there is no second copy to drift.

    Note what the derivation HIDES: two payloads at two prices put the same
    number in the table for different reasons, and `setdefault` keeps only
    the first arithmetic that reached it. +5.00 prints as "+2 +3b" but is
    equally "minion two lanes away" — so a value here is no longer a
    fingerprint of what happened. That collision is the competition v5 added,
    and the note under the table is where it gets said out loud.
    """
    rows = {}
    payloads = [(0.0, ""), (BANANA, "b"), (MINION, "m")]
    for crashed in (True, False):
        for swipes in (0, 1, 2):
            for airborne in (False, True):
                for pay, tag in payloads:
                    if crashed and pay:
                        continue            # you do not collect it if you hit it
                    v = (-SWIPE_COST * swipes - (AIR_COST if airborne else 0.0)
                         + (CRASH if crashed else CLEAR) + pay)
                    arith = (f"{CRASH:+.0f}" if crashed else f"{CLEAR:+.0f}")
                    if pay:
                        arith += f" {pay:+.0f}{tag}"
                    arith += f" -{swipes} -{AIR_COST if airborne else 0.0}"
                    rows.setdefault(round(v, 2), arith)
    return sorted(rows.items(), reverse=True)


def print_table(Q):
    header = f"{'s':>4} {'lane':>5} {'track':>7} "
    header += " ".join(f"{n:>7}" for n in ACTION_NAMES) + "   best"
    print(header)
    print("-" * len(header))

    for s in range(N_STATES):
        lane, code = divmod(s, N_TRACK)
        row = " ".join(f"{v:>7.2f}" for v in Q[s])
        best = ACTION_NAMES[int(Q[s].argmax())]
        print(f"{s:>4} {lane:>5} {track_str(code):>7} {row}   {best:<6}")


def print_legend():
    print("\nhow to read a track:")
    for char, meaning in CELL_LEGEND:
        print(f"  {char}  {meaning}")
    print("\nhow to read a value:   clear/crash + payload - swipes - height")
    for value, arithmetic in legend_rows():
        print(f"  {value:>7.2f}  = {arithmetic}")
    print("\n  +7.00 is the new ceiling (minion underfoot, stand still). The "
          "value to\n  watch is +5.00: it is both 'banana underfoot, stay "
          "put' AND 'minion\n  two lanes away, go and get him'. Where a row "
          "offers both at once, the\n  tie is real and the agent is free to "
          "take either — and at +4.50, a\n  minion two lanes out *on an "
          "obstacle*, it should take neither.")


def print_ties(Q, limit=40):
    """A tie is either a symmetry worth keeping or a cost you forgot to charge.

    Both kinds are in here now. The old kind: standing in a WALL with a crate
    on one side and a beam on the other — jump right or roll left, same price.
    The v3 kind: a banana exactly two lanes away, where going for it and
    ignoring it are both worth the same.

    The v5 kind, and the only one that is a choice between two *prizes*: a
    banana underfoot and a minion two lanes off, both +5.00. Every earlier tie
    was the agent being indifferent about effort; this one is it being
    indifferent about which reward to chase.
    """
    ties = []
    for s in range(N_STATES):
        best = Q[s].max()
        tied = [ACTION_NAMES[a] for a in range(N_ACTIONS)
                if abs(Q[s, a] - best) < 1e-9]
        if len(tied) > 1:
            ties.append((s, tied))

    print(f"\nties among optimal actions: {len(ties)} states"
          f"{f' (showing first {limit})' if len(ties) > limit else ''}")
    for s, tied in ties[:limit]:
        lane, code = divmod(s, N_TRACK)
        print(f"  state {s:>4}  lane {lane}  track {track_str(code)}  ->  {tied}")


def print_policy_summary(Q):
    print("\npolicy by action:")
    chosen = Q.argmax(axis=1)
    for a, name in enumerate(ACTION_NAMES):
        states = [s for s in range(N_STATES) if chosen[s] == a]
        head = ", ".join(str(s) for s in states[:12])
        tail = ", ..." if len(states) > 12 else ""
        print(f"  {name:<6} chosen in {len(states):>4} states  [{head}{tail}]")


if __name__ == "__main__":
    Q = np.load(IN)
    print_table(Q)
    print_legend()
    print_ties(Q)
    print_policy_summary(Q)

    exact = exact_q()
    print(f"\nneedless jumps/rolls (left the ground onto a clear cell): {len(wasted_jumps(Q))}")
    print(f"payloads left on the table (a strictly better grab existed): "
          f"{len(missed_payloads(Q, exact))}")

    err = np.abs(Q - exact).max()
    print(f"\nmax |Q - exact| = {err:.6f}")
    print("exact" if err < 1e-9 else "not yet converged — train longer or use a 1/n step size")
