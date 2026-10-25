"""Print the whole Q-table — all 44640 cells — and check it against the exact
answer. The point of modelling this small is that the right answer is
verifiable by eye. Run:  python read_q.py
"""

import numpy as np

import minion_rush                          # registers MinionRush-v4
from minion_rush import (ACTION_NAMES, AIR_COST, BANANA, CLEAR, CRASH, LANES,
                         N_ACTIONS, N_TRACK, SWIPE_COST, exact_q,
                         missed_bananas, track_str, wasted_jumps)

IN = "minion_q.npy"
N_STATES = LANES * N_TRACK

CELL_LEGEND = [
    (".", "clear — safe at any height"),
    ("_", "LOW   — hits you at ground level, rolling included, so jump it"),
    ("^", "DUCK  — hits you standing or airborne, so roll under it"),
    ("o", "PIT   — a missing tile; you fall in at ground level, so jump it"),
    ("b", "banana on open ground — free +2, if you are already in the lane"),
    ("B", "banana on a crate — jump to take it"),
    ("v", "banana under a beam — roll to take it (`v` is `^` turned over)"),
    ("O", "banana dangling over a pit — jump to take it (`O` is `o` filled)"),
    ("=", "CHASM — a pit longer than the jump arc, so swipe around it"),
    ("#", "WALL  — hits you at any height, so swipe around it"),
]


def legend_rows():
    """Every value the table can hold, derived rather than hand-listed.

    v2 had 12 distinct values and they fitted in a comment. The banana axis
    doubles the safe half of that, so the legend is now computed from the same
    four constants the env charges — there is no second copy to drift.
    """
    rows = {}
    for crashed in (True, False):
        for swipes in (0, 1, 2):
            for airborne in (False, True):
                for banana in (False, True):
                    if crashed and banana:
                        continue            # you do not pocket it if you hit it
                    v = (-SWIPE_COST * swipes - (AIR_COST if airborne else 0.0)
                         + (CRASH if crashed else CLEAR)
                         + (BANANA if banana else 0.0))
                    arith = (f"{CRASH:+.0f}" if crashed else f"{CLEAR:+.0f}")
                    if banana:
                        arith += f" {BANANA:+.0f}b"
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
    print("\nhow to read a value:   clear/crash + banana - swipes - height")
    for value, arithmetic in legend_rows():
        print(f"  {value:>7.2f}  = {arithmetic}")
    print("\n  +4.00 is the new ceiling (banana underfoot, stand still) and "
          "+2.00 is\n  now ambiguous: it is both 'nothing here, stay put' and "
          "'banana two\n  lanes away, go and get it'. That collision is the "
          "break-even point.")


def print_ties(Q, limit=40):
    """A tie is either a symmetry worth keeping or a cost you forgot to charge.

    Both kinds are in here now. The old kind: standing in a WALL with a crate
    on one side and a beam on the other — jump right or roll left, same price.
    The new kind: a banana exactly two lanes away, where going for it and
    ignoring it are both worth +2.
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
    print(f"bananas left on the table (a strictly better grab existed): "
          f"{len(missed_bananas(Q, exact))}")

    err = np.abs(Q - exact).max()
    print(f"\nmax |Q - exact| = {err:.6f}")
    print("exact" if err < 1e-9 else "not yet converged — train longer or use a 1/n step size")
