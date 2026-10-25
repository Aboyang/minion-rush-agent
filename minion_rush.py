"""MinionRush-v4 — a 3-lane, two-axis, one-decision dodge-and-collect environment.

State  : lane * 992 + track_index     -> Discrete(2976)
Action : height * 5 + shift_index     -> Discrete(15)
Reward : +2 clear, +3 banana, -50 crash, -1 per lane the swipe intends to
         cross, -0.5 for leaving the ground (jump or roll)
Episode: exactly one step, always terminated

Two axes of movement. x is the lane. y is the height the minion takes through
the row: GROUND, AIR or ROLL. Six obstacles, each with exactly one answer:

    CLEAR  .   nothing there
    LOW    _   a short crate — hits you at ground level, so jump it
    DUCK   ^   an overhead beam — hits you standing or airborne, so roll under
    PIT    o   a missing floor tile — you fall in at ground level, so jump it
    CHASM  =   a pit longer than the jump arc — unclearable, so go around it
    WALL   #   a tall wall — hits you at any height, so go around it

New in v4: the floor can be missing. PIT and CHASM are the same hazard at two
sizes, and the size *is* the mechanic — one you clear by leaving the ground,
the other you can only walk around.

Neither is a new hit mask. PIT's is `(True, False, True)`, identical to LOW
bit for bit: a crate stops you at ground level and a roll is still ground
level; a hole swallows you at ground level and a roll still rolls you in.
CHASM's is `(True, True, True)`, identical to WALL. So v4 teaches the agent no
answer it did not already know — it only makes it learn those answers twice,
under two more symbols, which is a real cost (|S| 1026 -> 2976) knowingly paid
for a road that looks like Minion Rush.

**A chasm is wide along the row, never across rows.** This is the constraint
the whole design hangs on. A pit that spanned several rows would be a hazard
the agent could not see the end of, and the episode is exactly one row — it
has no next row to fall into. So a chasm is drawn long in z but lives entirely
inside one row's slot, with clear road at both ends, exactly the way a banana
bunch is a *line* of bananas that still belongs to a single cell. "Too wide to
jump" is then a fact about this row alone, visible in this row's state.

A lane can also hold a **banana**, worth +3 on top of the clear — priced so a
two-lane detour to collect one still beats staying on clear road.
A banana is not another obstacle — it is a second axis on the cell, because a
banana in an otherwise-empty lane is not a decision. The episode is one row
long, so the only way "going for the banana puts me in trouble" can exist at
all is if a banana can sit *on* an obstacle:

    BCLEAR b   a banana on open ground      — free money
    BLOW   B   a banana on top of a crate   — jump to take it
    BDUCK  v   a banana under a beam        — roll to take it
    BPIT   O   a banana dangling over a pit — jump to take it

(`v` is `^` turned over: the beam is still there, you go under it. `O` is `o`
with something in it: the hole is still there, the banana hangs above it.)

Bananas never sit in a WALL or a CHASM — you cannot reach into either — so the
lane alphabet is 4 passable cells x 2 (banana or not) + 2 impassable = 10.

### The free exclusion finally cost something

v1 through v3 had exactly one unsurvivable cell, so exactly one unsurvivable
row — all-WALL — and by numbering WALL last that row was exactly the top code
and could be dropped with `N_CELLS ** LANES - 1`. No renumbering, codes stayed
contiguous, and the game stayed always winnable for free.

CHASM ends that. With two impassable cells there are `2 ** LANES = 8`
unwinnable rows — `===`, `==#`, `=#=`, ... , `###` — scattered through the code
space rather than sitting at the top of it. There is no numbering that gathers
them, so the trick does not survive and the exclusion becomes an actual
filter: enumerate the 1000 raw base-10 codes, keep the 992 that have at least
one passable lane, and let the *state* index that list.

    N_TRACK = 10**3 - 8 = 992          valid track indices 0..991
    |S|     = LANES * N_TRACK = 2976   states 0..2975

So there are now two numbers in play and they must not be confused:

    raw code      a base-10 number 0..999, what the cells literally spell
    track index   0..991, position in TRACK_CODES, what the state carries

`code_to_track` still speaks raw codes. Everything that starts from a state
goes through `index_to_track`.

Importing this module registers MinionRush-v4.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

LANES = 3

# Cell values. The four passable obstacles come first, then their banana
# twins, then the two impassable cells. The ordering no longer buys the free
# top-code exclusion it did in v3 — see the module docstring — but keeping the
# impassable pair adjacent at the top still makes `cell >= CHASM` the test for
# "nothing gets you through this lane".
CLEAR_CELL, LOW, DUCK, PIT = 0, 1, 2, 3
BCLEAR, BLOW, BDUCK, BPIT = 4, 5, 6, 7
MCLEAR, MLOW, MDUCK = 8, 9, 10
CHASM, WALL = 11, 12
N_CELLS = 13

# The y-axis — an index into a list of heights, not a flag.
GROUND, AIR, ROLL = 0, 1, 2
N_HEIGHTS = 3
HEIGHT_NAMES = ["ground", "air", "roll"]

SHIFT = {0: -2, 1: -1, 2: 0, 3: +1, 4: +2}
N_ACTIONS = N_HEIGHTS * len(SHIFT)
CLEAR, CRASH, SWIPE_COST, AIR_COST = 2.0, -50.0, 1.0, 0.5
# Worth more than a cleared row on purpose. At 2.0 a banana only paid for a
# one-lane detour (BANANA > SWIPE_COST*lanes + HEIGHT_COST, so > 1.5), and the
# agent correctly walked past anything two lanes out. 3.0 clears the two-lane
# bar (> 2.5) while keeping the margin thin, so a detour onto a banana is worth
# it but never worth risking a crash for.
BANANA = 3.0
# A bystander minion, knocked over rather than collected — but priced on the
# same axis and paid out by the same rule, because mechanically that is what
# it is: a payload riding on a cell, taken at any height, forfeited on a
# crash. 5.0 is chosen against BANANA rather than in isolation:
#
#   minion underfoot, clear           +2 +5         = +7.0   <- new ceiling
#   minion one lane over, on a crate  +2 +5 -1 -0.5 = +5.5
#   banana underfoot, clear           +2 +3         = +5.0   <- ties with
#   minion two lanes over, clear      +2 +5 -2      = +5.0      this
#   minion two lanes over, on a beam  +2 +5 -2 -0.5 = +4.5   <- loses to it
#
# So a minion is worth crossing the whole track for *unless* doing so also
# costs a posture and there is a banana already underfoot. That is the first
# row in the project where two payloads compete, and the only reason it
# exists is that the two prices are close enough to be compared.
MINION = 5.0

# Cost of each height. Jump and roll cost the same, so the choice between them
# is decided purely by the obstacle and never by a tie-break on price.
HEIGHT_COST = (0.0, AIR_COST, AIR_COST)

ACTION_NAMES = ["L2", "L1", "stay", "R1", "R2",
                "L2+J", "L1+J", "jump", "R1+J", "R2+J",
                "L2+R", "L1+R", "roll", "R1+R", "R2+R"]
# `w` is `m` turned over, the same way `v` is `^` turned over: the beam is
# still there and the bystander is under it, crouching.
CELL_CHARS = {CLEAR_CELL: ".", LOW: "_", DUCK: "^", PIT: "o",
              BCLEAR: "b", BLOW: "B", BDUCK: "v", BPIT: "O",
              MCLEAR: "m", MLOW: "M", MDUCK: "w",
              CHASM: "=", WALL: "#"}

# Which heights each cell hits, indexed [cell][height]. A banana changes the
# payoff of a cell, never its geometry: each banana twin hits exactly like the
# obstacle it is sitting on. PIT duplicates LOW and CHASM duplicates WALL, on
# purpose and at a known price — see the module docstring.
HITS = {CLEAR_CELL: (False, False, False),
        LOW: (True, False, True),           # a crate: only a jump clears it
        DUCK: (True, True, False),          # a beam: only a roll clears it
        PIT: (True, False, True),           # a hole: jump it, same as a crate
        BCLEAR: (False, False, False),
        BLOW: (True, False, True),
        BDUCK: (True, True, False),
        BPIT: (True, False, True),
        MCLEAR: (False, False, False),      # a bystander is not an obstacle:
        MLOW: (True, False, True),          # you go through him, and the cell
        MDUCK: (True, True, False),         # underneath still rules the row
        CHASM: (True, True, True),          # too long to jump: go around
        WALL: (True, True, True)}

# The cells that pay. Membership, not a separate state bit — the payload is
# folded into the cell value so the track stays a single base-13 number and
# every downstream file keeps working on `track[lane]` alone.
BANANAS = frozenset({BCLEAR, BLOW, BDUCK, BPIT})
# A bystander minion stands on the road, on a crate, or crouches under a beam.
# Never over a PIT — there is nothing there to stand on, and drawing one would
# be the visualiser lying about a cell the agent is being paid for.
MINIONS = frozenset({MCLEAR, MLOW, MDUCK})
# What each cell pays on top of surviving it. One payload per cell by
# construction: a cell is bare, or carries a banana, or carries a minion.
# Making them independent axes would have multiplied the alphabet to 18 and
# the Q-table to 262k cells to buy stacked payoffs and no new *kind* of
# decision — the tension is already on the swipe axis, where it belongs.
PAYOUT = {**{c: BANANA for c in BANANAS}, **{c: MINION for c in MINIONS}}

# The cells no height and no banana can rescue. A row of nothing but these is
# unwinnable, which is what TRACK_CODES filters out.
IMPASSABLE = frozenset({CHASM, WALL})

# The obstacle each cell is, with any banana stripped off. Used by the row
# generator, which picks obstacles and bananas independently.
BASE_CELL = {CLEAR_CELL: CLEAR_CELL, LOW: LOW, DUCK: DUCK, PIT: PIT,
             BCLEAR: CLEAR_CELL, BLOW: LOW, BDUCK: DUCK, BPIT: PIT,
             MCLEAR: CLEAR_CELL, MLOW: LOW, MDUCK: DUCK,
             CHASM: CHASM, WALL: WALL}
WITH_BANANA = {CLEAR_CELL: BCLEAR, LOW: BLOW, DUCK: BDUCK, PIT: BPIT}
WITH_MINION = {CLEAR_CELL: MCLEAR, LOW: MLOW, DUCK: MDUCK}

# Chance a clustered row carries a banana run at all.
P_BANANA_RUN = 0.6
# ...and the chance it carries a bystander. Bananas come in runs; bystanders
# do not — in the game they stand about on their own, so this is one roll for
# one lane, and it is applied *after* the banana run so the two never land on
# the same cell. One payload per cell is a property of the alphabet, and the
# generator has to respect it or it deals rows that have no state.
P_MINION = 0.3


def _raw_code_to_track(code):
    track = []
    for _ in range(LANES):
        code, cell = divmod(code, N_CELLS)
        track.append(cell)
    return track


def _winnable(track):
    """At least one lane the minion can get through at some height."""
    return any(c not in IMPASSABLE for c in track)


# The 992 survivable rows, in raw-code order, plus the inverse map. Built once
# at import: 1000 iterations, and it replaces the arithmetic exclusion v3 got
# away with.
TRACK_CODES = [c for c in range(N_CELLS ** LANES)
               if _winnable(_raw_code_to_track(c))]
TRACK_INDEX = {c: i for i, c in enumerate(TRACK_CODES)}
TRACKS = [_raw_code_to_track(c) for c in TRACK_CODES]
N_TRACK = len(TRACK_CODES)                  # 992

assert N_TRACK == N_CELLS ** LANES - len(IMPASSABLE) ** LANES


def code_to_track(code):
    """Raw base-10 code 0..999 -> list of cells. Not a state index."""
    return _raw_code_to_track(code)


def track_to_code(track):
    """List of cells -> raw base-10 code. Not a state index."""
    return sum(cell * N_CELLS ** i for i, cell in enumerate(track))


def index_to_track(index):
    """Track index 0..991, as carried by a state -> list of cells."""
    return list(TRACKS[index])


def track_to_index(track):
    """List of cells -> track index. KeyError on an unwinnable row, which is
    the intended behaviour: such a row has no state to live in."""
    return TRACK_INDEX[track_to_code(track)]


def decode_action(action):
    action = int(action)
    return SHIFT[action % len(SHIFT)], action // len(SHIFT)


def hits(cell, height):
    return HITS[cell][height]


def has_banana(cell):
    return cell in BANANAS


def has_minion(cell):
    return cell in MINIONS


def payout(cell):
    """What this cell pays on top of surviving it: 0, BANANA, or MINION."""
    return PAYOUT.get(cell, 0.0)


def clustered_track(rng):
    """A game-shaped row: obstacles at random, bananas in a contiguous run.

    Bananas in the real game come in lines, not as independent coin flips, so
    this picks a *run* of adjacent lanes and fills it. Lanes holding a CHASM or
    a WALL are skipped — you cannot reach a banana in either.

    This is a demo and evaluation generator, not the training one. It cannot
    produce a split run like `b.b`, so a good fraction of the 2976 states are
    unreachable under it; training on this distribution would leave their rows
    of the Q-table permanently unvisited and end the "exact in every cell"
    property. See reset().
    """
    base = [CLEAR_CELL, LOW, DUCK, PIT, CHASM, WALL]
    while True:                              # reject the 8 unwinnable rows
        track = [base[int(rng.integers(len(base)))] for _ in range(LANES)]
        if _winnable(track):
            break

    if rng.random() < P_BANANA_RUN:
        length = int(rng.integers(1, LANES + 1))
        start = int(rng.integers(0, LANES - length + 1))
        for i in range(start, start + length):
            if track[i] in WITH_BANANA:
                track[i] = WITH_BANANA[track[i]]
    return track


class MinionRushEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, clustered=False):
        super().__init__()
        self.observation_space = spaces.Discrete(LANES * N_TRACK)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.render_mode = render_mode
        # Uniform over all 992 track indices by default. Every state has to be
        # reachable for the 1/n running average to converge to the truth in
        # every cell; clustered=True trades that away for game-shaped rows and
        # belongs in evaluation and the visualiser, not in training.
        self.clustered = clustered
        # Declared here as well as in reset(): check_env and some wrappers
        # inspect a freshly constructed env before reset() is ever called.
        self.lane = 1
        self.track = [CLEAR_CELL] * LANES
        self.height = GROUND

    def _encode(self, lane, index):
        return lane * N_TRACK + index

    def decode(self, s):
        """Inverse of _encode. A debugging tool — training never needs it."""
        return s // N_TRACK, s % N_TRACK

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)            # builds self.np_random from the seed
        clustered = self.clustered
        if options is not None and "clustered" in options:
            clustered = bool(options["clustered"])

        self.lane = int(self.np_random.integers(LANES))
        if clustered:
            self.track = clustered_track(self.np_random)
        else:
            self.track = index_to_track(int(self.np_random.integers(N_TRACK)))
        self.height = GROUND
        return self._encode(self.lane, track_to_index(self.track)), {}

    def step(self, action):
        shift, height = decode_action(action)
        self.lane = int(np.clip(self.lane + shift, 0, LANES - 1))
        self.height = height
        swiped = abs(shift)                 # charge for intent, not distance covered
        cell = self.track[self.lane]
        crashed = hits(cell, height)
        # You only collect the payload if you survive the cell it was sitting
        # on — no pocketing a banana off a crate you just ran into, and no
        # bowling a minion over on your way into a wall.
        collected = 0.0 if crashed else payout(cell)

        reward = (-SWIPE_COST * swiped - HEIGHT_COST[height]
                  + (CRASH if crashed else CLEAR)
                  + collected)
        obs = self._encode(self.lane, track_to_index(self.track))
        return obs, reward, True, False, {
            "crashed": crashed, "swiped": swiped, "height": height,
            "banana": collected == BANANA, "minion": collected == MINION}

    def render(self):
        if self.render_mode != "ansi":
            return None
        rows = []
        for h in (AIR, GROUND, ROLL):
            rows.append(" ".join(
                "M" if i == self.lane and self.height == h else " "
                for i in range(LANES)))
            if h == AIR:
                rows.append(" ".join(CELL_CHARS[c] for c in self.track))
        return "\n".join(rows) + "\n"


gym.register(id="MinionRush-v4", entry_point=MinionRushEnv, max_episode_steps=1)


def track_str(index):
    """Render a track index as '_bO' for printing tables."""
    return "".join(CELL_CHARS[c] for c in index_to_track(index))


def exact_q():
    """The true Q-table, by enumeration.

    Only possible because the env is deterministic and episodes are one step
    long, so Q(s, a) is just the immediate reward. Used as a correctness
    yardstick by train.py and read_q.py.
    """
    Q = np.zeros((LANES * N_TRACK, N_ACTIONS))
    for s in range(LANES * N_TRACK):
        lane, index = divmod(s, N_TRACK)
        track = TRACKS[index]
        for a in range(N_ACTIONS):
            shift, height = decode_action(a)
            landing = int(np.clip(lane + shift, 0, LANES - 1))
            cell = track[landing]
            crashed = hits(cell, height)
            Q[s, a] = (-SWIPE_COST * abs(shift) - HEIGHT_COST[height]
                       + (CRASH if crashed else CLEAR)
                       + (BANANA if has_banana(cell) and not crashed else 0.0))
    return Q


def wasted_jumps(Q):
    """States where greedy(Q) leaves the ground onto an already-clear cell.

    With no obstacle punishing a needless jump or roll, crashes no longer prove
    the height decision is right — a policy that jumps on every row is
    perfectly safe and merely poorer. This is the metric that catches that.
    """
    bad = []
    for s in range(Q.shape[0]):
        lane, index = divmod(s, N_TRACK)
        shift, height = decode_action(int(Q[s].argmax()))
        landing = int(np.clip(lane + shift, 0, LANES - 1))
        cell = TRACKS[index][landing]
        if height != GROUND and BASE_CELL[cell] == CLEAR_CELL:
            bad.append(s)
    return bad


def missed_bananas(Q, exact=None):
    """States where greedy(Q) walks past a banana that was strictly worth it.

    The banana counterpart of wasted_jumps(), and it exists for the same
    reason: a safe policy can still be a poor one, and crashes cannot see the
    difference. A banana two lanes away is *not* worth it — this only counts
    the ones where collecting strictly beats what greedy actually did.
    """
    exact = exact_q() if exact is None else exact
    bad = []
    for s in range(Q.shape[0]):
        lane, index = divmod(s, N_TRACK)
        track = TRACKS[index]
        chosen = int(Q[s].argmax())
        shift, _ = decode_action(chosen)
        landing = int(np.clip(lane + shift, 0, LANES - 1))
        if has_banana(track[landing]):
            continue                         # it did take one
        for a in range(N_ACTIONS):
            sh, h = decode_action(a)
            land = int(np.clip(lane + sh, 0, LANES - 1))
            if (has_banana(track[land]) and not hits(track[land], h)
                    and exact[s, a] > exact[s, chosen] + 1e-9):
                bad.append(s)
                break
    return bad
