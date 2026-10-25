"""MinionRush-v5 — a 3-lane, two-axis, one-decision dodge-and-collect environment. See README.md."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

LANES = 3

CLEAR_CELL, LOW, DUCK, PIT = 0, 1, 2, 3
BCLEAR, BLOW, BDUCK, BPIT = 4, 5, 6, 7
MCLEAR, MLOW, MDUCK = 8, 9, 10
CHASM, WALL = 11, 12
N_CELLS = 13

GROUND, AIR, ROLL = 0, 1, 2
N_HEIGHTS = 3
HEIGHT_NAMES = ["ground", "air", "roll"]

SHIFT = {0: -2, 1: -1, 2: 0, 3: +1, 4: +2}
N_ACTIONS = N_HEIGHTS * len(SHIFT)
CLEAR, CRASH, SWIPE_COST, AIR_COST = 2.0, -50.0, 1.0, 0.5
BANANA = 3.0
MINION = 5.0

HEIGHT_COST = (0.0, AIR_COST, AIR_COST)

ACTION_NAMES = ["L2", "L1", "stay", "R1", "R2",
                "L2+J", "L1+J", "jump", "R1+J", "R2+J",
                "L2+R", "L1+R", "roll", "R1+R", "R2+R"]
CELL_CHARS = {CLEAR_CELL: ".", LOW: "_", DUCK: "^", PIT: "o",
              BCLEAR: "b", BLOW: "B", BDUCK: "v", BPIT: "O",
              MCLEAR: "m", MLOW: "M", MDUCK: "w",
              CHASM: "=", WALL: "#"}

HITS = {CLEAR_CELL: (False, False, False),
        LOW: (True, False, True),
        DUCK: (True, True, False),
        PIT: (True, False, True),
        BCLEAR: (False, False, False),
        BLOW: (True, False, True),
        BDUCK: (True, True, False),
        BPIT: (True, False, True),
        MCLEAR: (False, False, False),
        MLOW: (True, False, True),
        MDUCK: (True, True, False),
        CHASM: (True, True, True),
        WALL: (True, True, True)}

BANANAS = frozenset({BCLEAR, BLOW, BDUCK, BPIT})
MINIONS = frozenset({MCLEAR, MLOW, MDUCK})
PAYOUT = {**{c: BANANA for c in BANANAS}, **{c: MINION for c in MINIONS}}

IMPASSABLE = frozenset({CHASM, WALL})

BASE_CELL = {CLEAR_CELL: CLEAR_CELL, LOW: LOW, DUCK: DUCK, PIT: PIT,
             BCLEAR: CLEAR_CELL, BLOW: LOW, BDUCK: DUCK, BPIT: PIT,
             MCLEAR: CLEAR_CELL, MLOW: LOW, MDUCK: DUCK,
             CHASM: CHASM, WALL: WALL}
WITH_BANANA = {CLEAR_CELL: BCLEAR, LOW: BLOW, DUCK: BDUCK, PIT: BPIT}
WITH_MINION = {CLEAR_CELL: MCLEAR, LOW: MLOW, DUCK: MDUCK}

P_BANANA_RUN = 0.6
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


TRACK_CODES = [c for c in range(N_CELLS ** LANES)
               if _winnable(_raw_code_to_track(c))]
TRACK_INDEX = {c: i for i, c in enumerate(TRACK_CODES)}
TRACKS = [_raw_code_to_track(c) for c in TRACK_CODES]
N_TRACK = len(TRACK_CODES)

assert N_TRACK == N_CELLS ** LANES - len(IMPASSABLE) ** LANES


def code_to_track(code):
    """Raw base-13 code 0..2196 -> list of cells. Not a state index."""
    return _raw_code_to_track(code)


def track_to_code(track):
    """List of cells -> raw base-13 code. Not a state index."""
    return sum(cell * N_CELLS ** i for i, cell in enumerate(track))


def index_to_track(index):
    """Track index 0..2188, as carried by a state -> list of cells."""
    return list(TRACKS[index])


def track_to_index(track):
    """List of cells -> track index. KeyError on an unwinnable row."""
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
    return PAYOUT.get(cell, 0.0)


def clustered_track(rng):
    """A game-shaped row for demos and evaluation: obstacles at random, bananas in a contiguous run."""
    base = [CLEAR_CELL, LOW, DUCK, PIT, CHASM, WALL]
    while True:
        track = [base[int(rng.integers(len(base)))] for _ in range(LANES)]
        if _winnable(track):
            break

    if rng.random() < P_BANANA_RUN:
        length = int(rng.integers(1, LANES + 1))
        start = int(rng.integers(0, LANES - length + 1))
        for i in range(start, start + length):
            if track[i] in WITH_BANANA:
                track[i] = WITH_BANANA[track[i]]

    if rng.random() < P_MINION:
        free = [i for i in range(LANES) if track[i] in WITH_MINION]
        if free:
            i = int(rng.integers(len(free)))
            track[free[i]] = WITH_MINION[track[free[i]]]
    return track


class MinionRushEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(self, render_mode=None, clustered=False):
        super().__init__()
        self.observation_space = spaces.Discrete(LANES * N_TRACK)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.render_mode = render_mode
        self.clustered = clustered
        self.lane = 1
        self.track = [CLEAR_CELL] * LANES
        self.height = GROUND

    def _encode(self, lane, index):
        return lane * N_TRACK + index

    def decode(self, s):
        """Inverse of _encode. A debugging tool — training never needs it."""
        return s // N_TRACK, s % N_TRACK

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
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
        swiped = abs(shift)
        cell = self.track[self.lane]
        crashed = hits(cell, height)
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


gym.register(id="MinionRush-v5", entry_point=MinionRushEnv, max_episode_steps=1)


def track_str(index):
    """Render a track index as '_bO' for printing tables."""
    return "".join(CELL_CHARS[c] for c in index_to_track(index))


def exact_q():
    """The true Q-table, by enumeration — a correctness yardstick for train.py and read_q.py."""
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
                       + (0.0 if crashed else payout(cell)))
    return Q


def wasted_jumps(Q):
    """States where greedy(Q) leaves the ground onto an already-clear cell."""
    bad = []
    for s in range(Q.shape[0]):
        lane, index = divmod(s, N_TRACK)
        shift, height = decode_action(int(Q[s].argmax()))
        landing = int(np.clip(lane + shift, 0, LANES - 1))
        cell = TRACKS[index][landing]
        if height != GROUND and BASE_CELL[cell] == CLEAR_CELL:
            bad.append(s)
    return bad


def missed_payloads(Q, exact=None):
    """States where greedy(Q) walks past a reachable, survivable payload that strictly beat its choice."""
    exact = exact_q() if exact is None else exact
    bad = []
    for s in range(Q.shape[0]):
        lane, index = divmod(s, N_TRACK)
        track = TRACKS[index]
        chosen = int(Q[s].argmax())
        for a in range(N_ACTIONS):
            sh, h = decode_action(a)
            land = int(np.clip(lane + sh, 0, LANES - 1))
            if (payout(track[land]) and not hits(track[land], h)
                    and exact[s, a] > exact[s, chosen] + 1e-9):
                bad.append(s)
                break
    return bad


missed_bananas = missed_payloads
