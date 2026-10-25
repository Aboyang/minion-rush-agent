# PROGRESS — what we added today

**Today's change in one line:** the minion learned to **want something**, and
for the first time the optimal policy will walk *into* an obstacle on purpose.

---

## 1. Where things stand

| | v1 (lanes) | v2 + jump | v2 + roll | **v3 + banana (today)** |
|---|---:|---:|---:|---:|
| cell types | 2 | 3 | 4 | **7** |
| heights | 1 | 2 | 3 | 3 |
| track codes | 7 | 26 | 63 | **342** |
| **states** | 21 | 78 | 189 | **1026** |
| **actions** | 5 | 10 | 15 | 15 |
| **Q-cells** | 105 | 780 | 2835 | **15390** |

Every one of those 15 390 cells is still learned to the exact optimum, and
`exact_q()` still enumerates the truth to check it against. That property is
the thing this project is actually about, and §5 is the story of nearly
losing it.

---

## 2. The modelling decision — why the banana is not a fifth obstacle

The obvious move was `CLEAR / LOW / DUCK / WALL / BANANA`, a fifth cell type.
It is wrong, and seeing why is the whole lesson of the change.

The goal was to watch the agent face a row where **going for the banana forces
a dodge that an empty lane would not have required.** But the episode is
**one row long**. A banana sitting in its own clear lane is just a free +2
with no decision attached — there is no later row for it to get you into
trouble in. For the dilemma to exist inside a single step, a banana has to be
able to sit *on top of* an obstacle.

So the banana is a **second axis on the cell**, not a fifth value of it:

```
cell = obstacle × (banana or not)
```

A banana can ride on `CLEAR`, `LOW` or `DUCK`, never in a `WALL` — you cannot
reach into a wall. So the lane alphabet is `3 × 2 + 1 = 7`, not 8:

| cell | symbol | hits you at | answer | pays |
|---|---|---|---|---:|
| `CLEAR` | `.` | — | stay down | |
| `LOW` | `_` | GROUND, ROLL | jump it | |
| `DUCK` | `^` | GROUND, AIR | roll under it | |
| `BCLEAR` | `b` | — | stay down | **+2** |
| `BLOW` | `B` | GROUND, ROLL | jump it | **+2** |
| `BDUCK` | `v` | GROUND, AIR | roll under it | **+2** |
| `WALL` | `#` | all three | swipe around it | |

`v` is `^` turned over: the beam is still there, you still go under it.

**A banana changes a cell's payoff, never its geometry.** Each banana twin
hits exactly like the obstacle it is sitting on, which is why `HITS` grew from
4 rows to 7 and nothing else in the collision path changed at all:

```python
HITS = {CLEAR_CELL: (False, False, False),
        LOW:        (True,  False, True ),
        DUCK:       (True,  True,  False),
        BCLEAR:     (False, False, False),   # same as CLEAR
        BLOW:       (True,  False, True ),   # same as LOW
        BDUCK:      (True,  True,  False),   # same as DUCK
        WALL:       (True,  True,  True )}
```

### Keeping the free exclusion

v2's neatest trick was numbering `WALL` **top**, so the one unsurvivable row —
all-`WALL` — was exactly the top code and could be dropped by subtracting 1,
leaving a contiguous `0..62` with no renumbering.

Putting the banana twins at 3, 4, 5 would have put `WALL` at 3 and broken
that: all-`WALL` would be code `3 + 7·3 + 49·3 = 171`, somewhere in the
middle, and excluding it would mean a renumbering table. So `WALL` is
numbered **6, last**, and the trick survives intact:

```
N_TRACK = 7³ − 1 = 343 − 1 = 342      codes 0..341, all-WALL = 342, dropped

|S| = LANES × N_TRACK = 3 × 342 = 1026     states 0..1025
```

### What it cost

```
v1:   3 × (2³ − 1) =  21
v2:   3 × (3³ − 1) =  78
v2r:  3 × (4³ − 1) = 189
v3:   3 × (7³ − 1) = 1026
```

The cost curve from §9 of last time held exactly: states are cubic in the cell
alphabet, and this change nearly doubled the alphabet. The action space did
not move at all — **the banana is collected at any height**, so it never
changes *which* posture is correct, only whether a cell is worth travelling
to. The entire new tension is priced in swipes, on the axis that already
existed.

---

## 3. Reward — one new term, and a deliberate tie

```python
reward = −SWIPE_COST · |shift|              # movement cost
         − HEIGHT_COST[height]              # posture cost
         + (CRASH if crashed else CLEAR)    # outcome
         + (BANANA if grabbed else 0)       # NEW
```

`BANANA = +2.0`, worth exactly one cleared row, and `grabbed` requires
`not crashed` — **you do not pocket a banana off a crate you just ran into.**

That one constant produces the whole ladder:

```
banana underfoot, clear ground      +2 +2         = +4.0   <- new ceiling
banana underfoot, under a beam      +2 +2 -0.5    = +3.5
banana one lane over, on a crate    +2 +2 -1 -0.5 = +2.5
banana two lanes over, clear        +2 +2 -2      = +2.0   <- ties with
nothing at all, stand still         +2            = +2.0      standing still
```

**The two-lane banana is exactly break-even, and that is on purpose.** It is
the price point where greed stops paying, and it shows up in the table as a
genuine indifference rather than as a cost we forgot to charge. At `BANANA =
3` the agent would cross the whole track for every banana; at `BANANA = 1` it
would never leave its lane for one. +2 puts the decision boundary in the
middle of the action space, where it is visible.

The table now holds **16 distinct values** (was 12): `−52.5` to `+4.0`. The
legend in `read_q.py` is no longer hand-written — it is derived from the same
four constants the env charges, because at 16 entries a hand-maintained copy
is a copy that drifts.

`+2.00` is now **ambiguous**, and that collision is the interesting one: it
means both "nothing here, stay put" and "banana two lanes away, go get it".

---

## 4. Row generation — bananas come in lines, down the wrong axis

Real bananas come in lines, not independent coin flips. The first attempt got
the **axis** wrong, which is worth recording because the fix is the most
interesting thing in the change.

`clustered_track()` picks a contiguous run **across the lanes of one row**:

```
'bb.'  '.bb'  'bbb'  'B_v'  'b#b'      yes
'b.b'                                   never
```

But that is not what Minion Rush puts on screen. In the game a banana line
runs **down a lane, along the direction you are running** — it is a trail
through *successive* rows, occasionally stepping sideways. A sideways run
inside a single row is the rare case, not the common one.

And here is the catch:

> **A trail spans consecutive rows. The episode is one row.**

So a trail cannot be part of the state, cannot be part of the observation,
and cannot be part of training. `clustered_track()` is genuinely the most
banana-clustering a *one-row* generator can express — within a row, sideways
is the only direction there is.

The trail therefore lives **only in the visualiser**, which is the one place
that strings one-step decisions into a continuous run. `nextTrack()` carries
a little state machine across rows — `{lane, left}` plus a gap counter
— and drops a banana in the trail lane each row, stepping sideways with
`p = 0.16` and marking **both** lanes on the row where it bends, so the
diagonal is drawn rather than teleported. The trail lane is never a `WALL` —
that would break the line.

It is tuned to the game's actual **rhythm**, which is long, solid and
**single file**: one banana per row, same lane, row after row for a serious
stretch of track, then a clean break with nothing at all, then another
stretch. The break is what makes the stream read as a stream.

| | |
|---|---:|
| rows carrying a banana | **89%** |
| bananas per row | **1.04** |
| rows with two (bends only) | 15% |
| trail length | 20-48 rows |
| the interim break | 3-6 empty rows |

**Width is deliberately not a parameter, and that is the whole point.** A
first pass made trails 1-3 lanes wide, which looked lush and quietly
destroyed the experiment: at 1.75 bananas per row there is a banana in
whatever lane the minion already occupies, so *"is it worth detouring for?"*
never gets asked and the axis stops testing anything. One lane means the
question is live on every single row. Two bananas appear only where the line
**bends**, and that is a connection being drawn, not a band.

### Row spacing — why none of the density tuning worked

The trails kept not looking like the game, and no amount of tuning the
*trail* fixed it, because the problem was underneath:

| | world units | in lane-widths |
|---|---:|---:|
| lane spacing | 2.1 | 1.0 |
| **row to row** (`ROW_GAP`) | **26** | **12.4** |
| the game's banana spacing | — | ~0.6-0.8 |

**Rows are 12.4 lane-widths apart; the game's bananas are under one.** One
banana per cell draws a single dot, then 26 units of empty boardwalk, then
another dot. The gap was never *between* trails — it was *inside* them, and
no trail parameter can reach it.

So a banana cell now renders a **run of 14 props** spanning the whole row
gap, at 1.86 units ≈ 0.88 lane-widths apart. The run spans 24.14 and the
next row's run starts 1.86 later — exactly one step — so consecutive banana
rows join into one continuous line with even spacing across the seam.

This is purely presentational. **The cell is still one decision and still
pays +2 once**, which is why collection frees the whole run rather than one
prop: leaving thirteen behind would draw bananas the minion can never have.

Only the prop on the decision point carries the face and halo; the other
thirteen are bare crescents. 14 faces × 7 visible cells is 1200 draw calls
for detail that is sub-pixel at the far end of the road — **345 meshes
instead of 1208.**

### The line arcs

The trail lane is **set outright** rather than left to the random obstacle
draw. Left random it would carry an obstacle two rows in three and the line
would thrash up and down; set, it runs flat with the occasional event:

| | | hover |
|---|---:|---:|
| clear road | 74% | 0.62 |
| **over a crate** (`B`) | 16% | **1.38** |
| **under a beam** (`v`) | 10% | **0.60** |

Those heights are not decoration — they are `BANANA_Y`, so the line visibly
**humps up over a crate and dips under a beam**. And because the cell type is
what carries the height, the arc is simultaneously the *mechanic*: following
the line over a crate means jumping, under a beam means rolling. The rows
where the visual arc happens are exactly the rows where greed costs a
posture.

With a 14-prop run the arc becomes a real **curve** rather than one raised
dot. A crate is 0.7 deep and the run is 26 long, so holding 1.38 across the
whole run would draw a plateau; `bananaRunY()` eases it back to road height
at the ends, which also lands the run's ends on 0.62 — exactly where the
neighbouring row's run begins, so the line stays joined:

```
dz   -12.07  -8.36  -4.64  -0.93  +0.93  +4.64  +8.36  +12.07
y     0.72    1.07   1.28   1.38   1.38   1.28   1.07   0.72
                            ^^^^^^^^^^^ the crate is here
```


```
lane:  0 1 2        the line's hover height
       ^ v _        0.60   dip under a beam
       · b ^        0.62
       _ B #        1.38   arc over a crate
       # v ^        0.60   dip under a beam
       · b ·        0.62
       _ B _        1.38   arc over a crate
       # b _        0.62
       · b ·        0.62
       ^ b #        0.62
       _ b _        0.62
       # b ·        0.62
       _ b B        <- bend
       · · b        0.62
       · _ v        0.60   dip under a beam
       ^ # #        <- the break
       ^ # _        <-
       # _ #        <-
       ^ · ^        <-
       · ^ ·        <-
       # · B        1.38   a new line starts in lane 2
       ^ · b        0.62
```

**The agent has no idea any of this is happening.** It is myopic by
construction — one row, one decision. Everything that looks like the minion
"following" a trail is two or three independent rows of greed in a row.

Training, meanwhile, is uniform. Which brings us to why.

---

## 5. The trap: clustering would have killed the exactness

A clustered draw **cannot produce a split run**. Lanes 0 and 2 holding
bananas with a *passable* non-banana lane between them is unreachable — there
is no run shape that makes it. That is exactly `3 × 3 × 3 = 27` codes:

```
b.b  B.b  v.b  b_b  B_b  v_b  b^b  B^b  v^b
b.B  B.B  v.B  b_B  B_B  v_B  b^B  B^B  v^B
b.v  B.v  v.v  b_v  B_v  v_v  b^v  B^v  v^v
```

27 codes × 3 lanes = **81 states**, 1215 Q-cells, that would be visited
**zero times, forever**. `max |Q − exact|` would never reach 0, not because
training was too short but because the generator cannot deal those rows.
(`b#b` *is* reachable — the wall interrupts the run legitimately, which is
why it is 27 and not 54.)

So the two distributions were split:

| | used by | shape | reaches |
|---|---|---|---|
| **uniform** over 0..341 | `reset()`, training | — | 342 / 342 codes |
| **clustered** | `evaluate.py` | sideways run, per row | 315 / 342 |
| **trail** | the visualiser | lengthwise, across rows | 86% of rows carry one |

**Game-shaped rows to look at, uniform rows to learn from.** This is the first
time in the project that the training distribution and the demo distribution
are deliberately different, and it is worth saying out loud why that is
allowed here: `Q(s,a)` is the immediate reward, so the *targets* do not depend
on the state distribution at all. The distribution only decides **coverage**.
Uniform is not more correct — it is just the only one that covers everything.

---

## 6. Training

| | before | after |
|---|---:|---:|
| Q-cells | 2835 | **15390** |
| `MIN_EPSILON` | 0.1 | **0.15** |
| `EPISODES` | 1 M | **5 M** |
| expected visits, rarest cell | ~35 | ~49 |
| actual least-visited cell | 19 | **25** |
| wall clock | 7.4 s | **40.2 s** |

```
max |Q − exact|                   0.000000     exact in all 15390 cells
greedy safe in all 1026 states    True
needless jumps/rolls              0 / 1026
bananas left on the table         0 / 1026
```

Expected reward per row under the optimal policy: **+2.7817** (was +1.4021).
Almost all of that gain is banana income, not better dodging.

---

## 7. Results — and the row that justifies the change

| | avg reward | crashes | bananas | avg swipes |
|---|---:|---:|---:|---:|
| random | −26.00 | 1039 / 2000 | 503 | 1.17 |
| **trained** | **+2.81** | **0 / 2000** | **1596** | 0.47 |
| random (clustered) | −29.85 | 1179 / 2000 | 305 | 1.17 |
| **trained (clustered)** | **+2.30** | **0 / 2000** | **1056** | 0.42 |

**Average swipes went 0.27 → 0.47.** The v2 policy moved only when something
was in the way; this one travels on purpose. That number is the behaviour
change, and it is the one a crashes-only evaluation cannot see.

### The decision, in one row

`evaluate.py`'s rollout is now seeded to **22**, which is the entire point of
v3 in a single step:

```
      _   v   .
lane  0   1   2        minion standing in lane 2, on clear road
```

Standing still is a free **+2** — lane 2 is empty, nothing can touch him.
Instead the policy picks `L1+R`: swipe left onto a beam, and roll under it.

```
+2 clear   +2 banana   −1 swipe   −0.5 roll   =   +2.5
```

**He took on a dodge he did not need, for half a point.** The lane he was
already standing in required nothing. This is exactly the situation the change
was built to produce, and the agent found it without being told it existed.

### How often, and when it refuses

Of the **837 states with a banana somewhere on the row**:

```
greedy takes one               801
  ...by swiping to it          360
  ...by leaving the ground     518
greedy walks past it            36
```

All 36 refusals are two-lanes-away bananas, and **none of them is a mistake**:

- **20 are exact ties** — `..b` from lane 0, where `stay` and `R2` both pay
  +2.0 and numpy's first-max tie-break picks `stay`.
- **16 are strict wins for refusing** — a banana two lanes away *and* on an
  obstacle pays `+2 +2 −2 −0.5 = +1.5`, which loses to standing still at +2.0.

`missed_bananas()` counts only the strictly-worth-it ones, and it is **0**.

### Ties: 65 states, two kinds

```
state  147  lane 0  track ..b  ->  ['stay', 'R2']      greed break-even
state  492  lane 1  track b.b  ->  ['L1', 'R1']        a banana either side
state  435  lane 1  track ^#_  ->  ['R1+J', 'L1+R']    the old v2 symmetry
state  585  lane 1  track v#B  ->  ['R1+J', 'L1+R']    ...with a banana on each
```

The last pair is the nicest thing in the table: standing in a `WALL` with a
banana-on-a-beam left and a banana-on-a-crate right. Roll left or jump right,
same price, and now with a reason to be glad either way.

---

## 8. What we learned today

**Two new diagnostics, and they decay in opposite directions.** `wasted_jumps()`
counts safety-but-wasteful; `missed_bananas()` counts greed left unclaimed.

| episode | unsafe | needless jumps/rolls | bananas missed | cells seen |
|---:|---:|---:|---:|---:|
| 0 | 732 | 0 | 396 | 0 |
| 1 000 | 582 | 99 | 360 | 966 |
| 5 000 | 238 | **203** | 243 | 3 867 |
| **30 000** | **0** | 160 | 106 | 8 325 |
| 200 000 | 0 | 26 | 12 | 14 064 |
| **500 000** | 0 | 1 | **0** | 15 333 |
| **1 000 000** | 0 | **0** | 0 | 15 389 |

`wasted_jumps` **rises before it falls** — it peaks at 203 around ep 5000,
because an agent that has learned "leaving the ground saves me" has not yet
learned "but only when something is there". `missed_bananas` **only ever
falls**, from 396 at episode zero. A thing you have not learned to want, you
never over-want. Caution has to be unlearned; greed only has to be acquired.

**The policy is safe at ep 30 000 and not optimal until ep 1 000 000.** For
970 000 episodes it crashes zero times and is quietly leaving money on the
floor. That gap is 10× wider than v2's, and the reason is coverage, not
difficulty: the last few thousand of 15 390 cells are simply rare.

**The interesting bug was the one we did not write.** Making the banana a
fifth cell type would have compiled, trained, converged exactly, and produced
a perfectly good agent that never once faced the decision the change was for.
The environment would have been *correct* and *pointless*. Checking that a
new axis can actually express the behaviour you want to observe — before
counting states — is the step that was nearly skipped.

---

## 9. Files changed today

| file | change |
|---|---|
| `minion_rush.py` | `v3`; 7-cell alphabet, base-7 code, `BANANA`, `clustered_track()`, `missed_bananas()` |
| `train.py` | `MIN_EPSILON` 0.1 → 0.15, `EPISODES` 1M → 5M, reports missed bananas |
| `evaluate.py` | banana column, uniform *and* clustered rows, rollout reseeded to 22 |
| `read_q.py` | legend derived rather than hand-listed; 7-cell legend; tie list capped |
| `export_checkpoints.py` | checkpoints to 5M, `missed_bananas` per snapshot, q rounded to 2dp |
| `minion_q.npy` | retrained — the 189×15 table was invalid at the new shape |
| `web/checkpoints.json` | regenerated at 1026×15 (400 KB → 2.3 MB) |
| `web/index.html` | banana mesh + happy face + halo, lengthwise trail generator, banana HUD counter, base-7 decode |

### On the visualiser

- The banana is a **180° torus arc** with a green stem at one end and a brown
  tip at the other, swung a quarter turn so its belly faces the camera. On
  that belly goes the **only happy face in the scene**: big wide-set eyes with
  oversized low pupils, a gleam on each, blush, and a smile made from the same
  torus primitive as the body at one sixth the size. Every obstacle scowls, so
  the expression carries the rule the way colour and silhouette already do —
  *this is the one you run towards.*
- It **wobbles rather than spins.** The first version turned a full circle,
  which swings the face away half the time, and the face is the entire reason
  you want it. Now it tilts ±0.38 rad on one axis and ±0.13 on another at a
  different rate, so the two never sync into a mechanical loop.
- A **soft unlit halo** (`MeshBasicMaterial`, opacity 0.11) means it glows the
  same at every point on the road — it has to be spottable three rows out,
  while the decision is still being made.
- **The squash caught me out.** `body.scale.set(1, 0.82, 1)` shrinks the
  torus's *bulge*, not its span, so the belly sits at `0.246`, not `0.3`. The
  face was first placed at `z = 0.345` and would have floated in front of the
  fruit. Worked out on paper rather than by eye, since there was no browser to
  check it in.
- Hover heights are derived from the half-extent (0.39) plus the bob (0.09),
  so nothing ever clips: `0.62` over open road, `1.38` resting exactly on a
  crate's lid at 0.90, `0.60` under a beam whose underside is at 1.15 (tops
  out at 1.08, clears by 0.07).
- It is **removed the instant it is collected**, in `resolve()`, so the +2 on
  the HUD and the banana disappearing are the same event.
- The page lays bananas in **lengthwise trails** (§4), not the per-row
  sideways runs `clustered_track()` makes, and says so in a comment — a reader
  who diffs it against `reset()` deserves to find out why before assuming it
  is a bug.

**Still stale:** `web/README.md` was last updated for v2-with-jump and still
says 78 states / 780 cells. It missed the roll change too.

---

## 10. Where the next axis would go

```
+1 cell type   ->  states × ~cubic      (expensive)
+1 height      ->  actions × linear     (cheap)
+1 cell AXIS   ->  states × (cubic in the multiplied alphabet)
```

The banana was the first *axis* rather than *value*, and it is the cheapest
kind of expansion per unit of new behaviour — it multiplied the alphabet by
<2 and left the action space untouched, while adding the first genuine
trade-off in the project.

What still would not fit, and would end the "one decision, exact Q" property:

- **multi-step episodes** — and the trail in §4 is now standing right on top
  of this door. The visualiser draws a line of bananas down a lane; the agent
  cannot see it, because a trail pays off over rows and `Q(s,a)` here is the
  immediate reward. Letting it see one would mean the state remembering that
  it is *already in* a trail, which ends the enumerable-truth property in one
  step. This is where the banana wants to go next, and it is a different
  project.
- **a wider track** — 5 lanes at 7 cells is 5 × (7⁵ − 1) = 84 030 states, and
  tabular is over.
- **obstacles at distance** — seeing two rows ahead squares the track term:
  342² ≈ 117 000 track codes.

At 1026 states `exact_q()` still enumerates the truth in milliseconds. That
is the budget being spent, and it has roughly one more doubling in it.
