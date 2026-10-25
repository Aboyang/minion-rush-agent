# Watching the Q-table learn

```bash
python export_checkpoints.py        # -> web/checkpoints.json
python -m http.server -d web 8777   # http://localhost:8777
```

Pick an episode on the left; the runner immediately plays that checkpoint's
greedy policy down the track until it crashes.

## The one thing to know

`MinionRush-v2` is a **one-step** environment: `reset()` deals a `(lane, track)`,
the agent acts once, the episode ends. A continuous "run until you crash" is
not something the agent was ever trained on.

So the runner does not invent a new policy — it chains the same one-step
decision over a stream of rows. Each incoming row is a fresh track code, the
agent reads `Q[lane * 26 + code]`, takes `argmax` (first-max tie-break, as
numpy did in training), and the lane it lands in becomes the `lane` of the next
decision. Rewards are charged exactly as the env charges them: `-1` per lane
the swipe intended to cross, `-0.5` to leave the ground, then `+2` survived or
`-50` hit.

## Two axes

`x` is the lane. `y` is the height taken through the row, carried in the action,
so the action space is `5 shifts × 2 heights = 10`. Two obstacles, the two the
real game leans on:

| cell | glyph | look | hits you | what to do |
|---|---|---|---|---|
| `CLEAR` | `·` | — | never | anything |
| `LOW` | `_` | amber hurdle | on the ground | jump it |
| `WALL` | `#` | pink crate, full height | at any height | swipe around it |

`WALL` is what keeps the lane axis worth using — without it, `jump` always
beats any swipe, since it is cheaper, and the swipe actions become dead code.

**Nothing here punishes a jump.** Restraint comes from `JUMP_COST = 0.5`: when
the landing cell is clear, the grounded action beats its airborne twin by
exactly 0.5, so the optimal policy leaves the ground only when that converts a
crash into a clear. The consequence is that **crashes no longer prove the
height decision is right** — a policy that jumps on every row is perfectly
safe, just 0.5/row poorer. That is why there is a second metric, `needless
jumps`: states where greedy goes airborne onto an already-clear cell.

Two consequences worth expecting:

- **A converged policy cannot crash.** With no unsafe states, every one of the
  78 states has a surviving `(lane, height)` and greedy finds it, so the run is
  endless by construction. That is the punchline, not a hang.
- **Code 26 never appears**, because `reset()` draws from `0..25`. `WALL` in all
  three lanes would be unsurvivable at any height, and excluding it is what
  makes the game fair — the same trick the old mask-7 exclusion played.

Courses are generated from a seed, so **Same course** sends a different
checkpoint down the identical obstacle stream — otherwise you are comparing
policies on different tracks and the comparison means nothing.

## What the checkpoints actually do

Exact expected reward per row of `greedy(Q)`, swept over all 78 states:

| episode | exp. reward / row | unsafe | needless jumps | cells seen |
|---:|---:|---:|---:|---:|
| 0 | -34.00 | 51 | 0 | 0 / 780 |
| 50 | -31.10 | 47 | 4 | 48 |
| 250 | -14.46 | 23 | **14** | 205 |
| 1000 | -2.06 | 5 | 8 | 564 |
| 2500 | 0.71 | 1 | 1 | 742 |
| 5000 | **1.38** | **0** | **0** | 777 |
| 12000+ | 1.40 | 0 | 0 | 780 / 780 |

Two curves, not one, and they are worth watching separately. Crashes fall
steadily from the start. Needless jumps *rise* first — peaking at 14 around
episode 250 — because the agent learns "jumping saves me from a crate" (a
52-point lesson) long before it learns "but it costs half a point when there is
no crate" (a 0.5-point lesson). Penalty magnitude sets the learning order.

The table only becomes exact (`max |Q - exact| == 0`) once all 780 cells have
been visited, but it is already *safe* at 5000 — correct ranking arrives well
before correct values.
