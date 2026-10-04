# DocJev — learning journal

What we learned along the way, explained simply, with intuition first and the
formula second. `JOURNAL.md` records decisions and results; this file records
understanding.

---

## 1. One answer, many kinds of "confidence"

*2026-10-04. From our first Clef-flash API call. (Rewritten the same day in
simpler words, at the user's request.)*

### The puzzle

We asked Clef-flash which team should handle a support ticket. It replied:

```
technical  0.9601
billing    0.0327
sales      0.0072
confidence 0.8844   ← why not 0.9601?
```

The model's best guess has probability 0.96, but its "confidence" says 0.88.
Both numbers are honest; they measure **different things**.

### The setup: a model places bets

Think of the probabilities as the model spreading 100 chips across the options.
"technical 0.96" means 96 chips on technical, 3 on billing, 1 on sales.

There are several fair ways to ask "how sure is the model?", and each one looks
at the chips differently.

### Four ways to read "how sure"

We use four example bets on three document types (letter / invoice / memo):

| | letter | invoice | memo | in words |
|---|---|---|---|---|
| **A** | 90 | 5 | 5 | clearly sure |
| **B** | 50 | 50 | 0 | torn between two |
| **C** | 50 | 25 | 25 | a favourite, two weak rivals |
| **D** | 33 | 33 | 33 | no idea at all |

**1. Top bet: "How many chips are on the winner?"**
Just read the biggest number. A = 0.90, B = 0.50, C = 0.50, D = 0.33.
Intuition: *how much is the model willing to bet on its pick.*

**2. Lead over the runner-up: "How far ahead is the winner?"**
Like a race: winner minus second place. A = 0.90 − 0.05 = **0.85**;
B = 0.50 − 0.50 = **0** (a dead heat); C = 0.50 − 0.25 = **0.25**; D = 0.
Intuition: *is there a clear winner, or a close fight?*

**3. Lead over the average loser (what Jev shows): "How far is the winner above
the rest, on average?"** Winner minus the average of the others.
A = 0.90 − (0.05 + 0.05)/2 = **0.85**; B = 0.50 − (0.50 + 0)/2 = **0.25**;
C = 0.50 − (0.25 + 0.25)/2 = **0.25**; D = 0.

**4. Concentration (what Clef's API shows): "If I ask the model twice, how
often will it say the same thing?"**
Imagine the model answering by drawing a chip at random, twice. The chance both
draws land on the same option is: square each probability and add them up.

- A: 0.90² + 0.05² + 0.05² = 0.81 + 0.0025 + 0.0025 = **0.815**
- D: 0.33² × 3 = **0.333** (with 3 options, a clueless model still agrees with
  itself 1 time in 3)

That "1 in 3 even when clueless" floor is awkward, so the score is stretched to
run from 0 (clueless) to 1 (all chips on one option):

    concentration = (agreement − floor) / (1 − floor),   floor = 1/K

- A: (0.815 − 0.333)/(1 − 0.333) = **0.72**
- B: (0.50 − 0.333)/0.667 = **0.25**
- C: (0.375 − 0.333)/0.667 = **0.06**
- D: **0**

Check against our real reply: agreement = 0.9601² + 0.0327² + 0.0072² = 0.9229;
(0.9229 − 0.3333)/0.6667 = **0.8844**. That is the API's number exactly, so this
is how Clef's hosted API computes `confidence`. (One example so far; we'll check
it on every reply in the baseline.)

*(There is a fifth common one, entropy, "how many options are still in play".
It behaves much like concentration, so we skip the formula for now.)*

### Put them side by side

| | top bet | lead over 2nd | lead over avg loser | concentration |
|---|---|---|---|---|
| **A** clearly sure | 0.90 | 0.85 | 0.85 | 0.72 |
| **B** torn between two | 0.50 | **0.00** | 0.25 | **0.25** |
| **C** favourite, two weak rivals | 0.50 | **0.25** | 0.25 | **0.06** |
| **D** no idea | 0.33 | 0 | 0 | 0 |

Look at **B vs C**. Both bet exactly 0.50 on their pick.

- *Lead over 2nd* says **C is surer** (there's a clear favourite; B is a tie).
- *Concentration* says **B is surer** (B has narrowed it down to two options;
  C is still spread across three).
- *Top bet* says they are **equally sure**.

None of these is "wrong". They answer different questions:

| measure | the question it answers |
|---|---|
| top bet | How likely is the picked answer to be right (if the model is honest)? |
| lead over 2nd | Is the model stuck between two specific options? |
| lead over avg loser | Does the pick stand out from the field? |
| concentration | Has the model narrowed things down, whatever it picked? |

### Why this matters for us

If we tell a system "auto-approve when confidence > 0.9", **which** confidence
we mean changes which documents get through. That is a design decision for
DocJev, and it is **still open** (see `JOURNAL.md`).

### Side note: rounding

The API rounds every probability to 4 decimals, so a very confident wrong answer
can come back as exactly 0.0000. Some scores take a log of the probability, and
log(0) is minus infinity, so our scoring code must replace 0 with a tiny number.
