# How far accuracy goes, and why

Written after a long attempt to reach 85–90 %. Several conclusions here reverse
earlier ones in this project; where that happens the reason is stated, because
the *reason* is more useful than the number.

---

## Final position

| Metric | Value | Repeats |
|---|---|---|
| Window accuracy | **0.8018 ± 0.0103** | 3 seeds |
| Macro-F1 | **0.7944 ± 0.0114** | 3 seeds |
| **Per-recording accuracy** | **0.8332 ± 0.0056** | 3 seeds |
| Majority baseline | 0.4153 | — |

Leave-one-bearing-out over all 29 bearings, pooled. Single model — no ensemble,
no test-time augmentation (`n_seeds: 1`, `tta_shifts: [0]` in
`artifacts/runs/lobo_summary.json`).

The exported INT8 weights were trained with the identical single-model recipe, on
all 29 bearings (`train.py:523`, `train.py:548`); 0.8018 is the leave-one-bearing-out
estimate of that recipe's accuracy on an unseen bearing. No LOBO fold model ships —
the 29 fold models each saw 28 bearings, and the quantised artefact is a 30th model
that saw all of them. The recipe is the same; the training set is not.

Corollary: the `float_accuracy: 0.9927` / `int8_accuracy: 0.9920` figures in
`artifacts/int8_export/verification_full.json` are **training-set** accuracies. They
exist to measure INT8-vs-float agreement (`argmax_agreement: 0.9988`), which is what
the export bar is about. Neither is a generalisation estimate and neither belongs on
a slide.

Against a 85 % target: per-recording is **1.7 points short**. Window accuracy is
5 points short.

---

## What actually worked

| Change | Effect | Why it worked |
|---|---|---|
| **Remove the inner validation split** | 0.7283 → 0.7823, **std 0.108 → 0.0018** | It consumed 3 bearings *and* injected the variance |
| **5-channel input + strong augmentation** | 0.7823 → **0.8018** | More signal, but only survivable with more regularisation |
| Drop the ensemble | 0.7837 → 0.7822, 3× faster | It was compensating for variance that no longer exists |

Total: **0.7823 → 0.8018 window, 0.8122 → 0.8332 per-recording.**

---

## The measurement problem, which came first

For most of this work the noise floor was larger than every effect being
measured. Three repeats of an identical config scored **0.6041 / 0.7790 /
0.8018** — standard deviation **0.108**.

Weight initialisation was not the cause; those runs already averaged three
models each. The base seed also chose **which bearings entered the inner
validation split**, and with only six healthy specimens, holding out a different
one changed what the model could learn about the class.

Removing the split was affordable because the schedule is a completed OneCycle
with early stopping off — there was nothing left for a validation set to decide.

| | Mean | Std |
|---|---|---|
| 1 validation bearing per class | 0.7283 | 0.1082 |
| **No validation split** | **0.7823** | **0.0018** |

### What that invalidated

Every comparison made before this point sampled from a ±0.11 distribution.
Four things reported as findings were not:

- *"patience 12 beats patience 20"* — single draws
- *"shorter schedules generalise better"* — single draws
- *"0.8063"* — a lucky draw, not a gain
- *"the 5-channel input is 10 points worse"* — **wrong twice over**: a single
  draw, *and* tested with weak augmentation. It is actually 2 points better.

**Measure the repeat-to-repeat spread before comparing anything.**
`scripts/experiments/repeat.py` exists for this and should be the first thing run after
any change to the training path.

---

## What does not work, now measured properly

With the noise floor at ±0.01, these are all within ~1 standard deviation:

| Knob | Result |
|---|---|
| Epochs 20 vs 40 vs 60 | 0.7837 / 0.7854 / — |
| Ensembling (1 vs 3 vs 5 seeds) | 0.7822 / 0.7837 |
| Test-time augmentation | +0.000 |
| `val_bearings_per_class` 1 vs 2 | starves the healthy class |

The model is **not under-tuned**. It sits where the data puts it.

---

## Three structural limits

### 1. The sample size is 29 bearings, not 16,127 windows

The seven windows of a recording are near-identical and the 560 windows of a
bearing are highly correlated. The independent unit is the **bearing**, there
are 29, and only **six are healthy** — leave-one-out leaves five to learn the
healthy class from. Any confidence interval computed on 16,127 rows is fiction.

### 2. The classes are entangled for a third of specimens

Taking each bearing's mean order spectrum and asking which class centroid it
sits closest to, excluding itself: **12 of 29 land on the wrong class**. K002
resembles nothing — its similarities are −0.121 / −0.052 / −0.082.

The CNN beats this crude test, but it is working on specimens that look
mislabelled at the level of their average spectrum.

### 3. Errors are all-or-nothing per bearing

Pooling N four-second observations of the same bearing:

| N | 3-class accuracy |
|---|---|
| 1 | 0.7629 |
| 5 | 0.7790 |
| 40 | 0.7826 |

**Saturates by N=5.** When the model is wrong about a bearing it is wrong about
*every* window of it, so more looks at the same machine cannot help. KI05 scored
0.005 and KA30 0.161 in the final run — not partially wrong, wholly wrong.

Those same bearings swing hard between runs (KA22 was 0.343 in one repeat and
0.637 in the next) while the pooled total stays put — the errors move around
among the difficult specimens rather than disappearing.

This matters for the product claim: **more observation time does not buy
accuracy on the 3-class task.** It does help the binary task (0.820 → 0.859).

---

## The protocol gap

Same model, same data, windows split at random instead of by bearing:

| Protocol | Accuracy | Macro-F1 |
|---|---|---|
| Stratified 5-fold over windows | 0.9865 | 0.9847 |
| Leave-one-bearing-out | 0.8018 | 0.7944 |

**18.5 points from protocol choice alone.**

Both are reported in the root `README.md`, each labelled with the protocol that
produced it. The 5-fold figure is what the Paderborn literature reports and is
the right number for comparing against published work; the leave-one-bearing-out
figure is the one that predicts behaviour on a bearing the model has never seen,
which is the only situation the deployed product is ever in.

Neither is wrong. Quoting either without naming the protocol is.

## What would close the remaining 1.7 points

1. **More bearings.** The binding constraint. Twenty more healthy specimens
   would do more than any architecture change. A data-collection decision, not
   an engineering one.
2. **A severity target instead of a location target.** Healthy / inner / outer
   asks *where*. The product asks *how bad*. The damage-profile PDFs carry an
   extent grade of 1–3, and the confusion matrix shows inner and outer race
   trading errors constantly — errors a severity ladder would not make.
3. **Understand the four failing specimens.** KI05, KA30, KA22 and K002 fail
   hard and consistently. Four bearings out of 29 is 14 % of the test set; if
   half of them are explicable, the target is met. This is more promising than
   any further sweep.

## What would not

More capacity, more epochs, longer windows, more ensembling. All tested under a
controlled noise floor; all neutral. With 29 bearings the model is not
capacity-limited, and every extra degree of freedom is something else for it to
overfit bearing identity with.
