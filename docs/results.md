# DriveSentinel v5 — results

Leave-one-bearing-out over all 29 bearings, pooled. Every bearing is
held out exactly once, so concatenating the folds gives one
out-of-sample prediction per window across the whole dataset.

> Per-**fold** macro-F1 and per-fold baselines are omitted deliberately:
> a single-bearing test set contains one class, so its macro-F1 caps at
> 1/3 and its majority baseline is always 1.0. Only the pooled set has
> all three classes in it.

## Headline

| Metric | Value |
|---|---|
| Window accuracy | **0.7979** |
| Macro-F1 | **0.7876** |
| Per-recording accuracy (majority vote) | **0.8382** |
| Majority-class baseline | 0.4153 |
| Windows / recordings | 16,127 / 2,306 |
| Per-bearing accuracy | 0.7983 ± 0.2852 (SEM 0.0530, n=29) |
| Sweep time | 2.5 min |

## Confusion matrix

Rows are truth, columns are prediction.

| | healthy | inner_race | outer_race | recall |
|---|---|---|---|---|
| **healthy** | 2,658 | 550 | 152 | 0.791 |
| **inner_race** | 764 | 4,878 | 428 | 0.804 |
| **outer_race** | 558 | 808 | 5,331 | 0.796 |

## By damage origin

Never pooled into one figure — v4 measured artificial→real transfer at
37.9 %. Healthy bearings join both groups as the reference class.

| Population | Bearings | Accuracy | Macro-F1 | Baseline |
|---|---|---|---|---|
| real | 17 | 0.8000 | 0.7952 | 0.3566 |
| artificial | 18 | 0.7935 | 0.7818 | 0.3884 |

## Per-bearing

| Bearing | Class | Origin | Accuracy | Recording acc | Epochs |
|---|---|---|---|---|---|
| `KI05` | inner_race | artificial | 0.0072 | 0.0000 | 19 |
| `K002` | healthy | none | 0.0625 | 0.0000 | 19 |
| `KA30` | outer_race | real | 0.2411 | 0.1000 | 19 |
| `KA22` | outer_race | real | 0.3537 | 0.4375 | 19 |
| `KA03` | outer_race | artificial | 0.5696 | 0.4750 | 19 |
| `KI03` | inner_race | artificial | 0.5946 | 0.7375 | 19 |
| `KI07` | inner_race | artificial | 0.7549 | 0.8375 | 19 |
| `K005` | healthy | none | 0.7661 | 0.9500 | 19 |
| `KA15` | outer_race | real | 0.8262 | 0.9750 | 19 |
| `KI14` | inner_race | real | 0.8655 | 1.0000 | 19 |
| `KA07` | outer_race | artificial | 0.8714 | 0.9500 | 19 |
| `KI17` | inner_race | real | 0.8730 | 0.9750 | 19 |
| `KA05` | outer_race | artificial | 0.8748 | 0.9875 | 19 |
| `KI21` | inner_race | real | 0.8943 | 1.0000 | 19 |
| `KI04` | inner_race | real | 0.9000 | 0.9375 | 19 |
| `KA08` | outer_race | artificial | 0.9094 | 1.0000 | 19 |
| `K004` | healthy | none | 0.9214 | 1.0000 | 19 |
| `KA16` | outer_race | real | 0.9589 | 1.0000 | 19 |
| `KA04` | outer_race | real | 0.9693 | 0.9873 | 19 |
| `KI08` | inner_race | artificial | 0.9785 | 0.9875 | 19 |
| `KI16` | inner_race | real | 0.9822 | 1.0000 | 19 |
| `KA06` | outer_race | artificial | 0.9857 | 1.0000 | 19 |
| `KA09` | outer_race | artificial | 0.9946 | 1.0000 | 19 |
| `K001` | healthy | none | 0.9982 | 1.0000 | 19 |
| `K006` | healthy | none | 0.9982 | 1.0000 | 19 |
| `KI01` | inner_race | artificial | 0.9982 | 1.0000 | 19 |
| `K003` | healthy | none | 1.0000 | 1.0000 | 19 |
| `KA01` | outer_race | artificial | 1.0000 | 1.0000 | 19 |
| `KI18` | inner_race | real | 1.0000 | 1.0000 | 19 |

## Evidence accumulation

Accuracy when N four-second observations of the same bearing are
pooled by averaging their per-window softmax. Still leave-one-
bearing-out — every probability is out-of-sample. A lift makes dozens
of trips a day, so a health verdict does not have to be called from a
single look at the machine.

| Observations pooled | Accuracy |
|---|---|
| 1 | 0.8376 |
| 2 | 0.8495 |
| 5 | 0.8584 |
| 10 | 0.8621 |
| 20 | 0.8624 |
| 40 | 0.8626 |

## INT8 export

- Weights: 27,024 int8, 179 int32 biases
- INT8 vs float argmax agreement: **99.90 %** (bar: ≥ 98 %), n=2,048
- Max |logit| difference: 4.0409

| Layer | Terms | Peak \|acc\| | Bits | int32 |
|---|---|---|---|---|
| conv1 | 45 | 733,804 | 21 | ok |
| conv2 | 112 | 1,841,594 | 22 | ok |
| conv3 | 160 | 2,636,875 | 23 | ok |
| conv4 | 192 | 3,151,891 | 23 | ok |
| fc | 64 | 1,041,698 | 21 | ok |

---

Generated from `artifacts/runs/lobo_summary.json`. DSP contract: window 1.0s hop 0.5s | 5 x 512 bins over 0-16.0 orders (0.03125 orders/bin) | classes ['healthy', 'inner_race', 'outer_race']
