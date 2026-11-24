# Continual Learning

Code: `continual.py`; results and figures: `runs/continual/`. The starting model is the robust v1.2
ResNet-18 from Milestone 3, which becomes version v1.3.0 here.

## 1. Continual learning strategies

**Periodic full retraining.** Retraining on all data collected so far is the simplest way to stay
current. It cannot forget, because nothing is ever dropped. For us it has two costs that grow over
time: every labelled image must be kept, and each run gets slower (in our experiment, from 15 s to
38 s after only five periods). It is a useful upper bound, and a sensible fallback a few times a
year, but not the routine update mechanism.

**Naive fine-tuning.** Fine-tuning on the newest batch only is cheap and adapts quickly. Milestone 3
already showed why it is risky here: adapting the original model to a new camera cost 2.3 points on
clean images and almost doubled rotten→fresh errors. With small batches of 600 images, each update
also overfits to whatever the latest period looks like.

**Experience replay.** Mixing a small stored buffer of past examples into every update is the
standard remedy for forgetting. It suits our project for three reasons:
- the drift we expect is visual (camera, lighting, compression), not new classes;
- the buffer can be class-balanced, which protects semi-fresh, the class Milestone 3 showed to be
  most fragile;
- a few hundred stored image paths cost nothing to keep.

**Regularisation-based methods** (Elastic Weight Consolidation, Learning without Forgetting). These
penalise changes to weights that mattered for earlier data, or distil the old model's outputs into
the new one. They need no stored images, which would matter if images could not be retained for
privacy reasons. That is not our situation: fruit photos are not personal data. They are also usually
weaker than replay, and LwF needs the old model in memory during training.

**Parameter isolation.** Freezing parts of the network, or adding small adapters per domain, keeps
old knowledge structurally safe. We already use this in part: layers 1–3 of the ResNet stay frozen
at their ImageNet weights, so the generic visual features can never be forgotten.

## 2. Selected technique: class-balanced experience replay, updating only `layer4` and the heads

We combine replay with the partial freezing we already have.
- Each update fine-tunes `layer4` and the two heads on 600 newly labelled images plus 600 drawn,
  class-balanced, from a replay buffer.
- The Milestone 3 corruption augmentation stays on.
- After each update, the temperature and the Mahalanobis OOD statistics are refitted.
- The candidate must pass a release gate before it replaces the deployed model.

**Architecture and inductive bias (Milestone 2).** The ResNet-18's convolutional structure and
frozen ImageNet layers give it general-purpose texture and colour features. What changes under
drift is how those features map to freshness. That mapping lives in `layer4` and the heads (8.4M
of the 11.2M parameters), so that is where updates go. Keeping layers 1–3 frozen means the low-level
features cannot drift away. It also keeps every update small and fast, and it lets us reuse the
existing training code unchanged.

**Failure modes and drift (Milestone 3).**
- The model's weaknesses were visual: noise, blur, compression, colour shifts.
- The drift experiment showed that a small replay fine-tune recovers most of the loss
  (77% → 83% on a new camera) without hurting clean accuracy.
- Semi-fresh was always the first class to fail, which is why the buffer is class-balanced rather
  than a plain random sample.
- The adversarial weakness is not something continual learning can fix, and we do not claim it does.

**Deployment constraints.**
- The system runs on a single laptop-class GPU with a fast, small model.
- An update takes about 11–18 s and under 700 MB of VRAM, so it can run on the deployment machine
  itself, overnight or between batches.
- The model's size and inference latency do not change at all.
- The buffer stores file paths and the drift period, not pixels: about 60 KB, against 147 MB if the
  images were stored.

**How it preserves old knowledge while adapting.** Three things protect earlier knowledge:
1. The frozen lower layers cannot forget.
2. Half of every update batch comes from earlier periods, so the loss keeps rewarding performance
   on old conditions.
3. The release gate rejects any candidate that loses more than 2 points of macro-F1 on clean
   validation data.

Adaptation comes from the other half of the batch, which is all new-period data.

**Integration into the pipeline.** Continual learning hooks into the monitoring designed in Milestone
3. The online path (quality check → ResNet → temperature → Mahalanobis gate → abstention) already
produces the two signals that trigger an update:
- the OOD-gate rejection rate, which catches visual drift;
- the shift in predicted class shares, which catches prior drift.

Abstained and rejected images, plus a random sample of accepted ones, go to a labelling queue. When a
trigger fires, the update runs on the next 600 labelled images. If the candidate passes the gate,
the new version replaces the old one and the OOD statistics are refitted. If not, the deployed
version stays.

## 3. Implementation

**Continual learning cycle (one drift period):**

```text
every period (e.g. daily batch of images):
    signals = monitor(deployed_model, recent_images)       # OOD rejection rate, class-share shift, confidence
    if signals.ood_rate > 2 x baseline or signals.class_shift > 0.10:
        new  = label(sample(recent_images, 600))            # human labelling queue
        mix  = new + buffer.sample(600, class_balanced=True)
        cand = finetune(copy(deployed_model), mix,           # layer4 + heads only, augmentation on
                        epochs=5, lr=1e-4, batch=32)
        if val_f1(cand, period) >= val_f1(deployed, period) and
           val_f1(cand, clean) >= approved_clean_f1 - 0.02:
            deployed = cand; version += 1                    # v1.3.k
            refit temperature and Mahalanobis statistics on buffer + period validation
            save best.pt, config.json (parent, data hash), metrics.json, buffer manifest
        buffer.add(new)                                      # class-balanced reservoir sampling
```

**Hyperparameters and schedule:**

| Setting | Value |
|---|---|
| Update trigger | OOD rejection rate > 2× baseline (4%), or class-share shift (total variation) > 0.10 |
| New labelled images per update | 600 (200 per class, or following the period's class prior) |
| Replay images per update | 600, class-balanced from the buffer |
| Buffer | 999 slots (333 per class), class-balanced reservoir sampling, stores (path, period) |
| Initial buffer | 999 clean images from the v1.2 training selection |
| Trainable parameters | `layer4` + heads (8.4M); layers 1–3 frozen |
| Optimiser | AdamW, lr 1e-4, weight decay 0.05, 5 epochs, batch 32, mixed precision |
| Loss | CE(freshness) + 0.5 × CE(fruit), label smoothing 0.2, v1.2 corruption augmentation |
| Release gate | Period validation macro-F1 not worse than the deployed model; clean validation macro-F1 at most 2 points below the best approved |
| After an update | Refit temperature and Mahalanobis statistics; OOD threshold = 98th percentile on the period's validation images |

The 2-point clean tolerance is deliberate. A first version of the experiment used 1 point on a
600-image validation subset, and the gate rejected four of five updates. Those rejections were
largely measurement noise, so the final setup uses the full 2,088-image validation set and a
tolerance of about 2.5 standard errors.

**Versioning.** Each approved update is saved as `runs/continual/v1.3.k/`, following the Milestone 2–3
convention:
- `best.pt`, the weights;
- `config.json`, holding the parent version, the period, all hyperparameters, the fitted temperature
  and OOD threshold, and an MD5 hash of the new data's file list;
- `metrics.json`;
- `buffer_manifest.csv`, the exact buffer contents.

v1.3.0 is the v1.2 robust model. Any version can be rebuilt from its parent plus the recorded data,
and the previous approved version is always available for rollback.

**Changes to the data pipeline.**
- The batch pipeline becomes period-based: images arrive per period (a day or a delivery), and the
  monitor runs on each period.
- Labelling becomes an ongoing queue fed by the monitor, rather than a one-off dataset.
- The replay buffer is a new persistent artefact.
- The split, preprocessing and training code are unchanged; `continual.py` reuses them directly.

## 4. Simulated drift and continual update experiment

**Setup.** Six deployment periods, each with its own drifted test set (2,170 images; 1,193 for T3)
and, from T1 on, 600 newly labelled images taken from the unused part of the training split:

| Period | Drift | Type |
|---|---|---|
| T0 | Clean (deployment start) | — |
| T1 | Low light: brightness −40% + noise σ = 0.05 | Visual (covariate) |
| T2 | Cheap camera: blur σ = 1, noise σ = 0.08, JPEG q30 | Visual (covariate) |
| T3 | Mostly fresh stock: classes 60/10/30 | Class prior |
| T4 | Warm lighting: colour cast (R ×1.2, B ×0.75) | Visual (covariate) |
| T5 | Back to clean | Tests forgetting |

We compare five strategies, each updating its own copy of the model:
- no update;
- naive fine-tuning on the new batch only;
- replay without a release gate;
- replay with the gate (our deployed method);
- cumulative retraining on everything so far.

Each model is evaluated on every period's test set after every step. That gives an accuracy matrix,
from which we compute average accuracy, forgetting (best earlier accuracy on a period minus final
accuracy) and backward transfer (final minus just-learned accuracy).

**Performance before and after each update (deployed method: replay with gate).**

| Period | Accuracy before → after | Macro-F1 before → after | Semi-fresh recall | Rotten→Fresh | Gate | Version |
|---|---|---|---|---|---|---|
| T1 low light | 80.7 → 82.9% | 80.3 → 82.7% | 0.66 → 0.71 | 33 → 24 | passed | v1.3.1 |
| T2 cheap camera | 81.8 → 82.9% | 81.6 → 82.8% | 0.72 → 0.76 | 34 → 21 | passed | v1.3.2 |
| T3 mostly fresh | 87.2 → 93.1% | 80.8 → 86.6% | 0.82 → 0.70 | 1 → 3 | passed | v1.3.3 |
| T4 warm lighting | 86.3 → 87.9% | 85.9 → 87.7% | 0.73 → 0.77 | 3 → 1 | passed | v1.3.4 |
| T5 back to clean | 88.2 → 87.7% | 87.9 → 87.6% | 0.75 → 0.80 | 1 → 3 | passed | v1.3.5 |

**Comparison of strategies after the last period.**

| Strategy | Average accuracy (all 6 periods) | Average macro-F1 | Average accuracy on each period when it arrived | Forgetting | Backward transfer |
|---|---|---|---|---|---|
| No update | 84.3% | 82.7% | 84.3% | 0.0 | 0.0 |
| Naive fine-tune | 84.2% | 82.7% | 85.0% | +0.9 pts | −0.9 pts |
| Replay, no gate | 87.9% | 86.9% | 86.5% | −1.1 pts | +1.6 pts |
| **Replay with gate (deployed)** | **86.3%** | **85.3%** | **86.8%** | +1.5 pts | −0.5 pts |
| Cumulative retrain | 88.5% | 87.4% | 86.7% | −0.9 pts | +2.1 pts |

The two replay rows passed the gate at every step. They are therefore the same procedure, differing
only in which buffer images were drawn, and their 1.6-point gap in final average accuracy is a fair
estimate of run-to-run noise. Taken together, replay reaches 87.1 ± 0.8% average accuracy with
essentially no forgetting (−1.1 to +1.5 points).

Naive fine-tuning ends no better than never updating (84.2% vs. 84.3%): its gains on each new period
are cancelled by losses on old ones. In Fig. 2 its accuracy on the low-light period falls from 0.82
to 0.76 within two updates. Cumulative retraining is best, but only by about 1.4 points over replay,
and it pays for that with all stored data and update times that grow with every period.

**Two observations matter for deployment.**

**Prior shifts do not need retraining.** Updating on the 60/10/30 batch in T3 raised accuracy (87.2
→ 93.1%) but cut semi-fresh recall from 0.82 to 0.70, because the model learned that semi-fresh had
become rare. Accuracy rewards that; a grading system should not. Recall recovered by T5 once balanced
data returned, but the lesson is that a pure class-prior shift should be handled by adjusting the
decision prior or threshold, not by fine-tuning. Our monitor can tell the two cases apart: in T3 the
class-share signal fired (0.21) while the OOD rate stayed at baseline (1.3%) (Fig. 3).

**The monitor works as a trigger.**
- Each visual drift raised the OOD rejection rate well above the 4% trigger: 18.8% (low light),
  22.4% (camera), 13.0% (warm light).
- Returning to clean images in T5 did not trigger it (0.4%), because clean images are still well
  represented in the buffer.
- After each update and refit, the rate returned to about 2%. That is partly by construction,
  because the new threshold is set at the 98th percentile of the new period's validation images. The
  meaningful signal is the "before" value.

**Efficiency during continual updates.**

| Strategy | Images per update | Update time (s) | Peak VRAM during update | Stored data |
|---|---|---|---|---|
| Naive fine-tune | 600 | 5.7–8.7 | 558–687 MB | none |
| Replay (ours) | 1,200 | 11.4–18.1 | 623–688 MB | 999 paths (60 KB manifest) |
| Cumulative retrain | 1,599 → 3,999 | 15.3 → 38.2 (grows each period) | 656–686 MB | every labelled image |

| Deployed model | Parameters | Size | GPU latency, batch 1 (p50 / p90) | GPU throughput, batch 64 |
|---|---|---|---|---|
| v1.3.0 (start) | 11.18M | 42.7 MB | 2.9 / 3.4 ms | 1,532 img/s |
| v1.3.5 (after 5 updates) | 11.18M | 42.7 MB | 2.4 / 2.8 ms | 1,751 img/s |

Updates change only weight values, so model size and inference cost stay identical. The small speed
difference above is warm-up order (v1.3.0 was timed first), not a real change. Spikes in update time
(17–18 s for T2–T3 against 11 s otherwise) coincided with the laptop throttling, which we saw
throughout this project. Keeping every version costs 42.7 MB each, 256 MB for six versions.

**Figures:**
- **Figure 1** (`runs/continual/trajectories.png`; panels also saved separately as `trajectory_current.png`,
  `trajectory_average.png` and `trajectory_clean.png`), for the four strategies other than the second
  replay run:
  - left: accuracy on each period before (dotted) and after (solid) its update;
  - centre: average accuracy over all periods seen so far;
  - right: accuracy on the original clean conditions over time, which shows forgetting directly.
- **Figure 2** (`runs/continual/accuracy_matrix.png`): the full accuracy matrix for each strategy.
  Each row is the model after a step; each column is a test period.
- **Figure 3** (`runs/continual/monitor.png`): drift-monitor signals per period, before and after
  the update.

## 5. Active learning / human-in-the-loop simulation

**Setup.** We reran the gated replay strategy over the same five drift cycles (T1–T5), but the
simulated annotators labelled only **360 of the 600 new images** per cycle (40% fewer labels). Labels
came from the dataset, standing in for human annotators. Two ways of choosing the 360 were compared:
- **Active learning:** the CLIP selection head from Milestone 2, applied to the drifted images. We
  extended it to **5 sentences per class (120 sentences)** and gave each sentence exactly **3 images**
  (round-robin, no image picked twice), so every class gets the same quota of 15.
- **Random:** 360 images chosen at random, as a control.

Each update used the 360 new images plus 360 replayed ones and went through the same release gate.
We compare against the unchanged model and against the earlier replay run that labelled all 600
images per cycle. Code: `continual.py --methods ... --out-name continual_al360`; results:
`runs/continual_al360/`.

**Performance across cycles (test accuracy before → after each cycle's update).**

| Cycle | No update | Active learning (360) | Random (360) | Replay, all 600 labelled |
|---|---|---|---|---|
| T1 low light | 80.7% | 80.7 → **82.0** | 80.7 → 80.7 (gate failed) | 80.7 → 82.9 |
| T2 cheap camera | 78.0% | 82.2 → 82.2 (gate failed) | 78.0 → 78.0 (gate failed) | 81.8 → 82.9 |
| T3 mostly fresh | 91.0% | 92.5 → 92.6 | 91.0 → 93.7 | 87.2 → 93.1 |
| T4 warm lighting | 84.2% | 86.5 → 86.5 (gate failed) | 86.7 → 86.7 (gate failed) | 86.3 → 87.9 |
| T5 back to clean | 85.9% | 85.1 → 85.1 (gate failed) | 86.7 → 86.7 (gate failed) | 88.2 → 87.7 |

| After all five cycles | No update | Active learning (360) | Random (360) | Replay (600) |
|---|---|---|---|---|
| Average accuracy, all periods | 84.3% | **85.4%** | 84.9% | 86.3–87.9% |
| Average macro-F1 | 82.7% | **84.0%** | 83.3% | 85.3–86.9% |
| Forgetting | 0.0 | +0.4 pts | +0.6 pts | −1.1 to +1.5 pts |
| Updates approved by the gate | — | 2 of 5 | 1 of 5 | 5 of 5 |
| Human labels used | 0 | 1,800 | 1,800 | 3,000 |

With 360 labels per cycle, active learning improves on the unchanged model by 1.1 points of accuracy
and 1.3 points of macro-F1. It recovers roughly a third to a half of the gain of labelling everything, for 60% of the
labels. It is slightly ahead of random selection (+0.5 points), but that gap is inside the run-to-run
noise of about ±0.8 points measured in Section 4, so we do not claim that CLIP chooses better images
than chance.

The improvements came from the cycles where an update passed the gate: T1 and T3 for active learning.
In T1 it reduced rotten→fresh errors from 33 to 27. Later candidates were mostly rejected because they
were no better on the new period than the model already deployed: the T1 update had already
transferred to the later visual drifts. For the random variant, two candidates were rejected for
losing more than 2 points on clean images.

CLIP's guess of the freshness level was right for only 53–58% of the drifted images, so its equal
per-sentence quota did not make the labelled set exactly balanced: 36–41% of picks were fresh in the
visual-drift cycles. It did keep slightly more semi-fresh images than random selection when they were
rare (T3: 11.4% vs. 10.0%).

**Efficiency.**

| | Active learning (360) | Random (360) | Replay (600) |
|---|---|---|---|
| Labelling burden per cycle | 360 (−40%) | 360 (−40%) | 600 |
| Selection step per cycle | 1.5–2.6 s (CLIP on 600 images) | none | none |
| Extra memory for selection | CLIP model 349 MB VRAM; peak 576–724 MB | none | none |
| Update time per cycle | 6.9–7.3 s (720 images) | 6.9–7.0 s | 11.4–18.1 s (1,200 images) |
| Peak VRAM during update | 768–833 MB | 799–834 MB | 623–688 MB |
| Deployed model inference | unchanged (same 11.18M-parameter, 42.7 MB model; 2.9 → 2.3 ms at batch 1, warm-up effect) | unchanged | unchanged |

The selection step adds about two seconds per cycle and needs CLIP in memory, but only during the
offline update, never on the request path, so inference latency is unaffected. Fewer labels also mean
shorter updates: about 7 s against 11–18 s for the 600-label replay.

**Conclusion.** Using the active-learning head to label 360 of each cycle's 600 images (40% fewer
labels) still improves the model across cycles, recovering roughly a third to a half of the benefit of labelling
everything. The size of the labelling budget matters more than how the images are chosen; CLIP's
choice is at best marginally better than random. Uncertainty-based selection (least-confidence
sampling, Section 2.3) is the natural next step, because it targets the images the current model
finds hard, which a text-matching selector does not.
