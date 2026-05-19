# FootContact-EventTiming

Pose-window baseline for foot-contact event timing.

This repo is intentionally separate from `Vision-to-Stability`: the baseline
there still predicts frame-wise contact, while this project tests a different
target. For each contact onset or departure event, it crops a fixed window
around the event and trains a model to regress the event offset inside that
window.

During validation and testing, each event is evaluated with several different
window offsets. This avoids the trivial setup where every event is always at the
center of the window and a constant prediction can score well.

## Why This Exists

The frame-wise FootFormer baseline can produce reasonable contact labels while
still missing contact/departure timing by many frames. This repo tests whether
predicting event time directly is a better target before adding new modalities
such as audio.

## Install

```bash
git clone https://github.com/luofugui/FootContact-EventTiming.git
cd FootContact-EventTiming
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## PSU-TMM100 Pose Event Offset Training

Edit `configs/psu_pose_event_offset_50fps.yaml` so these paths point to your
server data:

```yaml
data:
  chunk_dir: /path/to/Chunked_PSU_Real50fps/BODY25_1fps_distribution
  foot_mask_path: /path/to/Vision-to-Stability/assets/foot_mask_nans.npy
```

Debug with a small number of chunks:

```bash
python -m scripts.train_event_offset \
  --config configs/psu_pose_event_offset_50fps.yaml \
  --subject 1 \
  --limit-files 10
```

Run LOSO for one subject:

```bash
python -m scripts.train_event_offset \
  --config configs/psu_pose_event_offset_50fps.yaml \
  --subject 1
```

Run all ten LOSO folds:

```bash
python -m scripts.train_event_offset \
  --config configs/psu_pose_event_offset_50fps.yaml \
  --subject all
```

Outputs are written under `results/pose_event_offset/<config_name>/`, including
per-subject checkpoints, `event_offset_results.csv`, and `summary.json`.

The default config preloads joint chunks into shared-memory tensors. This uses
more CPU memory, but avoids repeatedly rebuilding 51-frame windows from pickle
objects and keeps the GPU fed more consistently.

Training also saves centered event-window plots, using GT contact/departure
timestamps and a +/-0.5s window around each event. The training target remains
`event_frame - window_start`: with the default jittered windows this target
varies across samples; if `centered_window: true`, the target is always the
middle frame and is only useful as a literal sanity check, not as a meaningful
regression task.

Plot the final LOSO CSV without retraining:

```bash
python -m scripts.plot_event_offset_results \
  --csv results/pose_event_offset/psu_pose_event_offset_50fps/event_offset_results.csv
```

## Analyze Existing FootFormer PKL Outputs

You can also inspect frame-wise model outputs before training the event-offset
model:

```bash
python -m scripts.analyze_pkl_events \
  --dir /path/to/eval/output \
  --fps 50 \
  --threshold 0.5 \
  --out-prefix psu_50fps
```

For logits, use `--threshold 0.0`; for probabilities, use `--threshold 0.5`.
This writes an event-timing CSV and average onset/departure window plots.

## UnderPressure Event-Time Regression

This variant keeps the FootFormer-style pose embedder and temporal transformer,
removes the temporal attention mask, and trains directly on event time error.
Each sample is a 1s sliding pose window sampled to 30fps from the synchronized
UnderPressure sequence. Event presence and event time targets are extracted
from the 100Hz contact timeline.

```text
input shape = [T, 23, 4]
output shape = [4]
```

The first version uses four foot-level event times:

```text
left_contact, left_departure, right_contact, right_departure
```

The model has two main outputs:

```text
event_presence = [4]
event_time = [4]
```

`event_presence` predicts whether each event occurs in the window. `event_time`
is a normalized offset inside the current window and is trained only for events
that are present. The default `model.time_head: direct` regresses this vector
directly. `model.time_head: soft_argmax` is also available for diagnostics; it
produces frame-wise temporal scores and converts them to a continuous offset
with soft-argmax, while still using the same time loss.

Run a quick debug pass for one LOSO fold:

```bash
python -m scripts.train_underpressure_event_time \
  --config configs/underpressure_event_time_30fps.yaml \
  --subject S1 \
  --limit-files 10
```

Run one LOSO fold:

```bash
python -m scripts.train_underpressure_event_time \
  --config configs/underpressure_event_time_30fps.yaml \
  --subject S1
```

Run a tiny same-subset overfit diagnostic:

```bash
python -m scripts.train_underpressure_event_time \
  --config configs/underpressure_event_time_30fps.yaml \
  --subject S1 \
  --limit-files 1 \
  --tiny-overfit \
  --overfit-windows 512 \
  --overfit-epochs 100
```

The tiny-overfit log prints `time_loss`, `presence_loss`, a constant-center
baseline MAE, and the predicted/target time standard deviations. If
`pred_std` stays near zero while `target_std` is nonzero, the time head is
collapsing to an almost constant offset.

To isolate the time objective, disable the presence loss for the same test:

```bash
python -m scripts.train_underpressure_event_time \
  --config configs/underpressure_event_time_30fps.yaml \
  --subject S1 \
  --limit-files 1 \
  --tiny-overfit \
  --overfit-time-only \
  --overfit-lambda-time 100 \
  --overfit-windows 512 \
  --overfit-epochs 50
```

Run all ten LOSO folds:

```bash
python -m scripts.train_underpressure_event_time \
  --config configs/underpressure_event_time_30fps.yaml
```

The output directory follows the original FootFormer layout more closely:

```text
results/underpressure_event_time/underpressure_event_time_30fps/
  config.yaml
  checkpoint/S1/best.pth
  checkpoint/S1/final.pth
  eval/subjectS1.json
  eval/output/subjectS1_output.pkl
  log/S1.log
  log/S1_history.csv
  Tensorboard/S1/
  visualizations/S1/learning_curves/train_losses.png
  visualizations/S1/learning_curves/val_losses.png
```

The eval pkl stores direct time-regression outputs:

```text
predictions/event_time
predictions/event_presence
predictions/event_time_scores
predictions/event_time_prob
targets/event_time
targets/event_valid
event_names
input_fps
label_fps
window_sec
```
