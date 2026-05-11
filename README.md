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
