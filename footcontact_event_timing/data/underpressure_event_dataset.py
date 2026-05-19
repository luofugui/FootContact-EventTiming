import random
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import Dataset

from footcontact_event_timing.data.contact_events import get_event_indices


def list_underpressure_files(data_root):
    root = Path(data_root)
    return sorted(root.glob("S*/preprocessed/*.pth"))


def subject_from_path(path):
    for part in Path(path).parts:
        if part.startswith("S") and part[1:].isdigit():
            return part
    raise ValueError(f"Could not infer subject from path: {path}")


def split_underpressure_files(data_root, test_subjects, train_val_split=0.9, seed=0):
    files = list_underpressure_files(data_root)
    test_subjects = set(str(s) for s in test_subjects)
    train_val = [p for p in files if subject_from_path(p) not in test_subjects]
    test = [p for p in files if subject_from_path(p) in test_subjects]
    rng = random.Random(seed)
    rng.shuffle(train_val)
    split = int(len(train_val) * train_val_split)
    return train_val[:split], train_val[split:], test


class UnderPressureEventWindowDataset(Dataset):
    """Sliding pose windows with direct foot contact event-time targets.

    The preprocessed UnderPressure files are aligned to the insole/contact
    timeline. We treat that timeline as the label timeline and sample a
    lower-frame-rate pose window for the model input.
    """

    def __init__(
        self,
        files,
        split="train",
        input_fps=30,
        label_fps=100,
        window_sec=1.0,
        stride_sec=0.1,
        pose_key="positions",
        contact_key="contacts",
        joint_dim=4,
        event_names=None,
        max_cached_sequences=64,
        preload=True,
        share_memory=True,
        seed=0,
        verbose=True,
    ):
        self.files = [Path(p) for p in files]
        self.split = split
        self.input_fps = int(input_fps)
        self.label_fps = int(label_fps)
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.pose_key = pose_key
        self.contact_key = contact_key
        self.joint_dim = int(joint_dim)
        self.event_names = list(
            event_names
            if event_names is not None
            else ["left_contact", "left_departure", "right_contact", "right_departure"]
        )
        if len(self.event_names) != 4:
            raise ValueError("The first direct-time regression version expects exactly 4 events.")

        self.input_frames = int(round(self.window_sec * self.input_fps)) + 1
        self.label_window_frames = int(round(self.window_sec * self.label_fps)) + 1
        self.stride_frames = max(1, int(round(self.stride_sec * self.label_fps)))
        self.max_cached_sequences = int(max_cached_sequences)
        self.preload = bool(preload)
        self.share_memory = bool(share_memory)
        self.verbose = verbose
        self.sequence_cache = OrderedDict()
        self.pose_tensors = {}
        self.contact_tensors = {}
        self.windows = self._build_windows(seed)
        if not self.windows:
            raise ValueError(f"No UnderPressure event-time windows found for split={split}.")

    @property
    def num_event_classes(self):
        return len(self.event_names)

    @property
    def window_frames(self):
        return self.input_frames

    def _load_raw(self, seq_idx):
        if seq_idx in self.sequence_cache:
            self.sequence_cache.move_to_end(seq_idx)
            return self.sequence_cache[seq_idx]
        raw = torch.load(self.files[seq_idx], map_location="cpu", weights_only=False)
        self.sequence_cache[seq_idx] = raw
        if len(self.sequence_cache) > self.max_cached_sequences:
            self.sequence_cache.popitem(last=False)
        return raw

    def _extract_pose(self, raw):
        for key in [self.pose_key, "positions", "skeleton", "joints"]:
            if key in raw:
                pose = torch.as_tensor(raw[key]).float()
                break
        else:
            raise KeyError(f"Could not find pose key in {raw.keys()}")

        if pose.ndim != 3:
            raise ValueError(f"Expected pose shape [T, J, D], got {tuple(pose.shape)}")
        if pose.shape[-1] < self.joint_dim:
            pad = torch.ones(*pose.shape[:-1], self.joint_dim - pose.shape[-1])
            pose = torch.cat([pose, pad], dim=-1)
        elif pose.shape[-1] > self.joint_dim:
            pose = pose[..., : self.joint_dim]
        return torch.nan_to_num(pose)

    def _extract_contacts(self, raw):
        for key in [self.contact_key, "contacts", "contact"]:
            if key in raw:
                contacts = torch.as_tensor(raw[key])
                break
        else:
            raise KeyError(f"Could not find contact key in {raw.keys()}")

        contacts = torch.nan_to_num(contacts.float()) >= 0.5
        if contacts.ndim == 3:
            if contacts.shape[1] < 2:
                raise ValueError(f"Expected at least two feet in contacts, got {tuple(contacts.shape)}")
            return contacts[:, :2, :].any(dim=2).float()
        if contacts.ndim == 2:
            if contacts.shape[1] == 2:
                return contacts.float()
            if contacts.shape[1] >= 4:
                left = contacts[:, :2].any(dim=1)
                right = contacts[:, 2:4].any(dim=1)
                return torch.stack([left, right], dim=1).float()
        raise ValueError(f"Expected contacts [T, 2, R], [T, 2], or [T, >=4], got {tuple(contacts.shape)}")

    def _get_pose_contacts(self, seq_idx):
        if seq_idx in self.pose_tensors:
            return self.pose_tensors[seq_idx], self.contact_tensors[seq_idx]
        raw = self._load_raw(seq_idx)
        pose = self._extract_pose(raw)
        contacts = self._extract_contacts(raw)
        return pose, contacts

    def _foot_events(self, contacts):
        events = []
        for foot_idx in range(2):
            onsets, departures = get_event_indices(contacts[:, foot_idx].numpy() >= 0.5)
            events.append((torch.as_tensor(onsets, dtype=torch.long), torch.as_tensor(departures, dtype=torch.long)))
        return events

    def _target_for_window(self, foot_events, start, end):
        target_time = torch.zeros(self.num_event_classes, dtype=torch.float32)
        event_valid = torch.zeros(self.num_event_classes, dtype=torch.float32)
        center = start + (end - start - 1) / 2.0
        denom = max(end - start - 1, 1)

        for foot_idx, (onsets, departures) in enumerate(foot_events):
            for event_type, frames in [(0, onsets), (1, departures)]:
                cls = foot_idx * 2 + event_type
                in_window = frames[(frames >= start) & (frames < end)]
                if in_window.numel() == 0:
                    continue
                nearest = in_window[(in_window.float() - center).abs().argmin()].float()
                target_time[cls] = (nearest - start) / denom
                event_valid[cls] = 1.0

        return target_time, event_valid

    def _build_windows(self, seed):
        windows = []
        rng = random.Random(seed)
        if self.verbose:
            print(f"[{self.split}] scanning {len(self.files)} UnderPressure sequences...", flush=True)

        for seq_idx in range(len(self.files)):
            raw = self._load_raw(seq_idx)
            pose = self._extract_pose(raw)
            contacts = self._extract_contacts(raw)
            if pose.shape[0] != contacts.shape[0]:
                n = min(pose.shape[0], contacts.shape[0])
                pose = pose[:n]
                contacts = contacts[:n]
            if pose.shape[0] < self.label_window_frames:
                continue
            if self.preload:
                if self.share_memory:
                    pose = pose.contiguous().share_memory_()
                    contacts = contacts.contiguous().share_memory_()
                self.pose_tensors[seq_idx] = pose
                self.contact_tensors[seq_idx] = contacts

            foot_events = self._foot_events(contacts)
            last_start = pose.shape[0] - self.label_window_frames
            for start in range(0, last_start + 1, self.stride_frames):
                end = start + self.label_window_frames
                target_time, event_valid = self._target_for_window(foot_events, start, end)
                if event_valid.sum() == 0:
                    continue
                windows.append(
                    {
                        "seq_idx": seq_idx,
                        "start": int(start),
                        "end": int(end),
                        "target_time": target_time,
                        "event_valid": event_valid,
                    }
                )

            if self.verbose and ((seq_idx + 1) % 25 == 0 or seq_idx + 1 == len(self.files)):
                print(
                    f"[{self.split}] {seq_idx + 1}/{len(self.files)} sequences -> "
                    f"{len(windows)} event-time windows",
                    flush=True,
                )

        if self.split == "train":
            rng.shuffle(windows)
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = self.windows[idx]
        pose, _ = self._get_pose_contacts(item["seq_idx"])
        sample_positions = torch.linspace(item["start"], item["end"] - 1, self.input_frames)
        sample_indices = sample_positions.round().long().clamp(0, pose.shape[0] - 1)
        return {
            "joint": pose[sample_indices],
            "target_time": item["target_time"].clone(),
            "event_valid": item["event_valid"].clone(),
            "window_start_frame": torch.tensor(item["start"], dtype=torch.long),
            "window_end_frame": torch.tensor(item["end"], dtype=torch.long),
        }
