import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
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
    """Centered event windows with per-frame heel/toe on/off heatmap targets."""

    def __init__(
        self,
        files,
        split="train",
        fps=30,
        half_window_sec=0.5,
        pose_key="positions",
        contact_key="contacts",
        joint_dim=4,
        event_sigma_frames=1.0,
        contact_channel_names=None,
        max_cached_sequences=64,
        preload=True,
        share_memory=True,
        seed=0,
        verbose=True,
    ):
        self.files = [Path(p) for p in files]
        self.split = split
        self.fps = int(fps)
        self.half_window_frames = int(round(float(half_window_sec) * self.fps))
        self.window_frames = 2 * self.half_window_frames + 1
        self.pose_key = pose_key
        self.contact_key = contact_key
        self.joint_dim = int(joint_dim)
        self.event_sigma_frames = float(event_sigma_frames)
        self.contact_channel_names = list(
            contact_channel_names
            if contact_channel_names is not None
            else ["left_heel", "left_toe", "right_heel", "right_toe"]
        )
        self.max_cached_sequences = int(max_cached_sequences)
        self.preload = bool(preload)
        self.share_memory = bool(share_memory)
        self.verbose = verbose
        self.sequence_cache = OrderedDict()
        self.pose_tensors = {}
        self.contact_tensors = {}
        self.event_names = self._build_event_names()
        self.events = self._build_events(seed)
        if not self.events:
            raise ValueError(f"No UnderPressure event windows found for split={split}.")

    def _build_event_names(self):
        names = []
        for contact_name in self.contact_channel_names:
            names.append(f"{contact_name}_on")
            names.append(f"{contact_name}_off")
        return names

    @property
    def num_event_classes(self):
        return len(self.event_names)

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
                contacts = torch.as_tensor(raw[key]).float()
                break
        else:
            raise KeyError(f"Could not find contact key in {raw.keys()}")

        if contacts.ndim > 2:
            contacts = contacts.reshape(contacts.shape[0], -1)
        if contacts.shape[1] != len(self.contact_channel_names):
            raise ValueError(
                f"Expected {len(self.contact_channel_names)} contact channels "
                f"({self.contact_channel_names}), got shape {tuple(contacts.shape)}"
            )
        return (torch.nan_to_num(contacts) >= 0.5).float()

    def _store_sequence_tensors(self, seq_idx, raw):
        if not self.preload or seq_idx in self.pose_tensors:
            return
        pose = self._extract_pose(raw).contiguous()
        contacts = self._extract_contacts(raw).contiguous()
        if self.share_memory:
            pose = pose.share_memory_()
            contacts = contacts.share_memory_()
        self.pose_tensors[seq_idx] = pose
        self.contact_tensors[seq_idx] = contacts

    def _get_pose_contacts(self, seq_idx):
        if seq_idx in self.pose_tensors:
            return self.pose_tensors[seq_idx], self.contact_tensors[seq_idx]
        raw = self._load_raw(seq_idx)
        pose = self._extract_pose(raw)
        contacts = self._extract_contacts(raw)
        return pose, contacts

    def _add_events_for_contact_channel(self, seq_idx, contacts, channel_idx, events):
        onsets, departures = get_event_indices(contacts[:, channel_idx].numpy() >= 0.5)
        nframes = contacts.shape[0]
        for event_type, frames in [(0, onsets), (1, departures)]:
            for frame in frames:
                frame = int(frame)
                start = frame - self.half_window_frames
                end = frame + self.half_window_frames + 1
                if start < 0 or end > nframes:
                    continue
                events.append(
                    {
                        "seq_idx": seq_idx,
                        "event_frame": frame,
                        "contact_channel": int(channel_idx),
                        "event_type": int(event_type),
                    }
                )

    def _build_events(self, seed):
        events = []
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
            if pose.shape[0] < self.window_frames:
                continue
            if self.preload:
                if self.share_memory:
                    pose = pose.contiguous().share_memory_()
                    contacts = contacts.contiguous().share_memory_()
                self.pose_tensors[seq_idx] = pose
                self.contact_tensors[seq_idx] = contacts

            for channel_idx in range(contacts.shape[1]):
                self._add_events_for_contact_channel(seq_idx, contacts, channel_idx, events)

            if self.verbose and ((seq_idx + 1) % 25 == 0 or seq_idx + 1 == len(self.files)):
                print(
                    f"[{self.split}] {seq_idx + 1}/{len(self.files)} sequences -> "
                    f"{len(events)} event windows",
                    flush=True,
                )

        if self.split == "train":
            rng.shuffle(events)
        return events

    def _event_class(self, contact_channel, event_type):
        return int(contact_channel) * 2 + int(event_type)

    def _make_heatmap_target(self, contacts, start, end):
        target = torch.zeros(self.window_frames, self.num_event_classes, dtype=torch.float32)
        xs = torch.arange(self.window_frames, dtype=torch.float32)
        sigma = max(self.event_sigma_frames, 1e-6)

        for channel_idx in range(contacts.shape[1]):
            onsets, departures = get_event_indices(contacts[:, channel_idx].numpy() >= 0.5)
            for event_type, frames in [(0, onsets), (1, departures)]:
                cls = self._event_class(channel_idx, event_type)
                for frame in frames:
                    frame = int(frame)
                    if frame < start or frame >= end:
                        continue
                    local_frame = frame - start
                    if self.event_sigma_frames <= 0:
                        target[local_frame, cls] = 1.0
                    else:
                        heat = torch.exp(-0.5 * ((xs - local_frame) / sigma) ** 2)
                        target[:, cls] = torch.maximum(target[:, cls], heat)
        return target

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event = self.events[idx]
        pose, contacts = self._get_pose_contacts(event["seq_idx"])
        start = event["event_frame"] - self.half_window_frames
        end = event["event_frame"] + self.half_window_frames + 1
        target = self._make_heatmap_target(contacts, start, end)
        return {
            "joint": pose[start:end],
            "event_heatmap": target,
            "anchor_class": torch.tensor(
                self._event_class(event["contact_channel"], event["event_type"]),
                dtype=torch.long,
            ),
            "anchor_frame": torch.tensor(self.half_window_frames, dtype=torch.long),
        }
