import pickle
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from footcontact_event_timing.data.contact_events import (
    RegionalContactProcessor,
    get_event_indices,
)


def split_chunk_paths(chunk_dir, subject, train_val_split=0.9, shuffle=True, seed=0):
    files = sorted(Path(chunk_dir).glob("subject_*.pkl"))
    train_val = [p for p in files if f"subject_{subject}_" not in p.name]
    test = [p for p in files if f"subject_{subject}_" in p.name]
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(train_val)
    split = int(len(train_val) * train_val_split)
    return train_val[:split], train_val[split:], test


class PSUEventOffsetDataset(Dataset):
    """Pose-window dataset for contact onset/departure offset regression."""

    def __init__(
        self,
        chunk_files,
        foot_mask_path,
        split="train",
        window_frames=51,
        samples_per_event=1,
        eval_offsets_per_event=5,
        min_event_offset=5,
        max_event_offset=None,
        num_regions=(2, 1),
        contact_threshold=0.003,
        active_only=True,
        max_cached_chunks=8,
        seed=0,
        verbose=True,
    ):
        self.chunk_files = [Path(p) for p in chunk_files]
        self.split = split
        self.window_frames = int(window_frames)
        self.samples_per_event = int(samples_per_event)
        self.eval_offsets_per_event = int(eval_offsets_per_event)
        self.min_event_offset = int(min_event_offset)
        self.max_event_offset = (
            int(max_event_offset)
            if max_event_offset is not None
            else self.window_frames - 1 - self.min_event_offset
        )
        self.max_cached_chunks = int(max_cached_chunks)
        self.chunk_cache = OrderedDict()
        self.verbose = verbose
        self.processor = RegionalContactProcessor(
            foot_mask_path=foot_mask_path,
            num_regions=num_regions,
            contact_threshold=contact_threshold,
            active_only=active_only,
        )

        if self.min_event_offset > self.max_event_offset:
            raise ValueError("min_event_offset must be <= max_event_offset.")

        self.events = self._build_events(seed)
        if not self.events:
            raise ValueError(f"No event windows found for split={split}.")

    def _load_chunk(self, chunk_idx):
        if chunk_idx in self.chunk_cache:
            self.chunk_cache.move_to_end(chunk_idx)
            return self.chunk_cache[chunk_idx]
        with open(self.chunk_files[chunk_idx], "rb") as f:
            chunk = pickle.load(f)
        self.chunk_cache[chunk_idx] = chunk
        if len(self.chunk_cache) > self.max_cached_chunks:
            self.chunk_cache.popitem(last=False)
        return chunk

    @staticmethod
    def _sample_fields(sample):
        if isinstance(sample, dict):
            return sample["joint"], sample["pressure"]
        if isinstance(sample, (tuple, list)) and len(sample) >= 2:
            return sample[0], sample[1]
        raise TypeError(f"Unsupported sample type: {type(sample)}")

    def _chunk_contacts(self, chunk):
        contacts = []
        for sample in chunk:
            _, pressure = self._sample_fields(sample)
            contacts.append(self.processor.pressure_to_contact(pressure))
        return np.stack(contacts).astype(np.float32)

    def _valid_offsets(self, event_frame, nframes):
        lo = max(self.min_event_offset, event_frame - (nframes - self.window_frames))
        hi = min(self.max_event_offset, event_frame)
        if lo > hi:
            return []
        return list(range(lo, hi + 1))

    def _select_offsets(self, offsets, rng):
        if self.split == "train":
            return [rng.choice(offsets) for _ in range(self.samples_per_event)]

        count = min(self.eval_offsets_per_event, len(offsets))
        if count <= 1:
            return [offsets[len(offsets) // 2]]

        indices = np.linspace(0, len(offsets) - 1, count).round().astype(int)
        return [offsets[int(idx)] for idx in indices]

    def _build_events(self, seed):
        events = []
        rng = random.Random(seed)
        if self.verbose:
            print(f"[{self.split}] scanning {len(self.chunk_files)} chunks...", flush=True)

        for chunk_idx in range(len(self.chunk_files)):
            chunk = self._load_chunk(chunk_idx)
            nframes = len(chunk)
            if nframes < self.window_frames:
                continue

            contacts = self._chunk_contacts(chunk)
            for channel in range(contacts.shape[1]):
                onsets, departures = get_event_indices(contacts[:, channel] >= 0.5)
                for event_type, event_frames in [(0, onsets), (1, departures)]:
                    for event_frame in event_frames:
                        offsets = self._valid_offsets(int(event_frame), nframes)
                        if not offsets:
                            continue
                        for offset in self._select_offsets(offsets, rng):
                            events.append(
                                {
                                    "chunk_idx": chunk_idx,
                                    "event_frame": int(event_frame),
                                    "offset": int(offset),
                                    "channel": int(channel),
                                    "event_type": int(event_type),
                                }
                            )

            if self.verbose and ((chunk_idx + 1) % 10 == 0 or chunk_idx + 1 == len(self.chunk_files)):
                print(
                    f"[{self.split}] {chunk_idx + 1}/{len(self.chunk_files)} chunks -> "
                    f"{len(events)} windows",
                    flush=True,
                )

        if self.split == "train":
            rng.shuffle(events)
        return events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        event = self.events[idx]
        chunk = self._load_chunk(event["chunk_idx"])
        start = event["event_frame"] - event["offset"]
        end = start + self.window_frames

        joints = []
        for frame_idx in range(start, end):
            joint, _ = self._sample_fields(chunk[frame_idx])
            joints.append(torch.as_tensor(joint).float())

        return {
            "joint": torch.stack(joints),
            "offset": torch.tensor(event["offset"] / (self.window_frames - 1), dtype=torch.float32),
            "event_type": torch.tensor(event["event_type"], dtype=torch.long),
            "channel": torch.tensor(event["channel"], dtype=torch.long),
        }
