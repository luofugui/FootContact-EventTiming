import numpy as np
import torch


def get_event_indices(binary_signal):
    """Return onset and departure frame indices from a binary contact signal."""
    signal = np.asarray(binary_signal).astype(int)
    padded = np.concatenate(([0], signal, [0]))
    diff = np.diff(padded)
    onsets = np.where(diff == 1)[0]
    departures = np.where(diff == -1)[0]
    return onsets, departures


class RegionalContactProcessor:
    """Convert PSU-TMM100 pressure maps into regional contact labels."""

    def __init__(
        self,
        foot_mask_path,
        num_regions=(2, 1),
        contact_threshold=0.003,
        active_only=True,
    ):
        self.foot_mask = np.load(foot_mask_path)
        self.num_regions = tuple(num_regions)
        self.contact_threshold = float(contact_threshold)
        self.active_only = bool(active_only)

        self.active_indices = np.where(~np.isnan(self.foot_mask.flatten()))[0]
        coords = np.unravel_index(self.active_indices, self.foot_mask.shape)
        self.active_rows = coords[0]
        self.active_cols = coords[1]
        self.channel_idx = coords[2]
        self.pixel_regions = self._build_region_ids()
        self.output_dim = self.num_regions[0] * self.num_regions[1] * 2

    def _build_region_ids(self):
        rows, cols = self.num_regions
        regions_per_foot = rows * cols
        pixel_regions = np.zeros(len(self.active_indices), dtype=int)

        for foot_idx in (0, 1):
            foot_mask = self.channel_idx == foot_idx
            if not np.any(foot_mask):
                continue

            foot_rows = self.active_rows[foot_mask]
            foot_cols = self.active_cols[foot_mask]
            min_row, max_row = np.min(foot_rows), np.max(foot_rows)
            min_col, max_col = np.min(foot_cols), np.max(foot_cols)
            row_size = max((max_row - min_row) / rows, 1e-6)
            col_size = max((max_col - min_col) / cols, 1e-6)

            region_rows = np.floor((foot_rows - min_row) / row_size).clip(0, rows - 1)
            region_cols = np.floor((foot_cols - min_col) / col_size).clip(0, cols - 1)
            region_ids = (region_rows * cols + region_cols).astype(int)
            region_ids += foot_idx * regions_per_foot
            pixel_regions[np.where(foot_mask)[0]] = region_ids

        return pixel_regions

    def pressure_to_contact(self, pressure):
        pressure = torch.as_tensor(pressure).float().reshape(-1)
        if pressure.numel() == len(self.active_indices):
            active_pressure = pressure
        else:
            active_pressure = pressure[self.active_indices]

        contact = torch.zeros(self.output_dim, dtype=torch.float32)
        for region_idx in range(self.output_dim):
            mask = self.pixel_regions == region_idx
            if np.any(mask):
                contact[region_idx] = (active_pressure[mask].max() > self.contact_threshold).float()
        return contact.numpy()
