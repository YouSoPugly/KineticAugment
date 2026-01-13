"""
Temporal Augmentations for KineticAugment.

These augmentations modify the time dimension of sequences:
- Speed up or slow down portions of the motion
- Non-linearly warp time
- Drop or interpolate frames

For Sign Language Recognition:
- TimeWarping preserves sign meaning while varying speed
- SpeedVariation simulates different signing speeds
- FrameDropping simulates video compression artifacts
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Union
import numpy as np
from scipy.interpolate import CubicSpline, interp1d

from kinetic_augment.augmentations.base import (
    TemporalAugmentation,
    AugmentationRegistry,
)


@AugmentationRegistry.register("TimeWarping")
class TimeWarping(TemporalAugmentation):
    """
    Non-linearly warp the time axis of the sequence.

    Different parts of the sequence are stretched or compressed
    while maintaining smooth motion. Uses cubic splines to create
    smooth warping curves.

    Parameters:
        max_warp_factor (float): Maximum time stretch factor (default: 0.2)
        num_knots (int): Number of control points for warp curve (default: 5)
        preserve_endpoints (bool): Keep first/last frames fixed (default: True)

    Example:
        >>> aug = TimeWarping({'max_warp_factor': 0.3, 'num_knots': 4})
        >>> warped = aug.apply(landmarks)
    """

    name = "TimeWarping"
    category = "temporal"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config)
        self.max_warp_factor = self.config.get('max_warp_factor', 0.2)
        self.num_knots = self.config.get('num_knots', 5)
        self.preserve_endpoints = self.config.get('preserve_endpoints', True)

    def apply(
        self,
        data: Union[np.ndarray, List[Dict[str, np.ndarray]]],
        **kwargs
    ) -> Union[np.ndarray, List[Dict[str, np.ndarray]]]:
        """
        Apply time warping to sequence.

        Args:
            data: Either landmarks (num_frames, num_landmarks, 3) or
                  list of SMPL-X parameter dicts

        Returns:
            Time-warped data with same structure
        """
        if isinstance(data, list):
            return self._warp_smplx_sequence(data)
        else:
            return self._warp_landmark_sequence(data)

    def _warp_landmark_sequence(self, data: np.ndarray) -> np.ndarray:
        """Warp landmark sequence."""
        num_frames, num_landmarks, num_dims = data.shape

        # Original time axis
        t_orig = np.arange(num_frames)

        # Create warping curve
        t_warped = self._create_warp_curve(num_frames)

        # Resample data at warped time points
        warped = np.zeros_like(data)
        for i in range(num_landmarks):
            for j in range(num_dims):
                warped[:, i, j] = np.interp(t_warped, t_orig, data[:, i, j])

        return warped

    def _warp_smplx_sequence(
        self,
        params_list: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        """Warp sequence of SMPL-X parameters."""
        num_frames = len(params_list)

        # Create warping curve
        t_orig = np.arange(num_frames)
        t_warped = self._create_warp_curve(num_frames)

        # Get warped frame indices
        warped_indices = np.interp(t_warped, t_orig, t_orig)

        # Interpolate parameters
        result = []
        for t in range(num_frames):
            # Find surrounding frames
            idx = warped_indices[t]
            idx_floor = int(np.floor(idx))
            idx_ceil = min(int(np.ceil(idx)), num_frames - 1)
            alpha = idx - idx_floor

            if idx_floor == idx_ceil:
                result.append({k: v.copy() for k, v in params_list[idx_floor].items()})
            else:
                # Linear interpolation between frames
                interpolated = {}
                for key in params_list[idx_floor].keys():
                    v1 = params_list[idx_floor][key]
                    v2 = params_list[idx_ceil][key]
                    if isinstance(v1, np.ndarray):
                        interpolated[key] = (1 - alpha) * v1 + alpha * v2
                    else:
                        interpolated[key] = v1
                result.append(interpolated)

        return result

    def _create_warp_curve(self, num_frames: int) -> np.ndarray:
        """Create smooth warping curve using cubic splines."""
        # Control points
        knot_x = np.linspace(0, num_frames - 1, self.num_knots)
        knot_y_offsets = (
            np.random.uniform(-1, 1, size=self.num_knots) *
            self.max_warp_factor * num_frames
        )

        # Fix endpoints if requested
        if self.preserve_endpoints:
            knot_y_offsets[0] = 0
            knot_y_offsets[-1] = 0

        # Create spline
        spline = CubicSpline(knot_x, knot_x + knot_y_offsets)
        t_warped = spline(np.arange(num_frames))

        # Ensure monotonic and within bounds
        t_warped = np.clip(t_warped, 0, num_frames - 1)

        # Ensure strictly increasing (fix any inversions)
        for i in range(1, len(t_warped)):
            if t_warped[i] <= t_warped[i-1]:
                t_warped[i] = t_warped[i-1] + 0.01

        return t_warped


@AugmentationRegistry.register("SpeedVariation")
class SpeedVariation(TemporalAugmentation):
    """
    Globally speed up or slow down the sequence.

    Unlike TimeWarping, this applies uniform speed change.
    Useful for simulating different signing speeds.

    Parameters:
        speed_range (List[float]): [min_speed, max_speed] multipliers
            Default: [0.8, 1.2] (80% to 120% speed)
        output_frames (str): 'same', 'proportional', or int
            'same': Keep original frame count (interpolate)
            'proportional': Scale frame count with speed
            int: Fixed output frame count

    Example:
        >>> aug = SpeedVariation({
        ...     'speed_range': [0.7, 1.3],
        ...     'output_frames': 'same'
        ... })
    """

    name = "SpeedVariation"
    category = "temporal"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config)
        self.speed_range = self.config.get('speed_range', [0.8, 1.2])
        self.output_frames = self.config.get('output_frames', 'same')

    def apply(
        self,
        data: Union[np.ndarray, List[Dict[str, np.ndarray]]],
        **kwargs
    ) -> Union[np.ndarray, List[Dict[str, np.ndarray]]]:
        """Apply speed variation."""
        speed_factor = np.random.uniform(self.speed_range[0], self.speed_range[1])

        if isinstance(data, list):
            return self._vary_speed_smplx(data, speed_factor)
        else:
            return self._vary_speed_landmarks(data, speed_factor)

    def _vary_speed_landmarks(self, data: np.ndarray, speed: float) -> np.ndarray:
        """Vary speed of landmark sequence."""
        num_frames, num_landmarks, num_dims = data.shape

        # Determine output frame count
        if self.output_frames == 'same':
            out_frames = num_frames
        elif self.output_frames == 'proportional':
            out_frames = int(num_frames / speed)
        else:
            out_frames = int(self.output_frames)

        out_frames = max(2, out_frames)

        # Original and target time axes
        t_orig = np.linspace(0, 1, num_frames)
        t_new = np.linspace(0, 1, out_frames)

        # Interpolate
        result = np.zeros((out_frames, num_landmarks, num_dims))
        for i in range(num_landmarks):
            for j in range(num_dims):
                f = interp1d(t_orig, data[:, i, j], kind='linear', fill_value='extrapolate')
                result[:, i, j] = f(t_new)

        return result

    def _vary_speed_smplx(
        self,
        params_list: List[Dict[str, np.ndarray]],
        speed: float
    ) -> List[Dict[str, np.ndarray]]:
        """Vary speed of SMPL-X parameter sequence."""
        num_frames = len(params_list)

        # Determine output frame count
        if self.output_frames == 'same':
            out_frames = num_frames
        elif self.output_frames == 'proportional':
            out_frames = int(num_frames / speed)
        else:
            out_frames = int(self.output_frames)

        out_frames = max(2, out_frames)

        # Map output frames to input frames
        t_orig = np.linspace(0, num_frames - 1, num_frames)
        t_new = np.linspace(0, num_frames - 1, out_frames)

        result = []
        for t in t_new:
            idx_floor = int(np.floor(t))
            idx_ceil = min(int(np.ceil(t)), num_frames - 1)
            alpha = t - idx_floor

            if idx_floor == idx_ceil:
                result.append({k: v.copy() for k, v in params_list[idx_floor].items()})
            else:
                interpolated = {}
                for key in params_list[idx_floor].keys():
                    v1 = params_list[idx_floor][key]
                    v2 = params_list[idx_ceil][key]
                    if isinstance(v1, np.ndarray):
                        interpolated[key] = (1 - alpha) * v1 + alpha * v2
                    else:
                        interpolated[key] = v1
                result.append(interpolated)

        return result


@AugmentationRegistry.register("FrameDropping")
class FrameDropping(TemporalAugmentation):
    """
    Randomly drop frames and interpolate to simulate video artifacts.

    Useful for robustness to poor video quality or compression.

    Parameters:
        drop_probability (float): Probability of dropping each frame (default: 0.1)
        max_consecutive_drops (int): Maximum consecutive frames to drop (default: 3)
        interpolation (str): 'linear', 'cubic', or 'nearest'

    Example:
        >>> aug = FrameDropping({
        ...     'drop_probability': 0.15,
        ...     'max_consecutive_drops': 2
        ... })
    """

    name = "FrameDropping"
    category = "temporal"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config)
        self.drop_probability = self.config.get('drop_probability', 0.1)
        self.max_consecutive = self.config.get('max_consecutive_drops', 3)
        self.interpolation = self.config.get('interpolation', 'linear')

    def apply(
        self,
        data: Union[np.ndarray, List[Dict[str, np.ndarray]]],
        **kwargs
    ) -> Union[np.ndarray, List[Dict[str, np.ndarray]]]:
        """Apply frame dropping with interpolation."""
        if isinstance(data, list):
            return self._drop_frames_smplx(data)
        else:
            return self._drop_frames_landmarks(data)

    def _drop_frames_landmarks(self, data: np.ndarray) -> np.ndarray:
        """Drop frames from landmark sequence."""
        num_frames, num_landmarks, num_dims = data.shape

        # Determine which frames to keep
        keep_mask = self._create_keep_mask(num_frames)

        if keep_mask.sum() < 2:
            # Need at least 2 frames to interpolate
            keep_mask[0] = True
            keep_mask[-1] = True

        # Get kept frames
        kept_indices = np.where(keep_mask)[0]
        kept_data = data[keep_mask]

        # Interpolate back to original length
        result = np.zeros_like(data)
        for i in range(num_landmarks):
            for j in range(num_dims):
                f = interp1d(
                    kept_indices,
                    kept_data[:, i, j],
                    kind=self.interpolation,
                    fill_value='extrapolate'
                )
                result[:, i, j] = f(np.arange(num_frames))

        return result

    def _drop_frames_smplx(
        self,
        params_list: List[Dict[str, np.ndarray]]
    ) -> List[Dict[str, np.ndarray]]:
        """Drop frames from SMPL-X sequence."""
        num_frames = len(params_list)

        keep_mask = self._create_keep_mask(num_frames)

        if keep_mask.sum() < 2:
            keep_mask[0] = True
            keep_mask[-1] = True

        kept_indices = np.where(keep_mask)[0]
        kept_params = [params_list[i] for i in kept_indices]

        # Interpolate back
        result = []
        for t in range(num_frames):
            # Find surrounding kept frames
            lower_idx = np.searchsorted(kept_indices, t, side='right') - 1
            lower_idx = max(0, lower_idx)
            upper_idx = min(lower_idx + 1, len(kept_indices) - 1)

            if lower_idx == upper_idx or kept_indices[lower_idx] == t:
                result.append({k: v.copy() for k, v in kept_params[lower_idx].items()})
            else:
                # Interpolate
                t_lower = kept_indices[lower_idx]
                t_upper = kept_indices[upper_idx]
                alpha = (t - t_lower) / (t_upper - t_lower)

                interpolated = {}
                for key in kept_params[lower_idx].keys():
                    v1 = kept_params[lower_idx][key]
                    v2 = kept_params[upper_idx][key]
                    if isinstance(v1, np.ndarray):
                        interpolated[key] = (1 - alpha) * v1 + alpha * v2
                    else:
                        interpolated[key] = v1
                result.append(interpolated)

        return result

    def _create_keep_mask(self, num_frames: int) -> np.ndarray:
        """Create mask for which frames to keep."""
        keep_mask = np.ones(num_frames, dtype=bool)
        consecutive_drops = 0

        for i in range(num_frames):
            if np.random.rand() < self.drop_probability:
                if consecutive_drops < self.max_consecutive:
                    keep_mask[i] = False
                    consecutive_drops += 1
                else:
                    consecutive_drops = 0
            else:
                consecutive_drops = 0

        # Always keep first and last frames
        keep_mask[0] = True
        keep_mask[-1] = True

        return keep_mask
