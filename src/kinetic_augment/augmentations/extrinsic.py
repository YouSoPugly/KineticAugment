"""
Extrinsic (Scene-level) Augmentations for KineticAugment.

These augmentations operate directly on MediaPipe landmarks without
requiring SMPL-X. They are fast (~1ms per frame) but don't guarantee
kinematic chain consistency.

Use for:
- Scene-level transformations (rotation, scaling)
- Camera simulation (viewpoint changes, jittering)
- Pose mirroring for symmetry augmentation

Augmentations:
- GlobalRotation: Rotate entire pose
- GlobalScaling: Scale pose uniformly
- PoseFlipping: Mirror across sagittal plane
- TrajectoryJittering: Add smooth camera shake effect
"""

from __future__ import annotations

from typing import Dict, Optional, Any
import numpy as np
from scipy.spatial.transform import Rotation

from kinetic_augment.augmentations.base import (
    LandmarkAugmentation,
    AugmentationRegistry,
)
from kinetic_augment.utils.data_formats import (
    POSE_START, POSE_END,
    FACE_START, FACE_END,
    LH_START, LH_END,
    RH_START, RH_END,
)
from kinetic_augment.utils.landmark_maps import POSE_FULL_MAP, FACE_FULL_MAP


@AugmentationRegistry.register("GlobalRotation")
class GlobalRotation(LandmarkAugmentation):
    """
    Apply random global rotation to the entire sequence.

    Simulates different camera viewpoints. All landmarks are rotated
    together, preserving relative positions.

    Parameters:
        max_angle_deg (Dict[str, float]): Maximum rotation per axis in degrees
            Keys: 'x' (pitch), 'y' (yaw), 'z' (roll)
            Default: {'x': 15, 'y': 15, 'z': 10}

    Example:
        >>> aug = GlobalRotation({'max_angle_deg': {'x': 20, 'y': 30, 'z': 15}})
        >>> rotated = aug.apply(landmarks)
    """

    name = "GlobalRotation"
    category = "extrinsic"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config)
        self.max_angle_deg = self.config.get(
            'max_angle_deg',
            {'x': 15, 'y': 15, 'z': 10}
        )

    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply random global rotation.

        Args:
            data: Landmarks of shape (num_frames, num_landmarks, 3)

        Returns:
            Rotated landmarks with same shape
        """
        rx = np.random.uniform(
            -self.max_angle_deg.get('x', 0),
            self.max_angle_deg.get('x', 0)
        )
        ry = np.random.uniform(
            -self.max_angle_deg.get('y', 0),
            self.max_angle_deg.get('y', 0)
        )
        rz = np.random.uniform(
            -self.max_angle_deg.get('z', 0),
            self.max_angle_deg.get('z', 0)
        )

        rotation = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True)

        num_frames, num_landmarks, _ = data.shape
        reshaped = data.reshape(-1, 3)
        rotated = rotation.apply(reshaped)

        return rotated.reshape(num_frames, num_landmarks, 3)


@AugmentationRegistry.register("GlobalScaling")
class GlobalScaling(LandmarkAugmentation):
    """
    Apply random uniform scaling to the entire sequence.

    Simulates different distances from camera or body sizes.

    Parameters:
        scale_range (List[float]): [min_scale, max_scale]
            Default: [0.8, 1.2]

    Example:
        >>> aug = GlobalScaling({'scale_range': [0.9, 1.1]})
        >>> scaled = aug.apply(landmarks)
    """

    name = "GlobalScaling"
    category = "extrinsic"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config)
        self.scale_range = self.config.get('scale_range', [0.8, 1.2])

    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply random uniform scaling."""
        scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return data * scale_factor


@AugmentationRegistry.register("PoseFlipping")
class PoseFlipping(LandmarkAugmentation):
    """
    Mirror pose across the sagittal (Y-Z) plane.

    For Sign Language: Flips the sign to its mirror version.
    Landmark labels are swapped (left <-> right) to maintain semantics.

    Note: This changes the meaning of directional signs. Use with care
    for sign language datasets - some signs have different meanings
    when mirrored.

    Parameters:
        flip_probability (float): Probability of applying flip (default: 0.5)

    Example:
        >>> aug = PoseFlipping({'flip_probability': 0.5})
        >>> flipped = aug.apply(landmarks)
    """

    name = "PoseFlipping"
    category = "extrinsic"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config)
        self.flip_probability = self.config.get('flip_probability', 0.5)

    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Mirror pose and swap left/right labels."""
        # Check if we should flip
        if np.random.rand() > self.flip_probability:
            return data

        flipped = data.copy()
        original = data.copy()

        # 1. Geometric flip: negate X coordinate
        flipped[:, :, 0] *= -1

        # 2. Swap pose landmarks (left <-> right)
        for left_idx, right_idx in POSE_FULL_MAP.items():
            flipped[:, POSE_START + left_idx, :] = original[:, POSE_START + right_idx, :]
            flipped[:, POSE_START + right_idx, :] = original[:, POSE_START + left_idx, :]
            # Apply X flip to the swapped values
            flipped[:, POSE_START + left_idx, 0] *= -1
            flipped[:, POSE_START + right_idx, 0] *= -1

        # 3. Swap face landmarks
        for left_idx, right_idx in FACE_FULL_MAP.items():
            flipped[:, FACE_START + left_idx, :] = original[:, FACE_START + right_idx, :]
            flipped[:, FACE_START + right_idx, :] = original[:, FACE_START + left_idx, :]
            flipped[:, FACE_START + left_idx, 0] *= -1
            flipped[:, FACE_START + right_idx, 0] *= -1

        # 4. Swap hands entirely
        left_hand = original[:, LH_START:LH_END, :].copy()
        right_hand = original[:, RH_START:RH_END, :].copy()

        flipped[:, LH_START:LH_END, :] = right_hand
        flipped[:, RH_START:RH_END, :] = left_hand

        # Apply X flip to swapped hands
        flipped[:, LH_START:LH_END, 0] *= -1
        flipped[:, RH_START:RH_END, 0] *= -1

        return flipped


@AugmentationRegistry.register("TrajectoryJittering")
class TrajectoryJittering(LandmarkAugmentation):
    """
    Add smooth, low-frequency noise to landmark trajectories.

    Simulates camera shake or natural movement variation.
    Uses sum of sine waves for smooth, continuous noise.

    Parameters:
        amplitude (float): Maximum displacement (default: 0.01)
        frequency (float): Base frequency of noise (default: 0.5)
        per_landmark (bool): Different noise per landmark (default: False)

    Example:
        >>> aug = TrajectoryJittering({
        ...     'amplitude': 0.02,
        ...     'frequency': 0.3,
        ...     'per_landmark': False  # Camera shake effect
        ... })
    """

    name = "TrajectoryJittering"
    category = "extrinsic"

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config)
        self.amplitude = self.config.get('amplitude', 0.01)
        self.frequency = self.config.get('frequency', 0.5)
        self.per_landmark = self.config.get('per_landmark', False)

    def _generate_smooth_noise(self, num_frames: int, num_dims: int = 3) -> np.ndarray:
        """Generate smooth noise using sum of sine waves."""
        noise = np.zeros((num_frames, num_dims))

        for dim in range(num_dims):
            # Sum of 3 sine waves with random phases
            for _ in range(3):
                freq = np.random.uniform(0.1, self.frequency)
                phase = np.random.uniform(0, 2 * np.pi)
                t = np.linspace(0, freq * 2 * np.pi, num_frames)
                noise[:, dim] += np.sin(t + phase)

        # Normalize and scale
        noise /= np.abs(noise).max() + 1e-8
        noise *= self.amplitude

        return noise

    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply trajectory jittering."""
        num_frames, num_landmarks, _ = data.shape

        if self.per_landmark:
            # Different noise for each landmark
            jittered = data.copy()
            for i in range(num_landmarks):
                noise = self._generate_smooth_noise(num_frames)
                jittered[:, i, :] += noise
        else:
            # Same noise for all landmarks (camera shake)
            noise = self._generate_smooth_noise(num_frames)
            jittered = data + noise[:, np.newaxis, :]

        return jittered


@AugmentationRegistry.register("GaussianNoise")
class GaussianNoise(LandmarkAugmentation):
    """
    Add independent Gaussian noise to all landmarks.

    Simple augmentation for robustness to detection noise.
    For anatomically-aware noise, use JointAnglePerturbation instead.

    Parameters:
        stddev (float): Standard deviation of noise (default: 0.005)
        temporal_smoothing (bool): Smooth noise across time (default: True)
        landmark_groups (List[str]): Which groups to affect
            Options: 'pose', 'face', 'left_hand', 'right_hand', 'all'

    Example:
        >>> aug = GaussianNoise({
        ...     'stddev': 0.01,
        ...     'landmark_groups': ['pose', 'left_hand', 'right_hand']
        ... })
    """

    name = "GaussianNoise"
    category = "extrinsic"

    LANDMARK_RANGES = {
        'pose': (POSE_START, POSE_END),
        'face': (FACE_START, FACE_END),
        'left_hand': (LH_START, LH_END),
        'right_hand': (RH_START, RH_END),
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config)
        self.stddev = self.config.get('stddev', 0.005)
        self.temporal_smoothing = self.config.get('temporal_smoothing', True)
        self.landmark_groups = self.config.get('landmark_groups', ['all'])

    def _get_landmark_mask(self, num_landmarks: int) -> np.ndarray:
        """Create mask for which landmarks to affect."""
        mask = np.zeros(num_landmarks, dtype=bool)

        if 'all' in self.landmark_groups:
            mask[:] = True
        else:
            for group in self.landmark_groups:
                if group in self.LANDMARK_RANGES:
                    start, end = self.LANDMARK_RANGES[group]
                    mask[start:end] = True

        return mask

    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Gaussian noise."""
        num_frames, num_landmarks, num_dims = data.shape

        mask = self._get_landmark_mask(num_landmarks)
        noisy = data.copy()

        if self.temporal_smoothing:
            # Generate smooth noise (low-pass filtered)
            noise = np.random.randn(num_frames, num_landmarks, num_dims) * self.stddev

            # Simple smoothing via moving average
            kernel_size = max(3, num_frames // 10)
            kernel = np.ones(kernel_size) / kernel_size

            for i in range(num_landmarks):
                if mask[i]:
                    for d in range(num_dims):
                        noise[:, i, d] = np.convolve(
                            noise[:, i, d], kernel, mode='same'
                        )
        else:
            noise = np.random.randn(num_frames, num_landmarks, num_dims) * self.stddev

        # Apply noise only to masked landmarks
        noisy[:, mask, :] += noise[:, mask, :]

        return noisy
