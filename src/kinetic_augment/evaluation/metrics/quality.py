"""
Quality Metrics for KineticAugment Evaluation.

Metrics for evaluating pose quality and anatomical plausibility.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np

from kinetic_augment.evaluation.base import BaseMetric, MetricResult


# MediaPipe Pose landmark indices for limb pairs
# Format: (start_landmark, end_landmark, limb_name)
POSE_LIMB_PAIRS = [
    # Arms
    (11, 13, 'left_upper_arm'),   # left_shoulder to left_elbow
    (13, 15, 'left_forearm'),     # left_elbow to left_wrist
    (12, 14, 'right_upper_arm'),  # right_shoulder to right_elbow
    (14, 16, 'right_forearm'),    # right_elbow to right_wrist
    # Legs
    (23, 25, 'left_thigh'),       # left_hip to left_knee
    (25, 27, 'left_shin'),        # left_knee to left_ankle
    (24, 26, 'right_thigh'),      # right_hip to right_knee
    (26, 28, 'right_shin'),       # right_knee to right_ankle
    # Torso
    (11, 12, 'shoulders'),        # left_shoulder to right_shoulder
    (11, 23, 'left_torso'),       # left_shoulder to left_hip
    (12, 24, 'right_torso'),      # right_shoulder to right_hip
    (23, 24, 'hips'),             # left_hip to right_hip
]

# Hand landmark indices for finger segments (relative to hand base at 0)
HAND_FINGER_PAIRS = [
    # Thumb
    (0, 1, 'thumb_cmc'), (1, 2, 'thumb_mcp'), (2, 3, 'thumb_ip'), (3, 4, 'thumb_tip'),
    # Index
    (0, 5, 'index_mcp_base'), (5, 6, 'index_pip'), (6, 7, 'index_dip'), (7, 8, 'index_tip'),
    # Middle
    (0, 9, 'middle_mcp_base'), (9, 10, 'middle_pip'), (10, 11, 'middle_dip'), (11, 12, 'middle_tip'),
    # Ring
    (0, 13, 'ring_mcp_base'), (13, 14, 'ring_pip'), (14, 15, 'ring_dip'), (15, 16, 'ring_tip'),
    # Pinky
    (0, 17, 'pinky_mcp_base'), (17, 18, 'pinky_pip'), (18, 19, 'pinky_dip'), (19, 20, 'pinky_tip'),
]


class LimbLengthConsistencyMetric(BaseMetric):
    """
    Check limb length stability across frames.

    Measures how consistent limb lengths are throughout a sequence.
    High consistency indicates anatomically plausible augmentation.

    Computes:
    - mean_variance: Average variance in limb lengths across frames
    - max_deviation: Maximum frame-to-frame length change (relative)
    - stability_score: 1 - normalized variance (higher = better)
    """

    name = "LimbLengthConsistency"
    description = "Measures limb length stability across frames"

    def __init__(
        self,
        include_hands: bool = True,
        deviation_threshold: float = 0.05,  # 5% deviation threshold
    ):
        """
        Initialize the metric.

        Args:
            include_hands: Whether to include hand finger segments
            deviation_threshold: Threshold for acceptable deviation
        """
        self.include_hands = include_hands
        self.deviation_threshold = deviation_threshold

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """
        Compute limb length consistency.

        Args:
            data: Landmarks of shape (frames, 543, 3) or (frames, landmarks, 3)

        Returns:
            MetricResult with stability score and details
        """
        if data.ndim != 3:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': f'Expected 3D array, got {data.ndim}D'}
            )

        num_frames = data.shape[0]
        if num_frames < 2:
            return MetricResult(
                name=self.name,
                value=1.0,
                details={'note': 'Single frame, perfect consistency'}
            )

        # Compute limb lengths per frame
        limb_lengths = self._compute_limb_lengths(data)

        # Compute statistics
        variances = []
        max_deviations = []
        limb_details = {}

        for limb_name, lengths in limb_lengths.items():
            if len(lengths) < 2:
                continue

            mean_length = np.mean(lengths)
            if mean_length < 1e-6:
                continue

            # Relative variance
            rel_variance = np.var(lengths) / (mean_length ** 2)
            variances.append(rel_variance)

            # Max relative deviation
            frame_diffs = np.abs(np.diff(lengths)) / mean_length
            max_dev = np.max(frame_diffs) if len(frame_diffs) > 0 else 0
            max_deviations.append(max_dev)

            limb_details[limb_name] = {
                'mean_length': float(mean_length),
                'variance': float(rel_variance),
                'max_deviation': float(max_dev),
            }

        if not variances:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'No valid limbs found'}
            )

        mean_variance = np.mean(variances)
        max_deviation = np.max(max_deviations)

        # Stability score: 1 for perfect, 0 for very inconsistent
        # Use exponential decay based on variance
        stability_score = np.exp(-mean_variance * 100)

        return MetricResult(
            name=self.name,
            value=float(stability_score),
            details={
                'mean_variance': float(mean_variance),
                'max_deviation': float(max_deviation),
                'num_limbs': len(variances),
                'limbs': limb_details,
                'threshold_violations': sum(1 for d in max_deviations if d > self.deviation_threshold),
            }
        )

    def _compute_limb_lengths(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute limb lengths for each frame."""
        num_frames, num_landmarks, _ = data.shape
        lengths = {}

        # Pose landmarks (indices 0-32)
        for start, end, name in POSE_LIMB_PAIRS:
            if start < num_landmarks and end < num_landmarks:
                segment_lengths = np.linalg.norm(
                    data[:, end, :] - data[:, start, :],
                    axis=1
                )
                lengths[name] = segment_lengths

        # Hand landmarks (if full 543 landmarks)
        if self.include_hands and num_landmarks >= 543:
            # Left hand: landmarks 468-488 (21 landmarks)
            left_hand_base = 468
            for start, end, name in HAND_FINGER_PAIRS:
                l_start = left_hand_base + start
                l_end = left_hand_base + end
                if l_end < num_landmarks:
                    segment_lengths = np.linalg.norm(
                        data[:, l_end, :] - data[:, l_start, :],
                        axis=1
                    )
                    lengths[f'left_{name}'] = segment_lengths

            # Right hand: landmarks 489-509 (21 landmarks)
            right_hand_base = 489
            for start, end, name in HAND_FINGER_PAIRS:
                r_start = right_hand_base + start
                r_end = right_hand_base + end
                if r_end < num_landmarks:
                    segment_lengths = np.linalg.norm(
                        data[:, r_end, :] - data[:, r_start, :],
                        axis=1
                    )
                    lengths[f'right_{name}'] = segment_lengths

        return lengths


class PoseValidityMetric(BaseMetric):
    """
    Check if poses are geometrically valid.

    Validates:
    - No NaN/Inf values
    - Landmarks within reasonable bounds
    - No collapsed poses (all landmarks same position)
    """

    name = "PoseValidity"
    description = "Checks for geometrically valid poses"

    def __init__(
        self,
        bounds: Tuple[float, float] = (-10.0, 10.0),
        min_spread: float = 0.01,
    ):
        """
        Initialize the metric.

        Args:
            bounds: (min, max) acceptable coordinate values
            min_spread: Minimum spread of landmarks to be considered valid
        """
        self.bounds = bounds
        self.min_spread = min_spread

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """
        Compute pose validity score.

        Args:
            data: Landmarks of shape (frames, landmarks, 3)

        Returns:
            MetricResult with validity score
        """
        if data.ndim != 3:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': f'Expected 3D array, got {data.ndim}D'}
            )

        num_frames = data.shape[0]
        valid_frames = 0
        issues = {
            'nan_frames': 0,
            'inf_frames': 0,
            'out_of_bounds_frames': 0,
            'collapsed_frames': 0,
        }

        for i in range(num_frames):
            frame = data[i]

            # Check NaN
            if np.any(np.isnan(frame)):
                issues['nan_frames'] += 1
                continue

            # Check Inf
            if np.any(np.isinf(frame)):
                issues['inf_frames'] += 1
                continue

            # Check bounds
            if np.any(frame < self.bounds[0]) or np.any(frame > self.bounds[1]):
                issues['out_of_bounds_frames'] += 1
                continue

            # Check spread (not collapsed)
            spread = np.std(frame)
            if spread < self.min_spread:
                issues['collapsed_frames'] += 1
                continue

            valid_frames += 1

        validity_score = valid_frames / num_frames if num_frames > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=float(validity_score),
            details={
                'valid_frames': valid_frames,
                'total_frames': num_frames,
                **issues,
            }
        )


class LandmarkDetectionRate(BaseMetric):
    """
    Measure landmark detection quality.

    For MediaPipe data, checks visibility/presence of landmarks.
    Uses zero-valued landmarks as proxy for missing detections.
    """

    name = "LandmarkDetectionRate"
    description = "Measures landmark detection completeness"

    def __init__(self, zero_threshold: float = 1e-6):
        """
        Initialize the metric.

        Args:
            zero_threshold: Threshold below which coordinates are considered zero
        """
        self.zero_threshold = zero_threshold

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """
        Compute landmark detection rate.

        Args:
            data: Landmarks of shape (frames, landmarks, 3)

        Returns:
            MetricResult with detection rate
        """
        if data.ndim != 3:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': f'Expected 3D array, got {data.ndim}D'}
            )

        num_frames, num_landmarks, _ = data.shape

        # Count non-zero landmarks per frame
        is_detected = np.linalg.norm(data, axis=2) > self.zero_threshold
        detection_rate = np.mean(is_detected)

        # Per-landmark detection rate
        per_landmark_rate = np.mean(is_detected, axis=0)

        return MetricResult(
            name=self.name,
            value=float(detection_rate),
            details={
                'per_frame_rate': float(np.mean(np.mean(is_detected, axis=1))),
                'min_landmark_rate': float(np.min(per_landmark_rate)),
                'max_landmark_rate': float(np.max(per_landmark_rate)),
                'fully_detected_frames': int(np.sum(np.all(is_detected, axis=1))),
            }
        )


class AnatomicalPlausibilityMetric(BaseMetric):
    """
    Composite metric for overall anatomical plausibility.

    Combines multiple quality checks into a single score:
    - Limb length consistency
    - Pose validity
    - Detection rate
    """

    name = "AnatomicalPlausibility"
    description = "Composite anatomical plausibility score"

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the metric.

        Args:
            weights: Optional weights for component metrics
        """
        self.weights = weights or {
            'limb_consistency': 0.4,
            'pose_validity': 0.4,
            'detection_rate': 0.2,
        }

        self.limb_metric = LimbLengthConsistencyMetric()
        self.validity_metric = PoseValidityMetric()
        self.detection_metric = LandmarkDetectionRate()

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """
        Compute composite plausibility score.

        Args:
            data: Landmarks of shape (frames, landmarks, 3)

        Returns:
            MetricResult with plausibility score 0-1
        """
        # Compute component metrics
        limb_result = self.limb_metric.compute(data, **kwargs)
        validity_result = self.validity_metric.compute(data, **kwargs)
        detection_result = self.detection_metric.compute(data, **kwargs)

        # Weighted combination
        score = (
            self.weights['limb_consistency'] * limb_result.value +
            self.weights['pose_validity'] * validity_result.value +
            self.weights['detection_rate'] * detection_result.value
        )

        return MetricResult(
            name=self.name,
            value=float(score),
            details={
                'limb_consistency': limb_result.value,
                'pose_validity': validity_result.value,
                'detection_rate': detection_result.value,
                'weights': self.weights,
            }
        )
