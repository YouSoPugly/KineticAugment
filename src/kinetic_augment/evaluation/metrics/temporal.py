"""
Temporal Metrics for KineticAugment Evaluation.

Metrics for evaluating motion smoothness and temporal coherence.
"""

from __future__ import annotations

from typing import Dict, Optional
import numpy as np
from scipy import signal

from kinetic_augment.evaluation.base import BaseMetric, MetricResult


class VelocityMetric(BaseMetric):
    """
    Compute velocity statistics across a sequence.

    Measures motion speed and its distribution to evaluate
    temporal plausibility of augmented data.

    Computes:
    - mean_velocity: Average velocity magnitude
    - max_velocity: Peak velocity
    - velocity_std: Velocity standard deviation
    """

    name = "VelocityMetric"
    description = "Measures motion velocity statistics"

    def __init__(
        self,
        frame_rate: float = 30.0,
        velocity_limit: Optional[float] = None,
    ):
        """
        Initialize the metric.

        Args:
            frame_rate: Frame rate in fps
            velocity_limit: Optional maximum acceptable velocity (m/s)
        """
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate
        self.velocity_limit = velocity_limit

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """
        Compute velocity metrics.

        Args:
            data: Landmarks of shape (frames, landmarks, 3)

        Returns:
            MetricResult with velocity statistics
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
                value=0.0,
                details={'error': 'Need at least 2 frames for velocity'}
            )

        # Compute velocity (first derivative)
        velocity = np.diff(data, axis=0) / self.dt  # (frames-1, landmarks, 3)

        # Velocity magnitudes
        velocity_mag = np.linalg.norm(velocity, axis=2)  # (frames-1, landmarks)

        mean_velocity = np.mean(velocity_mag)
        max_velocity = np.max(velocity_mag)
        velocity_std = np.std(velocity_mag)

        # Per-landmark statistics
        per_landmark_mean = np.mean(velocity_mag, axis=0)

        # Violation rate if limit specified
        violation_rate = 0.0
        if self.velocity_limit is not None:
            violations = velocity_mag > self.velocity_limit
            violation_rate = np.mean(violations)

        # Score: lower velocity variance = smoother motion
        # Normalize by max velocity for scale invariance
        smoothness_score = 1.0 / (1.0 + velocity_std / (mean_velocity + 1e-6))

        return MetricResult(
            name=self.name,
            value=float(smoothness_score),
            details={
                'mean_velocity': float(mean_velocity),
                'max_velocity': float(max_velocity),
                'velocity_std': float(velocity_std),
                'violation_rate': float(violation_rate),
                'fastest_landmark': int(np.argmax(per_landmark_mean)),
            }
        )


class SmoothnessMetric(BaseMetric):
    """
    Measure motion smoothness via jerk analysis.

    Jerk is the third derivative of position (rate of change of acceleration).
    Lower jerk indicates smoother, more natural motion.

    Based on: Balasubramanian et al., "A robust and sensitive metric for
    quantifying movement smoothness" (2012).
    """

    name = "SmoothnessMetric"
    description = "Measures motion smoothness via jerk"

    def __init__(self, frame_rate: float = 30.0):
        """
        Initialize the metric.

        Args:
            frame_rate: Frame rate in fps
        """
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """
        Compute smoothness metrics.

        Args:
            data: Landmarks of shape (frames, landmarks, 3)

        Returns:
            MetricResult with smoothness score
        """
        if data.ndim != 3:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': f'Expected 3D array, got {data.ndim}D'}
            )

        num_frames = data.shape[0]
        if num_frames < 4:
            return MetricResult(
                name=self.name,
                value=1.0,
                details={'note': 'Too few frames for jerk analysis'}
            )

        # Compute derivatives
        velocity = np.diff(data, axis=0) / self.dt
        acceleration = np.diff(velocity, axis=0) / self.dt
        jerk = np.diff(acceleration, axis=0) / self.dt

        # Jerk magnitude
        jerk_mag = np.linalg.norm(jerk, axis=2)

        mean_jerk = np.mean(jerk_mag)
        max_jerk = np.max(jerk_mag)

        # Normalized jerk: dimensionless smoothness metric
        # Lower is smoother
        duration = num_frames * self.dt
        path_length = np.sum(np.linalg.norm(np.diff(data, axis=0), axis=2))

        if path_length > 1e-6:
            normalized_jerk = np.sqrt(
                0.5 * np.sum(jerk_mag ** 2) * (duration ** 5) / (path_length ** 2)
            )
        else:
            normalized_jerk = 0.0

        # Convert to smoothness score (higher = smoother)
        # Use exponential decay
        smoothness_score = np.exp(-normalized_jerk / 100)

        return MetricResult(
            name=self.name,
            value=float(smoothness_score),
            details={
                'mean_jerk': float(mean_jerk),
                'max_jerk': float(max_jerk),
                'normalized_jerk': float(normalized_jerk),
                'duration': float(duration),
                'path_length': float(path_length),
            }
        )


class TemporalCoherenceMetric(BaseMetric):
    """
    Measure temporal coherence of motion.

    Checks that motion flows smoothly between frames without
    sudden jumps or discontinuities.

    Computes:
    - frame_correlation: Correlation between consecutive frames
    - discontinuity_count: Number of sudden motion jumps
    - coherence_score: Overall temporal coherence
    """

    name = "TemporalCoherence"
    description = "Measures temporal continuity and coherence"

    def __init__(
        self,
        jump_threshold: float = 0.1,
        correlation_window: int = 5,
    ):
        """
        Initialize the metric.

        Args:
            jump_threshold: Threshold for detecting motion jumps (relative to mean motion)
            correlation_window: Window size for local correlation
        """
        self.jump_threshold = jump_threshold
        self.correlation_window = correlation_window

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """
        Compute temporal coherence.

        Args:
            data: Landmarks of shape (frames, landmarks, 3)

        Returns:
            MetricResult with coherence score
        """
        if data.ndim != 3:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': f'Expected 3D array, got {data.ndim}D'}
            )

        num_frames = data.shape[0]
        if num_frames < 3:
            return MetricResult(
                name=self.name,
                value=1.0,
                details={'note': 'Too few frames for coherence analysis'}
            )

        # Flatten landmarks for correlation
        flat_data = data.reshape(num_frames, -1)

        # Frame-to-frame correlation
        correlations = []
        for i in range(num_frames - 1):
            corr = np.corrcoef(flat_data[i], flat_data[i + 1])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        mean_correlation = np.mean(correlations) if correlations else 0.0

        # Detect motion discontinuities
        frame_motion = np.linalg.norm(np.diff(data, axis=0), axis=(1, 2))
        mean_motion = np.mean(frame_motion)

        if mean_motion > 1e-6:
            relative_motion = frame_motion / mean_motion
            discontinuities = np.sum(relative_motion > (1 + self.jump_threshold * 10))
        else:
            discontinuities = 0

        # Local smoothness: motion should be locally consistent
        motion_variance = np.var(frame_motion)
        local_smoothness = 1.0 / (1.0 + motion_variance / (mean_motion ** 2 + 1e-6))

        # Combined coherence score
        coherence_score = (
            0.4 * mean_correlation +
            0.3 * (1.0 - discontinuities / max(num_frames - 1, 1)) +
            0.3 * local_smoothness
        )
        coherence_score = max(0.0, min(1.0, coherence_score))

        return MetricResult(
            name=self.name,
            value=float(coherence_score),
            details={
                'mean_correlation': float(mean_correlation),
                'discontinuity_count': int(discontinuities),
                'mean_motion': float(mean_motion),
                'motion_variance': float(motion_variance),
                'local_smoothness': float(local_smoothness),
            }
        )


class SpectralSmoothnessMetric(BaseMetric):
    """
    Measure smoothness via spectral analysis.

    Analyzes frequency content of motion to detect high-frequency
    noise or artifacts from augmentation.
    """

    name = "SpectralSmoothness"
    description = "Measures smoothness via frequency analysis"

    def __init__(
        self,
        frame_rate: float = 30.0,
        high_freq_threshold: float = 0.4,  # Fraction of Nyquist
    ):
        """
        Initialize the metric.

        Args:
            frame_rate: Frame rate in fps
            high_freq_threshold: Threshold for high-frequency content
        """
        self.frame_rate = frame_rate
        self.high_freq_threshold = high_freq_threshold

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """
        Compute spectral smoothness.

        Args:
            data: Landmarks of shape (frames, landmarks, 3)

        Returns:
            MetricResult with spectral smoothness score
        """
        if data.ndim != 3:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': f'Expected 3D array, got {data.ndim}D'}
            )

        num_frames = data.shape[0]
        if num_frames < 8:
            return MetricResult(
                name=self.name,
                value=1.0,
                details={'note': 'Too few frames for spectral analysis'}
            )

        # Flatten and compute FFT
        flat_data = data.reshape(num_frames, -1)

        # Average spectral content across all dimensions
        freqs = np.fft.rfftfreq(num_frames, 1.0 / self.frame_rate)
        nyquist = self.frame_rate / 2

        low_energy = 0.0
        high_energy = 0.0

        for dim in range(flat_data.shape[1]):
            spectrum = np.abs(np.fft.rfft(flat_data[:, dim]))

            # Split energy by frequency
            low_mask = freqs < (self.high_freq_threshold * nyquist)
            high_mask = ~low_mask

            low_energy += np.sum(spectrum[low_mask] ** 2)
            high_energy += np.sum(spectrum[high_mask] ** 2)

        total_energy = low_energy + high_energy
        if total_energy < 1e-10:
            return MetricResult(
                name=self.name,
                value=1.0,
                details={'note': 'No motion detected'}
            )

        # High-frequency ratio
        high_freq_ratio = high_energy / total_energy

        # Smoothness score: lower high-freq content = smoother
        smoothness_score = 1.0 - high_freq_ratio

        return MetricResult(
            name=self.name,
            value=float(smoothness_score),
            details={
                'high_freq_ratio': float(high_freq_ratio),
                'low_energy': float(low_energy),
                'high_energy': float(high_energy),
                'freq_threshold_hz': float(self.high_freq_threshold * nyquist),
            }
        )
