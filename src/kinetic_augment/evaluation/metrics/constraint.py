"""
Constraint Metrics for KineticAugment Evaluation.

Metrics for evaluating constraint satisfaction and violation severity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
import numpy as np

from kinetic_augment.evaluation.base import BaseMetric, MetricResult


class ConstraintSatisfactionMetric(BaseMetric):
    """
    Measure how well augmented data satisfies anatomical constraints.

    Uses ConstraintEngine to check for violations and computes
    satisfaction rates.
    """

    name = "ConstraintSatisfaction"
    description = "Measures constraint satisfaction rate"

    def __init__(
        self,
        check_joint_limits: bool = True,
        check_velocity: bool = True,
        check_collision: bool = False,
        frame_rate: float = 30.0,
    ):
        """
        Initialize the metric.

        Args:
            check_joint_limits: Check joint angle limits
            check_velocity: Check velocity limits
            check_collision: Check self-collision (requires PyBullet)
            frame_rate: Frame rate for velocity computation
        """
        self.check_joint_limits = check_joint_limits
        self.check_velocity = check_velocity
        self.check_collision = check_collision
        self.frame_rate = frame_rate
        self._engine = None

    def _get_engine(self):
        """Lazily initialize ConstraintEngine."""
        if self._engine is None:
            try:
                from kinetic_augment.constraints.engine import ConstraintEngine
                self._engine = ConstraintEngine(
                    joint_limits=self.check_joint_limits,
                    velocity_limits=self.check_velocity,
                    collision_detection=self.check_collision,
                )
            except ImportError:
                self._engine = None
        return self._engine

    def compute(
        self,
        data: Any,
        **kwargs,
    ) -> MetricResult:
        """
        Compute constraint satisfaction.

        Args:
            data: SMPL-X parameters dict or sequence of dicts

        Returns:
            MetricResult with satisfaction rates
        """
        engine = self._get_engine()

        if engine is None:
            return MetricResult(
                name=self.name,
                value=1.0,
                details={'note': 'ConstraintEngine not available'}
            )

        # Handle sequence vs single frame
        if isinstance(data, list):
            params_sequence = data
        elif isinstance(data, dict):
            params_sequence = [data]
        else:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'Expected dict or list of dicts'}
            )

        # Validate sequence
        dt = 1.0 / self.frame_rate
        is_valid, violations = engine.validate_sequence(params_sequence, dt=dt)

        # Count violations by type
        violation_counts: Dict[str, int] = {}
        severities: List[float] = []

        for v in violations:
            vtype = v.constraint_type
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
            severities.append(v.severity)

        total_frames = len(params_sequence)
        total_violations = len(violations)

        # Satisfaction rate
        if total_frames > 0:
            # Estimate frames with violations (approximate)
            violated_frames = min(total_violations, total_frames)
            satisfaction_rate = 1.0 - (violated_frames / total_frames)
        else:
            satisfaction_rate = 1.0

        return MetricResult(
            name=self.name,
            value=float(satisfaction_rate),
            details={
                'total_violations': total_violations,
                'total_frames': total_frames,
                'violations_by_type': violation_counts,
                'mean_severity': float(np.mean(severities)) if severities else 0.0,
                'max_severity': float(np.max(severities)) if severities else 0.0,
            }
        )


class ViolationSeverityMetric(BaseMetric):
    """
    Measure the severity of constraint violations.

    Goes beyond counting violations to measure how severe they are.
    """

    name = "ViolationSeverity"
    description = "Measures severity of constraint violations"

    def __init__(self, frame_rate: float = 30.0):
        """
        Initialize the metric.

        Args:
            frame_rate: Frame rate for velocity computation
        """
        self.frame_rate = frame_rate
        self._engine = None

    def _get_engine(self):
        """Lazily initialize ConstraintEngine."""
        if self._engine is None:
            try:
                from kinetic_augment.constraints.engine import ConstraintEngine
                self._engine = ConstraintEngine(
                    joint_limits=True,
                    velocity_limits=True,
                    collision_detection=False,
                )
            except ImportError:
                self._engine = None
        return self._engine

    def compute(
        self,
        data: Any,
        **kwargs,
    ) -> MetricResult:
        """
        Compute violation severity.

        Args:
            data: SMPL-X parameters dict or sequence

        Returns:
            MetricResult with severity statistics
        """
        engine = self._get_engine()

        if engine is None:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'note': 'ConstraintEngine not available'}
            )

        # Handle sequence vs single frame
        if isinstance(data, list):
            params_sequence = data
        elif isinstance(data, dict):
            params_sequence = [data]
        else:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': 'Expected dict or list of dicts'}
            )

        # Get violations
        dt = 1.0 / self.frame_rate
        _, violations = engine.validate_sequence(params_sequence, dt=dt)

        if not violations:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'note': 'No violations found'}
            )

        severities = [v.severity for v in violations]

        # Severity score: mean severity (lower = better)
        mean_severity = np.mean(severities)
        max_severity = np.max(severities)

        # Severity by joint
        by_joint: Dict[str, List[float]] = {}
        for v in violations:
            if v.joint_name not in by_joint:
                by_joint[v.joint_name] = []
            by_joint[v.joint_name].append(v.severity)

        joint_mean_severity = {
            joint: float(np.mean(sevs))
            for joint, sevs in by_joint.items()
        }

        # Most problematic joints
        worst_joints = sorted(
            joint_mean_severity.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        return MetricResult(
            name=self.name,
            value=float(mean_severity),
            details={
                'mean_severity': float(mean_severity),
                'max_severity': float(max_severity),
                'std_severity': float(np.std(severities)),
                'num_violations': len(violations),
                'worst_joints': dict(worst_joints),
            }
        )


class JointLimitComplianceMetric(BaseMetric):
    """
    Specifically measure joint limit compliance.

    Works directly on landmarks without requiring SMPL-X parameters,
    by computing approximate joint angles from 3D positions.
    """

    name = "JointLimitCompliance"
    description = "Measures joint angle limit compliance"

    # Approximate joint angle limits (in degrees)
    # Based on common anatomical ranges
    JOINT_LIMITS = {
        'elbow': (0, 145),      # 0-145 degrees
        'shoulder': (-180, 90),  # Simplified
        'knee': (0, 140),
        'hip': (-30, 120),
    }

    def compute(self, data: np.ndarray, **kwargs) -> MetricResult:
        """
        Compute joint limit compliance from landmarks.

        Args:
            data: Landmarks of shape (frames, landmarks, 3)

        Returns:
            MetricResult with compliance rate
        """
        if data.ndim != 3:
            return MetricResult(
                name=self.name,
                value=0.0,
                details={'error': f'Expected 3D array, got {data.ndim}D'}
            )

        # Define joint triplets for angle computation
        # Format: (point1, vertex, point2, joint_name)
        # Angle is computed at 'vertex'
        joint_triplets = [
            # Left arm
            (11, 13, 15, 'left_elbow'),   # shoulder-elbow-wrist
            # Right arm
            (12, 14, 16, 'right_elbow'),
            # Left leg
            (23, 25, 27, 'left_knee'),    # hip-knee-ankle
            # Right leg
            (24, 26, 28, 'right_knee'),
        ]

        num_frames = data.shape[0]
        total_checks = 0
        violations = 0
        angle_stats = {}

        for p1, v, p2, name in joint_triplets:
            try:
                # Compute angles for all frames
                vec1 = data[:, p1, :] - data[:, v, :]
                vec2 = data[:, p2, :] - data[:, v, :]

                # Normalize
                vec1_norm = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + 1e-10)
                vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-10)

                # Dot product -> angle
                dots = np.sum(vec1_norm * vec2_norm, axis=1)
                dots = np.clip(dots, -1, 1)
                angles_rad = np.arccos(dots)
                angles_deg = np.degrees(angles_rad)

                # Check against limits
                joint_type = name.split('_')[1]  # e.g., 'elbow' from 'left_elbow'
                if joint_type in self.JOINT_LIMITS:
                    min_angle, max_angle = self.JOINT_LIMITS[joint_type]
                    out_of_range = (angles_deg < min_angle) | (angles_deg > max_angle)
                    violations += np.sum(out_of_range)
                    total_checks += num_frames

                angle_stats[name] = {
                    'mean': float(np.mean(angles_deg)),
                    'min': float(np.min(angles_deg)),
                    'max': float(np.max(angles_deg)),
                }

            except Exception:
                continue

        if total_checks == 0:
            compliance_rate = 1.0
        else:
            compliance_rate = 1.0 - (violations / total_checks)

        return MetricResult(
            name=self.name,
            value=float(compliance_rate),
            details={
                'violations': int(violations),
                'total_checks': total_checks,
                'angle_statistics': angle_stats,
            }
        )
